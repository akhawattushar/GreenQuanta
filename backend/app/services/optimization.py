"""Fleet optimisation.

Two solvers are implemented here in full:

* ``nsga2``   — NSGA-II: fast non-dominated sorting, crowding distance,
  binary-tournament selection, SBX crossover and polynomial mutation.
* ``quantum`` — QIEA (quantum-inspired evolutionary algorithm): a population of
  qubit probability vectors collapsed into candidate solutions each generation
  and updated with a rotation gate toward the incumbent best.

QIEA runs on classical hardware. It is *inspired by* quantum computation and
this project makes **no claim of quantum advantage** — the response reports the
measured objective values and wall-clock runtime of both solvers on the same
problem and nothing more.

Decision variables: sailing speed (continuous), fuel type (categorical), shore
power (binary). Objectives: minimise cost and minimise GHG, subject to an ETA
limit and optional cost/emission caps.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.evaluator import (
    CO2E_TONNES_PER_TONNE_FUEL,
    Evaluation,
    SUPPORTED_FUELS,
    Environment,
    EvaluationError,
    VesselState,
    VoyagePlan,
    assumptions_block,
    evaluate_many,
)
from app.services.model_registry import ModelUnavailableError, get_predictor

#: Speeds outside the training range would be extrapolation, so the search is
#: clamped to the interval observed in the shipped dataset.
DEFAULT_SPEED_BOUNDS = (12.0, 19.0)
KNOT_IN_M_S = 0.514444

ALGORITHMS = ("nsga2", "quantum")
ALGORITHM_LABELS = {
    "nsga2": "GA / NSGA-II",
    "quantum": "Quantum-Inspired (QIEA)",
    "baseline": "Baseline (no optimisation)",
}


@dataclass
class OptimizationProblem:
    distance_nm: float
    vessel: VesselState
    environment: Environment | None
    available_fuels: list
    model_id: str = "legacy"
    fuelcast_inputs: dict | None = None
    predictor: object = field(default=None, repr=False)
    model_run_id: str | None = None
    fuel_price_usd_per_tonne: float | None = None
    allow_shore_power: bool = True
    speed_bounds: tuple = DEFAULT_SPEED_BOUNDS
    max_eta_hours: float | None = None
    max_ghg_tonnes: float | None = None
    max_cost_inr: float | None = None
    cost_weight: float = 0.5
    population_size: int = 40
    generations: int = 60
    seed: int = 42
    #: Fixed objective normalisers so the scalar score is comparable across
    #: generations and across solvers. Set once from the baseline plan.
    cost_ref: float = 0.0
    ghg_ref: float = 0.0

    def validate(self) -> None:
        if not self.available_fuels:
            raise EvaluationError("Select at least one available fuel.")
        unknown = [f for f in self.available_fuels if f not in SUPPORTED_FUELS]
        if unknown:
            raise EvaluationError(
                f"Unsupported fuel(s): {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_FUELS)}."
            )
        low, high = self.speed_bounds
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low:
            raise EvaluationError("Invalid speed bounds.")
        if not math.isfinite(self.distance_nm) or self.distance_nm <= 0:
            raise EvaluationError("Distance must be greater than zero.")
        if not 0.0 <= self.cost_weight <= 1.0:
            raise EvaluationError("cost_weight must be between 0 and 1.")
        if (self.fuel_price_usd_per_tonne is not None
                and (not math.isfinite(self.fuel_price_usd_per_tonne)
                     or self.fuel_price_usd_per_tonne <= 0)):
            raise EvaluationError("Scenario fuel price must be finite and positive.")
        if self.model_id == "fuelcast_xgboost":
            if self.fuelcast_inputs is None:
                raise EvaluationError("FuelCast optimization requires explicit fuelcast_inputs.")
            if not low <= self.fuelcast_inputs["speed_over_ground"] <= high:
                raise EvaluationError("FuelCast reference speed must lie within the m/s speed bounds.")
            if self.available_fuels != ["Marine Diesel"] or self.allow_shore_power:
                raise EvaluationError("FuelCast optimization requires Marine Diesel only and shore power disabled.")
        elif self.model_id != "legacy":
            raise EvaluationError(f"Unknown model ID {self.model_id!r}.")
        elif self.environment is None:
            raise EvaluationError("Legacy optimization requires environment.")


@dataclass
class Candidate:
    speed: float
    fuel_index: int
    shore_power: bool
    objectives: tuple = (math.inf, math.inf)
    scalar: float = math.inf
    evaluation: object = None
    rank: int = 0
    crowding: float = 0.0
    provenance: dict = field(default_factory=dict)


@dataclass
class SolverResult:
    algorithm: str
    label: str
    best: Candidate | None
    front: list = field(default_factory=list)
    convergence: list = field(default_factory=list)
    runtime_seconds: float = 0.0
    evaluations: int = 0


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _plan(problem: OptimizationProblem, cand: Candidate) -> VoyagePlan:
    return VoyagePlan(
        distance_nm=problem.distance_nm,
        speed_knots=_candidate_speed_knots(problem, cand.speed),
        vessel=problem.vessel,
        environment=problem.environment,
        fuel_type=problem.available_fuels[cand.fuel_index],
        shore_power=cand.shore_power,
    )


def _candidate_speed_knots(problem: OptimizationProblem, speed: float) -> float:
    """FuelCast candidates are explicit m/s over ground; legacy candidates are knots."""
    if not math.isfinite(speed) or speed <= 0:
        raise EvaluationError("Candidate speed must be finite and positive.")
    return speed / KNOT_IN_M_S if problem.model_id == "fuelcast_xgboost" else speed


def _fuelcast_evaluation(problem: OptimizationProblem, cand: Candidate, plan: VoyagePlan) -> Evaluation | None:
    """Score one candidate; bad rates invalidate it without changing other candidates."""
    duration = plan.distance_nm / plan.speed_knots
    if not math.isfinite(duration) or duration <= 0:
        return None
    row = dict(problem.fuelcast_inputs)
    row["speed_over_ground"] = cand.speed
    try:
        prediction = problem.predictor.predict([row])
    except ModelUnavailableError as exc:
        if "non-finite" in str(exc) or "non-numeric fuel rate" in str(exc):
            return None
        raise
    rates = prediction.get("fuel_rates", [])
    if prediction.get("fuel_rate_unit") != "kg/s" or len(rates) != 1:
        raise ModelUnavailableError("FuelCast candidate must return one kg/s fuel rate.")
    rate = rates[0]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
        return None
    fuel_tonnes = rate * duration * 3.6
    if not math.isfinite(fuel_tonnes) or fuel_tonnes < 0:
        return None
    settings = get_settings()
    price = (settings.fuel_price_usd_per_tonne if problem.fuel_price_usd_per_tonne is None
             else problem.fuel_price_usd_per_tonne)
    cost_usd = fuel_tonnes * price
    cost_inr = cost_usd * settings.usd_to_inr
    ghg = fuel_tonnes * CO2E_TONNES_PER_TONNE_FUEL["Marine Diesel"]
    if not all(math.isfinite(value) for value in (cost_usd, cost_inr, ghg)):
        return None
    violations = []
    if problem.max_eta_hours is not None and duration > problem.max_eta_hours:
        violations.append(f"ETA {duration:.1f} h exceeds the {problem.max_eta_hours:.1f} h limit.")
    if problem.max_ghg_tonnes is not None and ghg > problem.max_ghg_tonnes:
        violations.append(f"GHG {ghg:.1f} t exceeds the {problem.max_ghg_tonnes:.1f} t cap.")
    if problem.max_cost_inr is not None and cost_inr > problem.max_cost_inr:
        violations.append(f"Cost exceeds the configured budget.")
    cand.provenance = {
        "model_id": "fuelcast_xgboost", "model_run_id": problem.model_run_id,
        "candidate_speed": cand.speed, "candidate_speed_unit": "m/s",
        "speed_over_ground_m_s": cand.speed, "raw_prediction": rate,
        "raw_prediction_unit": "kg/s", "conversion_duration_hours": duration,
        "normalized_voyage_fuel_tonnes": fuel_tonnes,
        "normalized_voyage_fuel_unit": "tonnes",
        "fixed_environment_snapshot": {key: value for key, value in row.items()
                                       if key != "speed_over_ground"},
    }
    return Evaluation(
        fuel_rate=rate, fuel_rate_unit="kg/s", duration_hours=duration,
        main_engine_fuel_tonnes=fuel_tonnes, auxiliary_fuel_tonnes=0.0,
        fuel_tonnes=fuel_tonnes, cost_usd=cost_usd, cost_inr=cost_inr,
        ghg_tonnes_co2e=ghg, feasible=not violations, violations=violations,
    )


def _evaluate_candidates(problem: OptimizationProblem, population: list[Candidate]) -> list[Evaluation | None]:
    plans = [_plan(problem, cand) for cand in population]
    if problem.model_id == "legacy":
        return evaluate_many(
            plans, max_eta_hours=problem.max_eta_hours,
            max_ghg_tonnes=problem.max_ghg_tonnes, max_cost_inr=problem.max_cost_inr,
            fuel_price_usd_per_tonne=problem.fuel_price_usd_per_tonne,
        )
    return [_fuelcast_evaluation(problem, cand, plan) for cand, plan in zip(population, plans)]


def _score(problem: OptimizationProblem, population: list) -> int:
    """Score a population through its selected prediction model."""
    evaluations = _evaluate_candidates(problem, population)
    valid = [ev for ev in evaluations if ev is not None]
    if not valid:
        raise EvaluationError("No valid optimization candidates remain.")
    # Normalisers are fixed for the whole run (see `calibrate`), so scalar
    # scores are comparable across generations and between solvers.
    if problem.cost_ref <= 0 or problem.ghg_ref <= 0:
        calibrate(problem, valid)
    cost_ref = max(problem.cost_ref, 1e-9)
    ghg_ref = max(problem.ghg_ref, 1e-9)

    for cand, ev in zip(population, evaluations):
        if ev is None:
            cand.evaluation = None
            cand.objectives = (1e300, 1e300)
            cand.scalar = 1e300
            continue
        penalty = 1.0 + 0.5 * len(ev.violations)
        cand.evaluation = ev
        cand.objectives = (ev.cost_usd * penalty, ev.ghg_tonnes_co2e * penalty)
        cand.scalar = (
            problem.cost_weight * (ev.cost_usd / cost_ref)
            + (1.0 - problem.cost_weight) * (ev.ghg_tonnes_co2e / ghg_ref)
        ) * penalty
    return len(population)


def calibrate(problem: OptimizationProblem, evaluations) -> None:
    """Fix the objective normalisers once per problem."""
    problem.cost_ref = max((e.cost_usd for e in evaluations), default=1.0) or 1.0
    problem.ghg_ref = max((e.ghg_tonnes_co2e for e in evaluations), default=1.0) or 1.0


def _dominates(a: Candidate, b: Candidate) -> bool:
    a_feasible = bool(a.evaluation and a.evaluation.feasible)
    b_feasible = bool(b.evaluation and b.evaluation.feasible)
    if a_feasible != b_feasible:
        return a_feasible
    better_or_equal = all(x <= y for x, y in zip(a.objectives, b.objectives))
    strictly_better = any(x < y for x, y in zip(a.objectives, b.objectives))
    return better_or_equal and strictly_better


def _pick_best(population: list) -> Candidate:
    feasible = [c for c in population if c.evaluation and c.evaluation.feasible]
    pool = feasible or population
    return min(pool, key=lambda c: c.scalar)


def _random_candidate(problem: OptimizationProblem, rng: random.Random) -> Candidate:
    low, high = problem.speed_bounds
    return Candidate(
        speed=rng.uniform(low, high),
        fuel_index=rng.randrange(len(problem.available_fuels)),
        shore_power=problem.allow_shore_power and rng.random() < 0.5,
    )


# --------------------------------------------------------------------------
# NSGA-II
# --------------------------------------------------------------------------
def _fast_non_dominated_sort(population: list) -> list:
    fronts: list = [[]]
    dominated: dict = {id(c): [] for c in population}
    counts: dict = {id(c): 0 for c in population}
    for p in population:
        for q in population:
            if p is q:
                continue
            if _dominates(p, q):
                dominated[id(p)].append(q)
            elif _dominates(q, p):
                counts[id(p)] += 1
        if counts[id(p)] == 0:
            p.rank = 0
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: list = []
        for p in fronts[i]:
            for q in dominated[id(p)]:
                counts[id(q)] -= 1
                if counts[id(q)] == 0:
                    q.rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def _crowding_distance(front: list) -> None:
    for c in front:
        c.crowding = 0.0
    n_obj = len(front[0].objectives)
    for m in range(n_obj):
        front.sort(key=lambda c: c.objectives[m])
        front[0].crowding = front[-1].crowding = math.inf
        lo, hi = front[0].objectives[m], front[-1].objectives[m]
        span = hi - lo
        if span <= 0:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / span


def _tournament(population: list, rng: random.Random) -> Candidate:
    a, b = rng.choice(population), rng.choice(population)
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding > b.crowding else b


def _sbx(p1: float, p2: float, low: float, high: float, rng: random.Random, eta: float = 15.0) -> float:
    if abs(p1 - p2) < 1e-12:
        return p1
    u = rng.random()
    beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    return min(max(child, low), high)


def _polynomial_mutation(value: float, low: float, high: float, rng: random.Random, eta: float = 20.0) -> float:
    span = high - low
    if span <= 0:
        return value
    u = rng.random()
    delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
    return min(max(value + delta * span, low), high)


def solve_nsga2(problem: OptimizationProblem) -> SolverResult:
    rng = random.Random(problem.seed)
    low, high = problem.speed_bounds
    started = time.perf_counter()

    population = [_random_candidate(problem, rng) for _ in range(problem.population_size)]
    n_eval = _score(problem, population)
    convergence: list = []

    for generation in range(problem.generations):
        fronts = _fast_non_dominated_sort(population)
        for front in fronts:
            _crowding_distance(front)

        offspring: list = []
        while len(offspring) < problem.population_size:
            p1, p2 = _tournament(population, rng), _tournament(population, rng)
            speed = _sbx(p1.speed, p2.speed, low, high, rng)
            if rng.random() < 0.2:
                speed = _polynomial_mutation(speed, low, high, rng)
            fuel = p1.fuel_index if rng.random() < 0.5 else p2.fuel_index
            if rng.random() < 0.15:
                fuel = rng.randrange(len(problem.available_fuels))
            shore = p1.shore_power if rng.random() < 0.5 else p2.shore_power
            if problem.allow_shore_power and rng.random() < 0.15:
                shore = not shore
            offspring.append(Candidate(speed=speed, fuel_index=fuel, shore_power=bool(shore and problem.allow_shore_power)))

        n_eval += _score(problem, offspring)

        merged = population + offspring
        fronts = _fast_non_dominated_sort(merged)
        survivors: list = []
        for front in fronts:
            _crowding_distance(front)
            if len(survivors) + len(front) <= problem.population_size:
                survivors.extend(front)
            else:
                front.sort(key=lambda c: (-c.crowding, c.scalar))
                survivors.extend(front[: problem.population_size - len(survivors)])
                break
        population = survivors
        convergence.append({"iteration": generation, "objective": round(_pick_best(population).scalar, 6)})

    fronts = _fast_non_dominated_sort(population)
    return SolverResult(
        algorithm="nsga2",
        label=ALGORITHM_LABELS["nsga2"],
        best=_pick_best(population),
        front=sorted(fronts[0], key=lambda c: c.objectives[0]) if fronts else [],
        convergence=convergence,
        runtime_seconds=round(time.perf_counter() - started, 4),
        evaluations=n_eval,
    )


# --------------------------------------------------------------------------
# Quantum-inspired evolutionary algorithm (classical simulation)
# --------------------------------------------------------------------------
@dataclass
class _Qubit:
    """|alpha|^2 is the probability of collapsing to 0, |beta|^2 to 1."""

    theta: float = math.pi / 4  # equal superposition

    @property
    def p_one(self) -> float:
        return math.sin(self.theta) ** 2

    def rotate(self, toward_one: bool, magnitude: float) -> None:
        self.theta += magnitude if toward_one else -magnitude
        self.theta = min(max(self.theta, 0.02), math.pi / 2 - 0.02)


SPEED_BITS = 8


def _collapse(register: list, rng: random.Random) -> list:
    return [1 if rng.random() < q.p_one else 0 for q in register]


def _decode(bits: list, problem: OptimizationProblem) -> Candidate:
    low, high = problem.speed_bounds
    speed_bits = bits[:SPEED_BITS]
    raw = sum(bit << i for i, bit in enumerate(speed_bits))
    speed = low + (high - low) * raw / (2**SPEED_BITS - 1)

    n_fuels = len(problem.available_fuels)
    fuel_bit_count = max(1, math.ceil(math.log2(max(n_fuels, 2))))
    fuel_bits = bits[SPEED_BITS : SPEED_BITS + fuel_bit_count]
    fuel_index = sum(bit << i for i, bit in enumerate(fuel_bits)) % n_fuels

    shore = bool(bits[SPEED_BITS + fuel_bit_count]) and problem.allow_shore_power
    return Candidate(speed=speed, fuel_index=fuel_index, shore_power=shore)


def solve_quantum_inspired(problem: OptimizationProblem) -> SolverResult:
    """QIEA. Classical simulation — no quantum hardware, no advantage claimed."""
    rng = random.Random(problem.seed + 1)
    started = time.perf_counter()

    n_fuels = len(problem.available_fuels)
    register_len = SPEED_BITS + max(1, math.ceil(math.log2(max(n_fuels, 2)))) + 1
    registers = [[_Qubit() for _ in range(register_len)] for _ in range(problem.population_size)]

    best: Candidate | None = None
    best_bits: list | None = None
    convergence: list = []
    n_eval = 0

    for generation in range(problem.generations):
        collapsed = [_collapse(r, rng) for r in registers]
        population = [_decode(bits, problem) for bits in collapsed]
        n_eval += _score(problem, population)

        generation_best = _pick_best(population)
        if best is None or (
            (generation_best.evaluation.feasible, -generation_best.scalar)
            > (best.evaluation.feasible, -best.scalar)
        ):
            best = generation_best
            best_bits = next(
                bits for bits, cand in zip(collapsed, population) if cand is generation_best
            )

        # Rotation gate: nudge each qubit toward the incumbent best solution,
        # with the step decaying over the run.
        magnitude = 0.05 * math.pi * (1.0 - generation / max(problem.generations, 1)) + 0.005 * math.pi
        if best_bits is not None:
            for register, bits, cand in zip(registers, collapsed, population):
                worse = cand.scalar > best.scalar
                for qubit, bit, target in zip(register, bits, best_bits):
                    if worse and bit != target:
                        qubit.rotate(toward_one=bool(target), magnitude=magnitude)
                    elif bit == target:
                        qubit.rotate(toward_one=bool(target), magnitude=magnitude * 0.25)

        convergence.append({"iteration": generation, "objective": round(best.scalar, 6)})

    front = [best] if best else []
    return SolverResult(
        algorithm="quantum",
        label=ALGORITHM_LABELS["quantum"],
        best=best,
        front=front,
        convergence=convergence,
        runtime_seconds=round(time.perf_counter() - started, 4),
        evaluations=n_eval,
    )


SOLVERS = {"nsga2": solve_nsga2, "quantum": solve_quantum_inspired}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def _candidate_to_plan_dict(
    problem: OptimizationProblem, cand: Candidate, label: str, algorithm_id: str | None = None,
) -> dict:
    ev = cand.evaluation
    plan = {
        "label": label,
        "vessel_type": problem.vessel.vessel_type,
        "speed_knots": round(_candidate_speed_knots(problem, cand.speed), 2),
        "fuel_type": problem.available_fuels[cand.fuel_index],
        "shore_power": bool(cand.shore_power),
        "fuel_rate": ev.fuel_rate,
        "fuel_rate_unit": ev.fuel_rate_unit,
        "fuel_tonnes": ev.fuel_tonnes,
        "cost_usd": ev.cost_usd,
        "cost_inr": ev.cost_inr,
        "cost_inr_lakh": round(ev.cost_inr / 1e5, 3),
        "ghg_tonnes_co2e": ev.ghg_tonnes_co2e,
        "eta_hours": ev.duration_hours,
        "feasible": ev.feasible,
        "violations": list(ev.violations),
        "objective": round(cand.scalar, 6),
    }
    if problem.model_id == "fuelcast_xgboost":
        plan.update(cand.provenance)
        plan["algorithm_id"] = algorithm_id
    return plan


def baseline_plan(problem: OptimizationProblem) -> dict:
    """Un-optimised reference on the first fuel, with no shore power."""
    speeds = [problem.speed_bounds[1]]
    if problem.model_id == "fuelcast_xgboost":
        low, high = problem.speed_bounds
        speeds = [problem.fuelcast_inputs["speed_over_ground"], high, (low + high) / 2, low]
    cand = None
    evaluation = None
    for speed in speeds:
        candidate = Candidate(speed=speed, fuel_index=0, shore_power=False)
        evaluation = _evaluate_candidates(problem, [candidate])[0]
        if evaluation is not None:
            cand = candidate
            break
    if cand is None:
        raise EvaluationError("No valid optimization candidates remain.")
    if problem.cost_ref <= 0 or problem.ghg_ref <= 0:
        calibrate(problem, [evaluation])
    cand.evaluation = evaluation
    cand.objectives = (evaluation.cost_usd, evaluation.ghg_tonnes_co2e)
    cand.scalar = (
        problem.cost_weight * (evaluation.cost_usd / max(problem.cost_ref, 1e-9))
        + (1.0 - problem.cost_weight) * (evaluation.ghg_tonnes_co2e / max(problem.ghg_ref, 1e-9))
    ) * (1.0 + 0.5 * len(evaluation.violations))
    return _candidate_to_plan_dict(problem, cand, ALGORITHM_LABELS["baseline"], "baseline")


def run_optimization(problem: OptimizationProblem, algorithms=("nsga2", "quantum")) -> dict:
    """Run the requested solvers on one problem and return a comparable report."""
    problem.validate()
    requested = [a for a in algorithms if a in SOLVERS]
    if not requested:
        raise EvaluationError(f"No known algorithm requested. Available: {', '.join(SOLVERS)}.")
    if problem.model_id == "fuelcast_xgboost":
        if problem.predictor is None:
            problem.predictor = get_predictor("fuelcast_xgboost")
        problem.model_run_id = problem.predictor.metadata().get("run_id")

    # Calibrate the objective normalisers on the baseline plan first so every
    # solver optimises exactly the same scalar function.
    base = baseline_plan(problem)

    results = {name: SOLVERS[name](problem) for name in requested}

    plans: list = []
    for name in requested:
        result = results[name]
        if result.best is not None:
            plans.append(_candidate_to_plan_dict(problem, result.best, result.label, name))
    plans.append(base)
    plans.sort(key=lambda p: (not p["feasible"], p["objective"]))

    # Convergence curves aligned on iteration index for the comparison chart.
    max_len = max((len(r.convergence) for r in results.values()), default=0)
    convergence: list = []
    for i in range(max_len):
        point = {"iteration": i}
        for name in requested:
            series = results[name].convergence
            if i < len(series):
                point[name] = series[i]["objective"]
        convergence.append(point)

    comparison: list = []
    if len(requested) > 1 or requested:
        def _metric(row_label: str, getter) -> dict:
            row = {"metric": row_label}
            for name in requested:
                row[name] = getter(results[name])
            return row

        comparison = [
            _metric("Best objective (normalised)", lambda r: round(r.best.scalar, 6) if r.best else None),
            _metric("Best cost (USD)", lambda r: r.best.evaluation.cost_usd if r.best else None),
            _metric("Best GHG (t CO2e)", lambda r: r.best.evaluation.ghg_tonnes_co2e if r.best else None),
            _metric("Measured runtime (s)", lambda r: r.runtime_seconds),
            _metric("Model evaluations", lambda r: r.evaluations),
            _metric("Feasible plan found", lambda r: bool(r.best and r.best.evaluation.feasible)),
        ]

    best_plan = plans[0] if plans else None
    improvement = None
    if best_plan and base["cost_usd"] > 0:
        improvement = {
            "vs_baseline_cost_pct": round((base["cost_usd"] - best_plan["cost_usd"]) / base["cost_usd"] * 100, 2),
            "vs_baseline_ghg_pct": (
                round((base["ghg_tonnes_co2e"] - best_plan["ghg_tonnes_co2e"]) / base["ghg_tonnes_co2e"] * 100, 2)
                if base["ghg_tonnes_co2e"] > 0
                else None
            ),
            "note": "Measured against the baseline plan defined above, not against any published benchmark.",
        }

    report = {
        "algorithms_run": requested,
        "algorithm_labels": {name: ALGORITHM_LABELS[name] for name in requested},
        "plans": plans,
        "best_plan": best_plan,
        "pareto_front": [
            _candidate_to_plan_dict(problem, c, results[name].label, name)
            for name in requested
            for c in results[name].front[:12]
            if c.evaluation is not None
        ],
        "convergence": convergence,
        "comparison": comparison,
        "improvement": improvement,
        "runtime_seconds": {name: results[name].runtime_seconds for name in requested},
        "settings": {
            "population_size": problem.population_size,
            "generations": problem.generations,
            "seed": problem.seed,
            "speed_bounds": list(problem.speed_bounds),
            "cost_weight": problem.cost_weight,
        },
        "assumptions": assumptions_block(),
        "disclaimer": (
            "Both solvers are classical. The quantum-inspired solver (QIEA) simulates qubit "
            "probability amplitudes on a CPU; runtimes and objectives shown are measured on this "
            "machine for this problem only and are not evidence of quantum advantage."
        ),
    }
    if problem.model_id == "fuelcast_xgboost":
        settings = get_settings()
        report["model_id"] = "fuelcast_xgboost"
        report["model_run_id"] = problem.model_run_id
        report["fixed_environment_snapshot"] = {
            key: value for key, value in problem.fuelcast_inputs.items()
            if key != "speed_over_ground"
        }
        report["settings"]["speed_bounds_unit"] = "m/s over ground"
        report["assumptions"] = {
            "base_fuel_price_usd_per_tonne": (
                settings.fuel_price_usd_per_tonne if problem.fuel_price_usd_per_tonne is None
                else problem.fuel_price_usd_per_tonne
            ),
            "usd_to_inr": settings.usd_to_inr,
            "co2e_tonnes_per_tonne_fuel": {
                "Marine Diesel": CO2E_TONNES_PER_TONNE_FUEL["Marine Diesel"],
            },
            "note": ("FuelCast kg/s is converted once to voyage tonnes. Cost and tank-to-wake "
                     "emissions use Marine Diesel factors; no fuel-type, auxiliary, or shore-power "
                     "adjustment is applied."),
        }
        report["disclaimer"] += (
            " FuelCast uses one fixed environmental snapshot for every candidate; "
            "this prototype is not physically validated."
        )
    return report
