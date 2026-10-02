"""Scenario analysis.

Each scenario is a named set of modifiers applied to a baseline voyage. The
modified voyage is re-optimised (or re-evaluated) through exactly the same
evaluator and the same trained model, so scenarios are directly comparable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math

from app.core.config import get_settings
from app.services.evaluator import (
    CO2E_TONNES_PER_TONNE_FUEL,
    Environment,
    EvaluationError,
    VesselState,
    VoyagePlan,
    assumptions_block,
    evaluate,
)
from app.services.model_registry import ModelUnavailableError, get_predictor
from app.services.optimization import KNOT_IN_M_S, OptimizationProblem, run_optimization


@dataclass(frozen=True)
class ScenarioDefinition:
    key: str
    label: str
    description: str
    fuel_price_multiplier: float = 1.0
    wind_speed_delta: float = 0.0
    wave_height_delta: float = 0.0
    wave_period_delta: float = 0.0
    current_speed_delta: float = 0.0
    displacement_multiplier: float = 1.0
    max_ghg_multiplier: float | None = None


#: Mirrors the scenarios named in the project's application-flow diagram.
SCENARIO_CATALOG = {
    "base": ScenarioDefinition(
        key="base",
        label="Base Scenario",
        description="The voyage exactly as entered, with no modifiers applied.",
    ),
    "high_fuel_price": ScenarioDefinition(
        key="high_fuel_price",
        label="High Fuel Price",
        description="Fuel price raised 40%. Physical conditions unchanged.",
        fuel_price_multiplier=1.40,
    ),
    "severe_weather": ScenarioDefinition(
        key="severe_weather",
        label="Severe Weather",
        description="Wind, wave height/period and current increased toward the upper training range.",
        wind_speed_delta=8.0,
        wave_height_delta=2.0,
        wave_period_delta=2.0,
        current_speed_delta=0.3,
    ),
    "high_cargo_demand": ScenarioDefinition(
        key="high_cargo_demand",
        label="High Cargo Demand",
        description="Displacement raised 15% to represent a heavier load.",
        displacement_multiplier=1.15,
    ),
    "strict_emission_regulation": ScenarioDefinition(
        key="strict_emission_regulation",
        label="Strict Emission Regulation",
        description="A GHG cap set to 70% of the baseline plan's emissions.",
        max_ghg_multiplier=0.70,
    ),
    "fuelcast_speed_change": ScenarioDefinition(
        key="fuelcast_speed_change",
        label="FuelCast Speed Change",
        description="Explicit speed-over-ground change in m/s supplied as a full FuelCast snapshot.",
    ),
}


def list_scenarios() -> list:
    return [
        {"key": s.key, "label": s.label, "description": s.description}
        for s in SCENARIO_CATALOG.values()
    ]


def _apply(definition: ScenarioDefinition, environment: Environment, vessel: VesselState):
    env = replace(
        environment,
        wind_speed=max(0.0, environment.wind_speed + definition.wind_speed_delta),
        combined_wave_height=max(0.0, environment.combined_wave_height + definition.wave_height_delta),
        combined_wave_period=max(0.0, environment.combined_wave_period + definition.wave_period_delta),
        sea_current_speed=max(0.0, environment.sea_current_speed + definition.current_speed_delta),
    )
    vsl = replace(vessel, displacement=vessel.displacement * definition.displacement_multiplier)
    return env, vsl


def run_scenarios(
    *,
    scenario_keys,
    distance_nm: float,
    vessel: VesselState,
    environment: Environment | None,
    available_fuels,
    model_id: str = "legacy",
    fuelcast_inputs: dict | None = None,
    fuelcast_scenario_inputs: dict[str, dict] | None = None,
    fuelcast_speed_bounds_m_s: tuple[float, float] | None = None,
    speed_knots: float | None = None,
    max_eta_hours: float | None = None,
    optimize: bool = True,
    population_size: int = 24,
    generations: int = 25,
    seed: int = 42,
) -> dict:
    """Evaluate (or optimise) the voyage under each requested scenario."""
    unknown = [k for k in scenario_keys if k not in SCENARIO_CATALOG]
    if unknown:
        raise EvaluationError(
            f"Unknown scenario(s): {', '.join(unknown)}. Available: {', '.join(SCENARIO_CATALOG)}."
        )
    if not scenario_keys:
        raise EvaluationError("Select at least one scenario.")
    if model_id == "fuelcast_xgboost":
        return _run_fuelcast_scenarios(
            scenario_keys=scenario_keys, distance_nm=distance_nm, vessel=vessel,
            available_fuels=available_fuels, fuelcast_inputs=fuelcast_inputs,
            fuelcast_scenario_inputs=fuelcast_scenario_inputs or {},
            fuelcast_speed_bounds_m_s=fuelcast_speed_bounds_m_s,
            max_eta_hours=max_eta_hours, optimize=optimize,
            population_size=population_size, generations=generations, seed=seed,
        )
    if model_id != "legacy":
        raise EvaluationError(f"Unknown model ID {model_id!r}.")
    if "fuelcast_speed_change" in scenario_keys:
        raise EvaluationError("fuelcast_speed_change requires fuelcast_xgboost.")
    if not optimize and (speed_knots is None or speed_knots <= 0):
        raise EvaluationError("speed_knots is required when optimisation is disabled.")

    settings = get_settings()
    base_price = settings.fuel_price_usd_per_tonne

    # The baseline result sets the reference for the emission-cap scenario.
    base_env, base_vessel = _apply(SCENARIO_CATALOG["base"], environment, vessel)
    baseline_ghg: float | None = None

    rows: list = []
    ordered = sorted(scenario_keys, key=lambda k: 0 if k == "base" else 1)
    if "strict_emission_regulation" in ordered and "base" not in ordered:
        ordered = ["base", *ordered]

    for key in ordered:
        definition = SCENARIO_CATALOG[key]
        env, vsl = _apply(definition, environment, vessel)

        cap = None
        if definition.max_ghg_multiplier is not None:
            if baseline_ghg is None:
                raise EvaluationError("The base scenario must run before the emission-cap scenario.")
            cap = baseline_ghg * definition.max_ghg_multiplier

        scenario_price = base_price * definition.fuel_price_multiplier
        if optimize:
            problem = OptimizationProblem(
                distance_nm=distance_nm,
                vessel=vsl,
                environment=env,
                available_fuels=list(available_fuels),
                max_eta_hours=max_eta_hours,
                max_ghg_tonnes=cap,
                fuel_price_usd_per_tonne=scenario_price,
                population_size=population_size,
                generations=generations,
                seed=seed,
            )
            report = run_optimization(problem, algorithms=("nsga2",))
            plan = report["best_plan"]
            row = {
                "scenario": definition.label,
                "key": definition.key,
                "description": definition.description,
                "speed_knots": plan["speed_knots"],
                "fuel_type": plan["fuel_type"],
                "shore_power": plan["shore_power"],
                "fuel_rate": plan["fuel_rate"],
                "fuel_rate_unit": plan["fuel_rate_unit"],
                "fuel_tonnes": plan["fuel_tonnes"],
                "cost_usd": plan["cost_usd"],
                "cost_inr": plan["cost_inr"],
                "cost_inr_crore": round(plan["cost_inr"] / 1e7, 4),
                "ghg_tonnes_co2e": plan["ghg_tonnes_co2e"],
                "eta_hours": plan["eta_hours"],
                "feasible": plan["feasible"],
                "violations": plan["violations"],
            }
        else:
            plan_obj = VoyagePlan(
                distance_nm=distance_nm,
                speed_knots=float(speed_knots),
                vessel=vsl,
                environment=env,
                fuel_type=list(available_fuels)[0],
            )
            ev = evaluate(
                plan_obj, max_eta_hours=max_eta_hours, max_ghg_tonnes=cap,
                fuel_price_usd_per_tonne=scenario_price,
            )
            row = {
                "scenario": definition.label,
                "key": definition.key,
                "description": definition.description,
                "speed_knots": round(float(speed_knots), 2),
                "fuel_type": plan_obj.fuel_type,
                "shore_power": False,
                "fuel_rate": ev.fuel_rate,
                "fuel_rate_unit": ev.fuel_rate_unit,
                "fuel_tonnes": ev.fuel_tonnes,
                "cost_usd": ev.cost_usd,
                "cost_inr": ev.cost_inr,
                "cost_inr_crore": round(ev.cost_inr / 1e7, 4),
                "ghg_tonnes_co2e": ev.ghg_tonnes_co2e,
                "eta_hours": ev.duration_hours,
                "feasible": ev.feasible,
                "violations": ev.violations,
            }

        if key == "base":
            baseline_ghg = row["ghg_tonnes_co2e"]
        rows.append(row)

    requested = set(scenario_keys)
    rows = [r for r in rows if r["key"] in requested]

    return {
        "rows": rows,
        "mode": "optimised" if optimize else "fixed-speed evaluation",
        "inputs": {
            "distance_nm": distance_nm,
            "vessel": vars(deepcopy(vessel)),
            "environment": environment.as_features(),
            "available_fuels": list(available_fuels),
            "max_eta_hours": max_eta_hours,
        },
        "assumptions": assumptions_block(),
        "note": (
            "Every row is produced by the same trained model and the same evaluator; only the "
            "scenario modifiers differ."
        ),
    }


def _run_fuelcast_scenarios(
    *, scenario_keys, distance_nm, vessel, available_fuels, fuelcast_inputs,
    fuelcast_scenario_inputs, fuelcast_speed_bounds_m_s, max_eta_hours,
    optimize, population_size, generations, seed,
) -> dict:
    """Evaluate explicit FuelCast snapshots without changing shared settings."""
    if fuelcast_inputs is None:
        raise EvaluationError("fuelcast_xgboost requires fuelcast_inputs.")
    if list(available_fuels) != ["Marine Diesel"]:
        raise EvaluationError("FuelCast scenarios require Marine Diesel only.")
    if "high_cargo_demand" in scenario_keys:
        raise EvaluationError("Cargo load is not a FuelCast model feature.")
    for key in ("severe_weather", "fuelcast_speed_change"):
        if key in scenario_keys and key not in fuelcast_scenario_inputs:
            raise EvaluationError(f"{key} requires an explicit FuelCast input snapshot.")
    if optimize and fuelcast_speed_bounds_m_s is None:
        raise EvaluationError("Optimized FuelCast scenarios require explicit m/s speed bounds.")
    if not math.isfinite(distance_nm) or distance_nm <= 0:
        raise EvaluationError("Scenario distance must be finite and positive.")

    settings = get_settings()
    base_price = settings.fuel_price_usd_per_tonne
    ghg_factor = CO2E_TONNES_PER_TONNE_FUEL["Marine Diesel"]
    predictor = get_predictor("fuelcast_xgboost")
    model_run_id = predictor.metadata().get("run_id")
    baseline_ghg = None
    ordered = sorted(scenario_keys, key=lambda key: 0 if key == "base" else 1)
    if "strict_emission_regulation" in ordered and "base" not in ordered:
        ordered = ["base", *ordered]
    rows = []

    for key in ordered:
        definition = SCENARIO_CATALOG[key]
        reference = dict(fuelcast_scenario_inputs.get(key, fuelcast_inputs))
        changed_fields = {name: value for name, value in reference.items()
                          if value != fuelcast_inputs[name]}
        scenario_price = base_price * definition.fuel_price_multiplier
        cap = None
        if definition.max_ghg_multiplier is not None:
            if baseline_ghg is None:
                raise EvaluationError("The base scenario must run before the emission-cap scenario.")
            cap = baseline_ghg * definition.max_ghg_multiplier
        applied_overrides = {"fuelcast_inputs": changed_fields}
        if definition.fuel_price_multiplier != 1.0:
            applied_overrides["fuel_price_multiplier"] = definition.fuel_price_multiplier
        if cap is not None:
            applied_overrides["max_ghg_tonnes_co2e"] = cap

        if optimize:
            problem = OptimizationProblem(
                distance_nm=distance_nm, vessel=vessel, environment=None,
                available_fuels=["Marine Diesel"], allow_shore_power=False,
                model_id="fuelcast_xgboost", fuelcast_inputs=reference,
                speed_bounds=fuelcast_speed_bounds_m_s, predictor=predictor,
                fuel_price_usd_per_tonne=scenario_price,
                max_eta_hours=max_eta_hours, max_ghg_tonnes=cap,
                population_size=population_size, generations=generations, seed=seed,
            )
            plan = run_optimization(problem, algorithms=("nsga2",))["best_plan"]
            actual_snapshot = {**reference, "speed_over_ground": plan["speed_over_ground_m_s"]}
            row = {
                "scenario": definition.label, "key": key, "description": definition.description,
                "speed_knots": plan["speed_knots"],
                "speed_over_ground_m_s": plan["speed_over_ground_m_s"],
                "fuel_type": "Marine Diesel", "shore_power": False,
                "fuel_rate": plan["fuel_rate"], "fuel_rate_unit": "kg/s",
                "fuel_tonnes": plan["fuel_tonnes"],
                "cost_usd": plan["cost_usd"], "cost_inr": plan["cost_inr"],
                "cost_inr_crore": round(plan["cost_inr"] / 1e7, 4),
                "ghg_tonnes_co2e": plan["ghg_tonnes_co2e"],
                "eta_hours": plan["eta_hours"], "feasible": plan["feasible"],
                "violations": plan["violations"],
                "optimization_algorithm_id": plan["algorithm_id"],
                "raw_prediction": plan["raw_prediction"],
                "duration_hours": plan["conversion_duration_hours"],
                "normalized_fuel_tonnes": plan["normalized_voyage_fuel_tonnes"],
            }
        else:
            speed_m_s = reference["speed_over_ground"]
            if not math.isfinite(speed_m_s) or speed_m_s <= 0:
                raise EvaluationError("FuelCast speed over ground must be finite and positive for scenario duration.")
            speed_knots = speed_m_s / KNOT_IN_M_S
            duration = distance_nm / speed_knots
            if not math.isfinite(duration) or duration <= 0:
                raise EvaluationError("FuelCast scenario duration must be finite and positive.")
            prediction = predictor.predict([reference])
            rates = prediction.get("fuel_rates", [])
            if prediction.get("fuel_rate_unit") != "kg/s" or len(rates) != 1:
                raise ModelUnavailableError("FuelCast scenario must return one kg/s fuel rate.")
            rate = rates[0]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
                raise ModelUnavailableError("FuelCast scenario returned an invalid fuel rate.")
            fuel_tonnes = rate * duration * 3.6
            if not math.isfinite(fuel_tonnes) or fuel_tonnes < 0:
                raise EvaluationError("FuelCast scenario fuel total must be finite and non-negative.")
            cost_usd = fuel_tonnes * scenario_price
            cost_inr = cost_usd * settings.usd_to_inr
            ghg = fuel_tonnes * ghg_factor
            if not all(math.isfinite(value) for value in (cost_usd, cost_inr, ghg)):
                raise EvaluationError("FuelCast scenario cost and emissions must be finite.")
            violations = []
            if max_eta_hours is not None and duration > max_eta_hours:
                violations.append("ETA exceeds the requested limit.")
            if cap is not None and ghg > cap:
                violations.append("GHG exceeds the scenario cap.")
            actual_snapshot = reference
            row = {
                "scenario": definition.label, "key": key, "description": definition.description,
                "speed_knots": round(speed_knots, 2), "speed_over_ground_m_s": speed_m_s,
                "fuel_type": "Marine Diesel", "shore_power": False,
                "fuel_rate": rate, "fuel_rate_unit": "kg/s", "fuel_tonnes": fuel_tonnes,
                "cost_usd": round(cost_usd, 2), "cost_inr": round(cost_inr, 2),
                "cost_inr_crore": round(cost_inr / 1e7, 4),
                "ghg_tonnes_co2e": round(ghg, 4), "eta_hours": duration,
                "feasible": not violations, "violations": violations,
                "optimization_algorithm_id": None, "raw_prediction": rate,
                "duration_hours": duration, "normalized_fuel_tonnes": fuel_tonnes,
            }
        row.update({
            "model_id": "fuelcast_xgboost", "model_run_id": model_run_id,
            "raw_prediction_unit": "kg/s", "normalized_fuel_unit": "tonnes",
            "fuelcast_input_snapshot": actual_snapshot,
            "applied_overrides": applied_overrides,
            "post_prediction_assumptions": {
                "fuel_price_usd_per_tonne": scenario_price,
                "co2e_tonnes_per_tonne_fuel": ghg_factor,
                "co2e_factor_unit": "tonnes CO2e per tonne Marine Diesel",
            },
        })
        if key == "base":
            baseline_ghg = row["ghg_tonnes_co2e"]
        rows.append(row)

    requested = set(scenario_keys)
    return {
        "rows": [row for row in rows if row["key"] in requested],
        "mode": "optimised" if optimize else "fixed-speed evaluation",
        "model_id": "fuelcast_xgboost", "model_run_id": model_run_id,
        "inputs": {
            "distance_nm": distance_nm, "fuelcast_inputs": dict(fuelcast_inputs),
            "fuelcast_scenario_inputs": {key: dict(value) for key, value in fuelcast_scenario_inputs.items()},
            "fuelcast_speed_bounds_m_s": list(fuelcast_speed_bounds_m_s) if fuelcast_speed_bounds_m_s else None,
            "available_fuels": ["Marine Diesel"], "max_eta_hours": max_eta_hours,
        },
        "assumptions": {
            "base_fuel_price_usd_per_tonne": base_price,
            "co2e_tonnes_per_tonne_fuel": ghg_factor,
            "co2e_factor_unit": "tonnes CO2e per tonne Marine Diesel",
            "note": "One fixed FuelCast snapshot per scenario; price and GHG cap are post-prediction assumptions.",
        },
        "note": (
            "FuelCast uses explicit snapshots only. Wind-direction convention remains unverified; "
            "a fixed snapshot over a voyage is a prototype assumption, not a model-quality claim."
        ),
    }
