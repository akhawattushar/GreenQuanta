"""Scenario analysis.

Each scenario is a named set of modifiers applied to a baseline voyage. The
modified voyage is re-optimised (or re-evaluated) through exactly the same
evaluator and the same trained model, so scenarios are directly comparable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from app.core.config import get_settings
from app.services.evaluator import (
    Environment,
    EvaluationError,
    VesselState,
    VoyagePlan,
    assumptions_block,
    evaluate,
)
from app.services.optimization import OptimizationProblem, run_optimization


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
    environment: Environment,
    available_fuels,
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

        # Fuel price is a settings-level assumption, so scale it per scenario.
        object.__setattr__(settings, "fuel_price_usd_per_tonne", base_price * definition.fuel_price_multiplier)
        try:
            if optimize:
                problem = OptimizationProblem(
                    distance_nm=distance_nm,
                    vessel=vsl,
                    environment=env,
                    available_fuels=list(available_fuels),
                    max_eta_hours=max_eta_hours,
                    max_ghg_tonnes=cap,
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
                ev = evaluate(plan_obj, max_eta_hours=max_eta_hours, max_ghg_tonnes=cap)
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
        finally:
            object.__setattr__(settings, "fuel_price_usd_per_tonne", base_price)

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
