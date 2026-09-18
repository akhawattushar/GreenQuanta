"""Common Evaluator — the single place where a voyage plan is scored.

Fuel consumption comes from the trained model. Cost and greenhouse-gas figures
are derived from it using published conversion factors that are *assumptions*,
not learned quantities; every response carries the `assumptions` block below so
a reviewer can see exactly what was applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.core.config import get_settings
from app.services.model_registry import FeatureValidationError, get_bundle

# --------------------------------------------------------------------------
# Documented assumptions. Not model output. Not measured by this project.
# --------------------------------------------------------------------------
#: Tank-to-wake CO2e emitted per tonne of fuel burned.
CO2E_TONNES_PER_TONNE_FUEL = {
    "Marine Diesel": 3.206,
    "LNG": 2.750,
    "Methanol": 1.375,
    "Ammonia": 0.000,
    "Hydrogen": 0.000,
}
#: Mass multiplier to deliver the same energy as one tonne of marine diesel
#: (ratio of lower heating values, MDO = 42.7 MJ/kg).
ENERGY_MASS_FACTOR = {
    "Marine Diesel": 1.000,
    "LNG": 0.890,
    "Methanol": 2.146,
    "Ammonia": 2.296,
    "Hydrogen": 0.356,
}
#: Price per tonne relative to marine diesel.
RELATIVE_PRICE = {
    "Marine Diesel": 1.00,
    "LNG": 0.82,
    "Methanol": 1.45,
    "Ammonia": 2.10,
    "Hydrogen": 5.50,
}
SUPPORTED_FUELS = tuple(CO2E_TONNES_PER_TONNE_FUEL)

#: Auxiliary load assumed while alongside, and the share removed by shore power.
PORT_STAY_HOURS = 12.0
AUXILIARY_FRACTION_OF_MAIN_RATE = 0.12
SHORE_POWER_AUXILIARY_REDUCTION = 0.90

ASSUMPTION_SOURCES = {
    "co2e_tonnes_per_tonne_fuel": "IMO MEPC carbon-conversion factors (tank-to-wake).",
    "energy_mass_factor": "Lower-heating-value ratios against marine diesel (42.7 MJ/kg).",
    "relative_price": "Indicative fuel price ratios; override via configuration.",
    "port_stay": (
        f"{PORT_STAY_HOURS} h alongside at {AUXILIARY_FRACTION_OF_MAIN_RATE:.0%} of the modelled "
        f"rate; shore power removes {SHORE_POWER_AUXILIARY_REDUCTION:.0%} of that."
    ),
    "note": "These are configuration constants, not learned by the model, and are not validated here.",
}


class EvaluationError(ValueError):
    """Raised when a plan cannot be scored."""


@dataclass
class Environment:
    """Sea state and weather, matching the model's training features."""

    wind_speed: float
    wind_direction_relative: float
    combined_wave_height: float
    combined_wave_period: float
    sea_current_speed: float
    sea_current_direction_relative: float
    sea_water_temperature: float

    def as_features(self) -> dict:
        return asdict(self)


@dataclass
class VesselState:
    vessel_type: str
    displacement: float
    trim: float


@dataclass
class VoyagePlan:
    """One candidate plan the evaluator can score."""

    distance_nm: float
    speed_knots: float
    vessel: VesselState
    environment: Environment
    fuel_type: str = "Marine Diesel"
    shore_power: bool = False


@dataclass
class Evaluation:
    fuel_rate: float
    fuel_rate_unit: str
    duration_hours: float
    main_engine_fuel_tonnes: float
    auxiliary_fuel_tonnes: float
    fuel_tonnes: float
    cost_usd: float
    cost_inr: float
    ghg_tonnes_co2e: float
    feasible: bool
    violations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cost_inr_lakh"] = round(self.cost_inr / 1e5, 3)
        data["cost_inr_crore"] = round(self.cost_inr / 1e7, 4)
        return data


def feature_row(plan: VoyagePlan) -> dict:
    """Assemble the exact raw feature dict the preprocessor expects."""
    row = {
        "sailing_speed": float(plan.speed_knots),
        "displacement": float(plan.vessel.displacement),
        "trim": float(plan.vessel.trim),
        "vessel_type": plan.vessel.vessel_type,
    }
    row.update(plan.environment.as_features())
    return row


def predict_fuel_rates(plans) -> list:
    """Batch inference through the trained model (single transform call)."""
    bundle = get_bundle()
    rows = [feature_row(p) for p in plans]
    return bundle.predict(rows)


def _fuel_tonnes_from_rate(rate: float, hours: float, fuel_type: str) -> float:
    """Rate is reported in the model's target unit (kg/h by configuration)."""
    baseline_tonnes = rate * hours / 1000.0
    return baseline_tonnes * ENERGY_MASS_FACTOR[fuel_type]


def evaluate_with_rate(
    plan: VoyagePlan,
    fuel_rate: float,
    *,
    max_eta_hours: float | None = None,
    max_ghg_tonnes: float | None = None,
    max_cost_inr: float | None = None,
) -> Evaluation:
    """Score a plan given an already-computed model rate."""
    settings = get_settings()
    if plan.fuel_type not in SUPPORTED_FUELS:
        raise EvaluationError(
            f"Unsupported fuel {plan.fuel_type!r}. Supported: {', '.join(SUPPORTED_FUELS)}."
        )
    if plan.speed_knots <= 0:
        raise EvaluationError("Speed must be greater than zero.")
    if plan.distance_nm <= 0:
        raise EvaluationError("Distance must be greater than zero.")

    hours = plan.distance_nm / plan.speed_knots
    main_tonnes = _fuel_tonnes_from_rate(fuel_rate, hours, plan.fuel_type)

    aux_rate = fuel_rate * AUXILIARY_FRACTION_OF_MAIN_RATE
    aux_tonnes = _fuel_tonnes_from_rate(aux_rate, PORT_STAY_HOURS, plan.fuel_type)
    if plan.shore_power:
        aux_tonnes *= 1.0 - SHORE_POWER_AUXILIARY_REDUCTION

    total_tonnes = main_tonnes + aux_tonnes
    price = settings.fuel_price_usd_per_tonne * RELATIVE_PRICE[plan.fuel_type]
    cost_usd = total_tonnes * price
    ghg = total_tonnes * CO2E_TONNES_PER_TONNE_FUEL[plan.fuel_type]

    violations: list = []
    if max_eta_hours is not None and hours > max_eta_hours:
        violations.append(f"ETA {hours:.1f} h exceeds the {max_eta_hours:.1f} h limit.")
    if max_ghg_tonnes is not None and ghg > max_ghg_tonnes:
        violations.append(f"GHG {ghg:.1f} t exceeds the {max_ghg_tonnes:.1f} t cap.")
    cost_inr = cost_usd * settings.usd_to_inr
    if max_cost_inr is not None and cost_inr > max_cost_inr:
        violations.append(f"Cost ₹{cost_inr:,.0f} exceeds the ₹{max_cost_inr:,.0f} budget.")

    return Evaluation(
        fuel_rate=round(fuel_rate, 4),
        fuel_rate_unit=settings.target_unit,
        duration_hours=round(hours, 3),
        main_engine_fuel_tonnes=round(main_tonnes, 4),
        auxiliary_fuel_tonnes=round(aux_tonnes, 4),
        fuel_tonnes=round(total_tonnes, 4),
        cost_usd=round(cost_usd, 2),
        cost_inr=round(cost_inr, 2),
        ghg_tonnes_co2e=round(ghg, 4),
        feasible=not violations,
        violations=violations,
    )


def evaluate(plan: VoyagePlan, **constraints) -> Evaluation:
    """Score a single plan, running the model for it."""
    try:
        rate = predict_fuel_rates([plan])[0]
    except FeatureValidationError as exc:
        raise EvaluationError(str(exc)) from exc
    return evaluate_with_rate(plan, rate, **constraints)


def evaluate_many(plans, **constraints) -> list:
    """Score many plans with one batched model call."""
    try:
        rates = predict_fuel_rates(plans)
    except FeatureValidationError as exc:
        raise EvaluationError(str(exc)) from exc
    return [evaluate_with_rate(p, r, **constraints) for p, r in zip(plans, rates)]


def assumptions_block() -> dict:
    settings = get_settings()
    return {
        "co2e_tonnes_per_tonne_fuel": dict(CO2E_TONNES_PER_TONNE_FUEL),
        "energy_mass_factor": dict(ENERGY_MASS_FACTOR),
        "relative_price": dict(RELATIVE_PRICE),
        "base_fuel_price_usd_per_tonne": settings.fuel_price_usd_per_tonne,
        "usd_to_inr": settings.usd_to_inr,
        "port_stay_hours": PORT_STAY_HOURS,
        "auxiliary_fraction_of_main_rate": AUXILIARY_FRACTION_OF_MAIN_RATE,
        "shore_power_auxiliary_reduction": SHORE_POWER_AUXILIARY_REDUCTION,
        "model_target_unit": settings.target_unit,
        "model_target_unit_verified": settings.target_unit_verified,
        "sources": ASSUMPTION_SOURCES,
    }
