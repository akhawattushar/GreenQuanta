"""Pydantic v2 request/response models for the public API."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, FiniteFloat, field_validator, model_validator

Role = Literal["operator", "admin", "regulator", "researcher"]
FuelType = Literal["Marine Diesel", "LNG", "Methanol", "Ammonia", "Hydrogen"]
AlgorithmChoice = Literal["nsga2", "quantum", "both"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Role = "operator"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(ORMModel):
    id: str
    name: str
    email: str
    role: Role
    status: str = "active"
    created_at: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


# --------------------------------------------------------------------------
# shared voyage inputs — these mirror the model's training features exactly
# --------------------------------------------------------------------------
class EnvironmentIn(BaseModel):
    wind_speed: float = Field(ge=0, le=60, description="m/s")
    wind_direction_relative: float = Field(ge=0, le=180, description="degrees relative to heading")
    combined_wave_height: float = Field(ge=0, le=20, description="metres")
    combined_wave_period: float = Field(gt=0, le=30, description="seconds")
    sea_current_speed: float = Field(ge=0, le=10, description="knots")
    sea_current_direction_relative: float = Field(ge=0, le=180, description="degrees relative to heading")
    sea_water_temperature: float = Field(ge=-5, le=45, description="degrees Celsius")


class VesselIn(BaseModel):
    vessel_type: str = Field(description="Must be one of the categories seen during training.")
    displacement: float = Field(gt=0, le=1000)
    trim: float = Field(ge=-10, le=10)


class FuelCastInputs(BaseModel):
    """Raw values supplied in the units expected by the FuelCast export."""

    model_config = ConfigDict(extra="forbid")

    speed_over_ground: FiniteFloat = Field(ge=0, description="Ship speed over ground in m/s.")
    wind_speed: FiniteFloat = Field(ge=0, description="Wind speed in m/s.")
    wind_direction: FiniteFloat = Field(
        ge=0, lt=360,
        description=("Wind direction in degrees expected by the exported FuelCast preprocessor. "
                     "Its absolute/relative and from/toward convention is not yet verified."),
    )
    wave_height: FiniteFloat = Field(ge=0, description="Wave height in metres.")
    wave_period: FiniteFloat = Field(gt=0, description="Wave period in seconds.")
    current_speed: FiniteFloat = Field(ge=0, description="Ocean current speed in m/s.")

    @model_validator(mode="before")
    @classmethod
    def _sanitize_nonfinite_for_http_errors(cls, values):
        if not isinstance(values, dict):
            return values
        # FastAPI includes rejected input in its 422 JSON response. A raw
        # NaN/Infinity token would make that response itself unserializable.
        return {name: "non-finite number" if isinstance(value, float) and not math.isfinite(value)
                else value for name, value in values.items()}


class PredictionRequest(BaseModel):
    """Legacy inputs by default, or explicit raw FuelCast inputs by model ID."""

    model_id: str = Field(default="legacy", description="Predictor ID. The legacy model is the default.")
    sailing_speed: float | None = Field(default=None, gt=0, le=60, description="Legacy sailing speed in knots.")
    vessel: VesselIn | None = None
    environment: EnvironmentIn | None = None
    distance_nm: float | None = Field(default=None, gt=0, le=25000)
    fuelcast_inputs: FuelCastInputs | None = Field(
        default=None, description="Required only for fuelcast_xgboost; values are used without legacy-field mapping."
    )

    @model_validator(mode="after")
    def _inputs_for_selected_model(self):
        if self.model_id == "legacy":
            if self.fuelcast_inputs is not None:
                raise ValueError("fuelcast_inputs cannot be used with the legacy model.")
            missing = [name for name in ("sailing_speed", "vessel", "environment") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"Legacy prediction requires: {', '.join(missing)}.")
        elif self.model_id == "fuelcast_xgboost":
            if self.fuelcast_inputs is None:
                raise ValueError("fuelcast_xgboost requires fuelcast_inputs with all six raw FuelCast fields.")
            legacy_fields = [name for name in ("sailing_speed", "vessel", "environment", "distance_nm")
                             if getattr(self, name) is not None]
            if legacy_fields:
                raise ValueError(f"fuelcast_xgboost does not accept legacy fields: {', '.join(legacy_fields)}.")
        return self


class PredictionResponse(BaseModel):
    model_id: str = "legacy"
    fuel_rate: float
    fuel_rate_unit: str
    unit_verified: bool
    voyage_fuel_tonnes: float | None = None
    duration_hours: float | None = None
    model_metadata: dict[str, Any]
    features_used: dict[str, Any]
    prediction_id: str | None = None
    note: str


class ModelInfoResponse(BaseModel):
    loaded: bool
    error: str | None = None
    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    models: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# optimisation
# --------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    model_id: str = Field(default="legacy", description="Prediction model; legacy is the default.")
    fuelcast_inputs: FuelCastInputs | None = Field(
        default=None, description="FuelCast snapshot; speed_over_ground is the reference speed in m/s."
    )
    fuelcast_speed_bounds_m_s: tuple[FiniteFloat, FiniteFloat] | None = Field(
        default=None, description="Required FuelCast candidate speed-over-ground bounds [minimum, maximum] in m/s."
    )
    origin: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    distance_nm: float = Field(gt=0, le=25000)
    vessel: VesselIn
    environment: EnvironmentIn | None = None
    available_fuels: list[FuelType] = Field(min_length=1)
    allow_shore_power: bool = True
    max_eta_hours: float | None = Field(default=None, gt=0, le=5000)
    max_ghg_tonnes: float | None = Field(default=None, gt=0)
    cost_weight: float = Field(default=0.5, ge=0, le=1)
    algorithm: AlgorithmChoice = "both"
    population_size: int = Field(default=40, ge=8, le=200)
    generations: int = Field(default=60, ge=5, le=400)
    seed: int = 42

    @model_validator(mode="after")
    def _model_contract(self):
        if self.model_id == "legacy":
            if self.environment is None:
                raise ValueError("Legacy optimization requires environment.")
            if self.fuelcast_inputs is not None or self.fuelcast_speed_bounds_m_s is not None:
                raise ValueError("FuelCast inputs and speed bounds cannot be used with the legacy model.")
        elif self.model_id == "fuelcast_xgboost":
            if self.fuelcast_inputs is None:
                raise ValueError("fuelcast_xgboost requires fuelcast_inputs.")
            if self.fuelcast_speed_bounds_m_s is None:
                raise ValueError("fuelcast_xgboost requires fuelcast_speed_bounds_m_s.")
            low, high = self.fuelcast_speed_bounds_m_s
            if low <= 0 or high <= low:
                raise ValueError("FuelCast speed bounds must be positive, finite, and increasing.")
            if not low <= self.fuelcast_inputs.speed_over_ground <= high:
                raise ValueError("FuelCast reference speed_over_ground must lie within the m/s speed bounds.")
            if self.environment is not None:
                raise ValueError("Legacy environment cannot be combined with FuelCast inputs.")
            if self.available_fuels != ["Marine Diesel"] or self.allow_shore_power:
                raise ValueError("FuelCast optimization currently requires Marine Diesel only and allow_shore_power=false.")
        else:
            raise ValueError(f"Unknown model ID {self.model_id!r}. Available IDs: legacy, fuelcast_xgboost.")
        return self

    @field_validator("destination")
    @classmethod
    def _distinct_ports(cls, value: str, info):
        if info.data.get("origin") and value.strip().lower() == info.data["origin"].strip().lower():
            raise ValueError("Origin and destination must be different.")
        return value

    @field_validator("available_fuels")
    @classmethod
    def _unique_fuels(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class OptimizationResponse(BaseModel):
    run_id: str
    request: dict[str, Any]
    result: dict[str, Any]


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
class ScenarioRequest(BaseModel):
    model_id: str = Field(default="legacy", description="Prediction model; legacy is the default.")
    fuelcast_inputs: FuelCastInputs | None = Field(
        default=None, description="Required six-field FuelCast reference snapshot for fuelcast_xgboost."
    )
    fuelcast_scenario_inputs: dict[str, FuelCastInputs] | None = Field(
        default=None, description="Complete explicit FuelCast snapshots keyed by scenario ID; required for weather or speed changes."
    )
    fuelcast_speed_bounds_m_s: tuple[FiniteFloat, FiniteFloat] | None = Field(
        default=None, description="Explicit m/s speed-over-ground bounds when optimizing FuelCast scenarios."
    )
    scenarios: list[str] = Field(min_length=1)
    distance_nm: float = Field(gt=0, le=25000)
    vessel: VesselIn
    environment: EnvironmentIn | None = None
    available_fuels: list[FuelType] = Field(min_length=1)
    shore_power: bool | None = Field(default=None, description="FuelCast scenarios require shore power disabled.")
    speed_knots: float | None = Field(default=None, gt=0, le=60)
    max_eta_hours: float | None = Field(default=None, gt=0, le=5000)
    optimize: bool = True
    population_size: int = Field(default=24, ge=8, le=120)
    generations: int = Field(default=25, ge=5, le=200)
    seed: int = 42

    @model_validator(mode="after")
    def _model_contract(self):
        fuelcast_fields = (self.fuelcast_inputs, self.fuelcast_scenario_inputs, self.fuelcast_speed_bounds_m_s)
        if self.model_id == "legacy":
            if self.environment is None:
                raise ValueError("Legacy scenarios require environment.")
            if any(value is not None for value in fuelcast_fields):
                raise ValueError("FuelCast fields cannot be combined with the legacy model.")
        elif self.model_id == "fuelcast_xgboost":
            if self.fuelcast_inputs is None:
                raise ValueError("fuelcast_xgboost requires fuelcast_inputs.")
            if self.environment is not None or self.speed_knots is not None:
                raise ValueError("Legacy environment and speed_knots cannot supply FuelCast inputs.")
            if self.available_fuels != ["Marine Diesel"]:
                raise ValueError("FuelCast scenarios currently require Marine Diesel only.")
            if self.shore_power:
                raise ValueError("FuelCast scenarios do not support shore power.")
            if "high_cargo_demand" in self.scenarios:
                raise ValueError("Cargo load is not a FuelCast model feature.")
            overrides = self.fuelcast_scenario_inputs or {}
            unsupported = set(overrides) - {"severe_weather", "fuelcast_speed_change"}
            if unsupported or set(overrides) - set(self.scenarios):
                raise ValueError("FuelCast scenario snapshots are accepted only for selected weather or speed-change scenarios.")
            missing = {key for key in self.scenarios if key in {"severe_weather", "fuelcast_speed_change"}} - set(overrides)
            if missing:
                raise ValueError(f"FuelCast scenarios require explicit snapshots for: {', '.join(sorted(missing))}.")
            if "fuelcast_speed_change" in overrides:
                base = self.fuelcast_inputs.model_dump()
                changed = overrides["fuelcast_speed_change"].model_dump()
                if changed["speed_over_ground"] == base["speed_over_ground"] or any(
                    changed[name] != base[name] for name in base if name != "speed_over_ground"
                ):
                    raise ValueError("fuelcast_speed_change must change only speed_over_ground in m/s.")
            if self.optimize:
                if self.fuelcast_speed_bounds_m_s is None:
                    raise ValueError("Optimized FuelCast scenarios require fuelcast_speed_bounds_m_s.")
                low, high = self.fuelcast_speed_bounds_m_s
                if low <= 0 or high <= low:
                    raise ValueError("FuelCast speed bounds must be positive and increasing.")
                for snapshot in (self.fuelcast_inputs, *overrides.values()):
                    if not low <= snapshot.speed_over_ground <= high:
                        raise ValueError("Each FuelCast reference speed must lie within the m/s speed bounds.")
            elif self.fuelcast_speed_bounds_m_s is not None:
                raise ValueError("FuelCast speed bounds are only used when optimize=true.")
        else:
            raise ValueError(f"Unknown model ID {self.model_id!r}. Available IDs: legacy, fuelcast_xgboost.")
        return self


class ScenarioResponse(BaseModel):
    run_id: str
    result: dict[str, Any]


# --------------------------------------------------------------------------
# voyages / admin / reports
# --------------------------------------------------------------------------
class VoyageCreateRequest(BaseModel):
    model_id: str = Field(default="legacy", description="Fuel model ID; legacy is the default.")
    fuelcast_inputs: FuelCastInputs | None = Field(
        default=None, description="Six explicit FuelCast inputs for fuelcast_xgboost only."
    )
    vessel: str = Field(min_length=1, max_length=80)
    vessel_type: str = Field(min_length=1, max_length=60)
    origin: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    distance_nm: float = Field(gt=0, le=25000)
    speed_knots: float = Field(gt=0, le=60)
    fuel_loaded_tonnes: float = Field(gt=0, le=100000)
    departed_at: str | None = Field(default=None, description="ISO timestamp; defaults to now.")

    @field_validator("departed_at")
    @classmethod
    def _valid_departure_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("departed_at must be a valid ISO-8601 timestamp.") from exc
        return value

    @model_validator(mode="after")
    def _model_inputs(self):
        if self.model_id == "legacy":
            if self.fuelcast_inputs is not None:
                raise ValueError("fuelcast_inputs cannot be used with the legacy model.")
        elif self.model_id == "fuelcast_xgboost":
            if self.fuelcast_inputs is None:
                raise ValueError("fuelcast_xgboost requires fuelcast_inputs with all six raw fields.")
        else:
            raise ValueError(f"Unknown model ID {self.model_id!r}. Available IDs: legacy, fuelcast_xgboost.")
        return self

    @field_validator("destination")
    @classmethod
    def _distinct_ports(cls, value: str, info):
        if info.data.get("origin") and value.strip().lower() == info.data["origin"].strip().lower():
            raise ValueError("Origin and destination must be different.")
        return value


class VoyageListResponse(BaseModel):
    voyages: list[dict[str, Any]]
    data_source: str
    live_telemetry: bool


class FleetSummaryResponse(BaseModel):
    vessels: int
    active_voyages: int
    total_voyages: int
    planned_fuel_tonnes: float
    fuel_consumed_tonnes: float
    fuel_model_available: bool
    data_source: str
    live_telemetry: bool
    note: str


class RoleUpdateRequest(BaseModel):
    role: Role


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: bool
    model_loaded: bool
    model_error: str | None = None
    pdf_export: bool


class ErrorResponse(BaseModel):
    detail: str
