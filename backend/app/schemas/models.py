"""Pydantic v2 request/response models for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

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


class PredictionRequest(BaseModel):
    """Raw features for a single inference call."""

    sailing_speed: float = Field(gt=0, le=60, description="knots")
    vessel: VesselIn
    environment: EnvironmentIn
    distance_nm: float | None = Field(default=None, gt=0, le=25000)


class PredictionResponse(BaseModel):
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


# --------------------------------------------------------------------------
# optimisation
# --------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    origin: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    distance_nm: float = Field(gt=0, le=25000)
    vessel: VesselIn
    environment: EnvironmentIn
    available_fuels: list[FuelType] = Field(min_length=1)
    allow_shore_power: bool = True
    max_eta_hours: float | None = Field(default=None, gt=0, le=5000)
    max_ghg_tonnes: float | None = Field(default=None, gt=0)
    cost_weight: float = Field(default=0.5, ge=0, le=1)
    algorithm: AlgorithmChoice = "both"
    population_size: int = Field(default=40, ge=8, le=200)
    generations: int = Field(default=60, ge=5, le=400)
    seed: int = 42

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
    scenarios: list[str] = Field(min_length=1)
    distance_nm: float = Field(gt=0, le=25000)
    vessel: VesselIn
    environment: EnvironmentIn
    available_fuels: list[FuelType] = Field(min_length=1)
    speed_knots: float | None = Field(default=None, gt=0, le=60)
    max_eta_hours: float | None = Field(default=None, gt=0, le=5000)
    optimize: bool = True
    population_size: int = Field(default=24, ge=8, le=120)
    generations: int = Field(default=25, ge=5, le=200)
    seed: int = 42


class ScenarioResponse(BaseModel):
    run_id: str
    result: dict[str, Any]


# --------------------------------------------------------------------------
# voyages / admin / reports
# --------------------------------------------------------------------------
class VoyageCreateRequest(BaseModel):
    vessel: str = Field(min_length=1, max_length=80)
    vessel_type: str = Field(min_length=1, max_length=60)
    origin: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    distance_nm: float = Field(gt=0, le=25000)
    speed_knots: float = Field(gt=0, le=60)
    fuel_loaded_tonnes: float = Field(gt=0, le=100000)
    departed_at: str | None = Field(default=None, description="ISO timestamp; defaults to now.")

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
