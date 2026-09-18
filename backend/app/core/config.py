"""Environment-driven configuration.

Deliberately free of third-party settings libraries so that the service layer
can be imported and tested without FastAPI/Pydantic installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = "GreenQuanta QuantaFleet API"
    app_env: str = field(default_factory=lambda: _env("APP_ENV", "development"))
    api_prefix: str = "/api/v1"

    # --- security --------------------------------------------------------
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY", "dev-insecure-change-me"))
    access_token_expire_minutes: int = field(
        default_factory=lambda: _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 720)
    )
    admin_emails: tuple = field(
        default_factory=lambda: tuple(
            e.strip().lower() for e in _env("ADMIN_EMAILS", "").split(",") if e.strip()
        )
    )

    # --- database (MongoDB Atlas) -----------------------------------------
    mongodb_uri: str = field(
        default_factory=lambda: _env("MONGODB_URI", "mongodb://localhost:27017")
    )
    mongodb_db: str = field(default_factory=lambda: _env("MONGODB_DB", "quantafleet"))

    # --- ML artifacts ----------------------------------------------------
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
    )
    model_path: str = field(default_factory=lambda: _env("MODEL_PATH", "model_trainer/best_model.pkl"))
    preprocessor_path: str = field(
        default_factory=lambda: _env("PREPROCESSOR_PATH", "data_transformation/preprocessor.pkl")
    )
    feature_names_path: str = field(
        default_factory=lambda: _env("FEATURE_NAMES_PATH", "data_transformation/feature_names.json")
    )
    transformation_report_path: str = field(
        default_factory=lambda: _env(
            "TRANSFORMATION_REPORT_PATH", "data_transformation/transformation_report.json"
        )
    )
    test_set_path: str = field(
        default_factory=lambda: _env(
            "TEST_SET_PATH", "data_transformation/transformed_data/transformed_test.npz"
        )
    )

    # The training artifacts never record the unit of `fuel_consumption_rate`,
    # so it is configurable and reported as unverified. See README.
    target_unit: str = field(default_factory=lambda: _env("MODEL_TARGET_UNIT", "kg/h"))
    target_unit_verified: bool = field(
        default_factory=lambda: _env_bool("MODEL_TARGET_UNIT_VERIFIED", False)
    )

    # --- economics / emissions (documented assumptions, NOT model output) --
    fuel_price_usd_per_tonne: float = field(
        default_factory=lambda: _env_float("FUEL_PRICE_USD_PER_TONNE", 650.0)
    )
    usd_to_inr: float = field(default_factory=lambda: _env_float("USD_TO_INR", 88.0))

    # --- http ------------------------------------------------------------
    cors_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in _env(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
            ).split(",")
            if o.strip()
        )
    )

    @property
    def resolved_model_path(self) -> Path:
        return self.artifacts_dir / self.model_path

    @property
    def resolved_preprocessor_path(self) -> Path:
        return self.artifacts_dir / self.preprocessor_path

    @property
    def resolved_feature_names_path(self) -> Path:
        return self.artifacts_dir / self.feature_names_path

    @property
    def resolved_transformation_report_path(self) -> Path:
        return self.artifacts_dir / self.transformation_report_path

    @property
    def resolved_test_set_path(self) -> Path:
        return self.artifacts_dir / self.test_set_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
