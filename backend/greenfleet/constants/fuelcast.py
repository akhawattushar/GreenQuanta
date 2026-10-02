"""FuelCast source contract. Canonical renaming belongs to the ETL stage."""

DATASET_ID = "krohnedigital/FuelCast"
EXPECTED_CONFIGS = ("cps_poseidon", "cps_triton", "oss_ceto")
SOURCE_SPLIT = "train"
SOURCE_LICENSE = "CC BY-NC-ND 4.0"
SOURCE_TIME_COLUMN = "index"
SOURCE_MODEL_COLUMNS = (
    "Ship_SpeedOverGround",
    "Weather_WindSpeed10M",
    "Weather_WindDirection10M",
    "Weather_WaveHeight",
    "Weather_WavePeriod",
    "Weather_OceanCurrentVelocity",
    "Consumer_Total_MomentaryFuel",
)
REQUIRED_SOURCE_COLUMNS = (SOURCE_TIME_COLUMN, *SOURCE_MODEL_COLUMNS)
RAW_COLUMNS = ("vessel_id", "time_index", *SOURCE_MODEL_COLUMNS)

SOURCE_TO_CANONICAL = {
    "Ship_SpeedOverGround": "speed_over_ground",
    "Weather_WindSpeed10M": "wind_speed",
    "Weather_WindDirection10M": "wind_direction",
    "Weather_WaveHeight": "wave_height",
    "Weather_WavePeriod": "wave_period",
    "Weather_OceanCurrentVelocity": "current_speed",
    "Consumer_Total_MomentaryFuel": "fuel_consumption_kg_s",
}
CANONICAL_FEATURES = tuple(SOURCE_TO_CANONICAL[name] for name in SOURCE_MODEL_COLUMNS[:-1])
CANONICAL_TARGET = "fuel_consumption_kg_s"
CANONICAL_COLUMNS = (
    "record_id", "dataset_version", "vessel_id", "time_index",
    *CANONICAL_FEATURES, CANONICAL_TARGET,
)
CANONICAL_UNITS = {
    "speed_over_ground": "m/s",
    "wind_speed": "m/s",
    "wind_direction": "degrees",
    "wave_height": "m",
    "wave_period": "s",
    "current_speed": "m/s",
    "fuel_consumption_kg_s": "kg/s",
}
