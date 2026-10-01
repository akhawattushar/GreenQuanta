"""Reporting reads stored results without rerunning prediction models."""

import csv
import io

from app.api.routes import report
from app.api.routes.report import _filtered, _rows_and_notes
from app.services.reporting import SNAPSHOT_NOTE, normalize_record, report_columns, rows_to_csv


def _run(kind, response, request=None):
    return {"id": f"{kind}-1", "created_at": "2026-10-01T00:00:00Z",
            "request": request or {}, "response": response}


def test_legacy_rows_keep_old_fields_and_unknown_provenance():
    record = _run("optimization", {"plans": [{"label": "Quantum-Inspired (QIEA)",
        "fuel_rate": 10.0, "fuel_rate_unit": "kg/h", "fuel_tonnes": 3.0,
        "cost_usd": 123.0, "ghg_tonnes_co2e": 8.0}]})
    rows, notes = _rows_and_notes("optimization", record)
    row = rows[0]
    assert row["label"] == "Quantum-Inspired (QIEA)"
    assert row["model_id"] == "unknown"
    assert row["optimization_algorithm"] == "quantum"
    assert row["raw_prediction_unit"] == "kg/h"
    assert row["normalized_fuel_tonnes"] == 3.0
    assert "model provenance unavailable" in row["data_quality_warnings"]
    assert not any("quantum prediction" in note.lower() for note in notes)


def test_xgboost_voyage_uses_stored_tonnes_without_reconversion():
    record = {"id": "vy-1", "departed_at": "2026-10-01", "vessel": "Test",
              "model_id": "fuelcast_xgboost", "model_run_id": "phase1",
              "raw_prediction": 2.0, "raw_prediction_unit": "kg/s",
              "conversion_duration_hours": 4.0,
              "normalized_voyage_fuel_tonnes": 28.8,
              "normalized_voyage_fuel_unit": "tonnes", "cost_usd": 300.0,
              "ghg_tonnes_co2e": 90.0, "fuelcast_inputs": {"wind_speed": 5.0}}
    rows, notes = normalize_record("voyage", record)
    row = rows[0]
    assert (row["model_id"], row["model_run_id"]) == ("fuelcast_xgboost", "phase1")
    assert (row["raw_prediction"], row["raw_prediction_unit"]) == (2.0, "kg/s")
    assert (row["normalized_fuel_tonnes"], row["normalized_fuel_unit"]) == (28.8, "tonnes")
    assert row["cost"] == 300.0 and row["emissions_unit"] == "tonnes CO2e"
    assert SNAPSHOT_NOTE in notes


def test_xgboost_scenario_and_optimization_provenance_and_filters():
    scenario = _run("scenario", {"model_id": "fuelcast_xgboost", "model_run_id": "phase1",
        "rows": [{"key": "severe_weather", "raw_prediction": 1.5,
                  "raw_prediction_unit": "kg/s", "duration_hours": 2.0,
                  "normalized_fuel_tonnes": 10.8, "normalized_fuel_unit": "tonnes",
                  "fuelcast_input_snapshot": {"wave_height": 3},
                  "applied_overrides": {"wave_height": 3},
                  "post_prediction_assumptions": {"fuel_price_usd_per_tonne": 900}}]})
    sr, _ = normalize_record("scenario", scenario)
    assert sr[0]["scenario_type"] == "severe_weather"
    assert sr[0]["scenario_overrides"] == {"wave_height": 3}
    assert sr[0]["post_prediction_assumptions"]["fuel_price_usd_per_tonne"] == 900
    opt = _run("optimization", {"model_id": "fuelcast_xgboost", "plans": [
        {"algorithm_id": "nsga2", "model_id": "fuelcast_xgboost"},
        {"algorithm_id": "quantum", "model_id": "fuelcast_xgboost"}]})
    rows, _ = normalize_record("optimization", opt)
    assert len(_filtered(rows, model_id="fuelcast_xgboost", optimization_algorithm="nsga2", source_type=None)) == 1
    assert len(_filtered(rows, model_id="legacy", optimization_algorithm=None, source_type=None)) == 0
    assert len(_filtered(rows, model_id=None, optimization_algorithm="quantum", source_type="optimization")) == 1


def test_raw_rate_without_duration_or_stored_total_stays_unavailable():
    rows, _ = normalize_record("prediction", _run("prediction", {
        "model_id": "fuelcast_xgboost", "fuel_rate": 2.0, "fuel_rate_unit": "kg/s"}))
    assert rows[0]["normalized_fuel_tonnes"] is None
    assert rows[0]["conversion_duration_hours"] is None
    assert "conversion duration unavailable" in rows[0]["data_quality_warnings"]


def test_legacy_prediction_retains_stored_output_fields():
    rows, _ = _rows_and_notes("prediction", _run("prediction", {
        "model_id": "legacy", "fuel_rate": 12.0, "fuel_rate_unit": "unverified",
        "voyage_fuel_tonnes": None, "duration_hours": None,
        "features_used": {"sailing_speed": 10.0}}))
    assert rows[0]["fuel_rate"] == 12.0
    assert rows[0]["raw_prediction_unit"] == "unverified"
    assert rows[0]["normalized_fuel_tonnes"] is None
    assert rows[0]["sailing_speed"] == 10.0


def test_malformed_rows_are_excluded_without_failing_report():
    rows, warnings = normalize_record("scenario", _run("scenario", {"rows": [None, {"fuel_tonnes": 2.0}]}))
    assert len(rows) == 1 and warnings == ["Stored result row 1 malformed; excluded."]
    assert normalize_record("optimization", _run("optimization", {"plans": "broken"}))[0] == []
    assert normalize_record("voyage", {"id": "old"})[0][0]["model_id"] == "unknown"


def test_csv_order_escaping_and_row_integrity():
    rows = [{"model_id": "fuelcast_xgboost", "source_type": "voyage",
             "vessel": '=HYPERLINK("bad","value")', "scenario_overrides": {"note": "+cmd"}}]
    text = rows_to_csv(rows, columns=report_columns(rows))
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0][0:3] == ["source_type", "source_id", "created_at"]
    assert len(parsed[0]) == len(parsed[1])
    assert parsed[1][parsed[0].index("vessel")].startswith("'=HYPERLINK")


def test_normalization_never_calls_a_predictor(monkeypatch):
    from app.services import model_registry
    monkeypatch.setattr(model_registry, "get_predictor", lambda *_: (_ for _ in ()).throw(AssertionError("model called")))
    assert normalize_record("voyage", {"id": "old"})[0]


def test_history_lists_voyages_and_filters_model_independently(monkeypatch):
    class Runs:
        def __init__(self, db, kind):
            self.kind = kind

        def history(self, user_id, limit):
            return [_run("optimization", {"model_id": "legacy", "plans": [
                {"algorithm_id": "nsga2", "fuel_tonnes": 1.0}]})] if self.kind == "optimization" else []

    class Voyages:
        def __init__(self, db):
            pass

        def list(self, user_id):
            return [{"id": "vy-1", "departed_at": "2026-10-01", "model_id": "fuelcast_xgboost",
                     "raw_prediction": 1.0, "raw_prediction_unit": "kg/s"}]

    monkeypatch.setattr(report, "RunRepository", Runs)
    monkeypatch.setattr(report, "VoyageRepository", Voyages)
    result = report.history({"id": "user-1"}, object(), model_id="fuelcast_xgboost")
    assert [item["kind"] for item in result["items"]] == ["voyage"]
    result = report.history({"id": "user-1"}, object(), optimization_algorithm="nsga2")
    assert [item["kind"] for item in result["items"]] == ["optimization"]


def test_malformed_provenance_and_mixed_units_do_not_crash():
    rows, warnings = normalize_record("optimization", _run("optimization", {"plans": [
        {"model_id": ["bad"], "fuel_rate_unit": ["bad"]},
        {"model_id": "legacy", "fuel_rate_unit": "kg/h"}]}))
    assert rows[0]["model_id"] == "unknown"
    assert rows[0]["raw_prediction_unit"] is None
    assert any("Mixed or unavailable" in warning for warning in warnings)


def test_csv_route_exports_stored_provenance_without_database(monkeypatch):
    class Runs:
        def __init__(self, db, kind):
            assert kind == "optimization"

        def latest(self, user_id):
            return _run("optimization", {"model_id": "fuelcast_xgboost", "plans": [
                {"model_id": "fuelcast_xgboost", "model_run_id": "phase1",
                 "algorithm_id": "nsga2", "raw_prediction": 1.0,
                 "raw_prediction_unit": "kg/s", "normalized_voyage_fuel_tonnes": 7.2,
                 "normalized_voyage_fuel_unit": "tonnes"}]})

    monkeypatch.setattr(report, "RunRepository", Runs)
    response = report.export_csv({"id": "u"}, object(), kind="optimization", run_id=None,
                                 model_id="fuelcast_xgboost", optimization_algorithm="nsga2", source_type=None)
    text = response.body.decode()
    assert "model_id" in text and "optimization_algorithm" in text
    assert "phase1" in text and "kg/s" in text and "7.2" in text
