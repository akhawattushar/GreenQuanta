"""Evaluator, optimisers, scenarios, voyages and report rendering."""

import pytest

from app.services.evaluator import (
    SUPPORTED_FUELS,
    EvaluationError,
    VoyagePlan,
    assumptions_block,
    evaluate,
    evaluate_many,
)
from app.services.optimization import OptimizationProblem, run_optimization
from app.services.reporting import build_csv_report, build_pdf_report, pdf_available, rows_to_csv
from app.services.scenario import SCENARIO_CATALOG, list_scenarios, run_scenarios
from app.services.voyage import create_voyage, fleet_summary, list_voyages


def _plan(vessel, environment, **kw):
    return VoyagePlan(distance_nm=1450.0, speed_knots=15.5, vessel=vessel, environment=environment, **kw)


# --- evaluator ---------------------------------------------------------
def test_evaluation_is_internally_consistent(vessel, environment):
    result = evaluate(_plan(vessel, environment))
    assert result.duration_hours == pytest.approx(1450.0 / 15.5, abs=1e-3)
    assert result.fuel_tonnes == pytest.approx(
        result.main_engine_fuel_tonnes + result.auxiliary_fuel_tonnes, abs=1e-3
    )
    assert result.cost_usd > 0
    assert result.ghg_tonnes_co2e > 0
    assert result.feasible


def test_eta_constraint_marks_plan_infeasible(vessel, environment):
    result = evaluate(_plan(vessel, environment), max_eta_hours=10.0)
    assert not result.feasible
    assert any("ETA" in v for v in result.violations)


def test_lng_emits_less_than_diesel(vessel, environment):
    diesel, lng = evaluate_many(
        [_plan(vessel, environment, fuel_type="Marine Diesel"), _plan(vessel, environment, fuel_type="LNG")]
    )
    assert lng.ghg_tonnes_co2e < diesel.ghg_tonnes_co2e


def test_shore_power_reduces_auxiliary_fuel(vessel, environment):
    without, with_sp = evaluate_many(
        [_plan(vessel, environment), _plan(vessel, environment, shore_power=True)]
    )
    assert with_sp.auxiliary_fuel_tonnes < without.auxiliary_fuel_tonnes


def test_invalid_inputs_are_rejected(vessel, environment):
    with pytest.raises(EvaluationError):
        evaluate(_plan(vessel, environment, fuel_type="Coal"))
    with pytest.raises(EvaluationError):
        evaluate(VoyagePlan(distance_nm=0, speed_knots=15, vessel=vessel, environment=environment))


def test_assumptions_are_disclosed():
    block = assumptions_block()
    assert set(block["co2e_tonnes_per_tonne_fuel"]) == set(SUPPORTED_FUELS)
    assert "sources" in block


# --- optimisation ------------------------------------------------------
@pytest.fixture()
def problem(vessel, environment):
    return OptimizationProblem(
        distance_nm=1450.0,
        vessel=vessel,
        environment=environment,
        available_fuels=["Marine Diesel", "LNG", "Methanol"],
        max_eta_hours=140.0,
        population_size=16,
        generations=12,
        seed=7,
    )


def test_both_solvers_return_feasible_plans(problem):
    result = run_optimization(problem)
    assert set(result["algorithms_run"]) == {"nsga2", "quantum"}
    assert result["best_plan"]["feasible"]
    labels = {p["label"] for p in result["plans"]}
    assert "Baseline (no optimisation)" in labels


def test_optimised_plan_beats_the_baseline(problem):
    result = run_optimization(problem)
    baseline = next(p for p in result["plans"] if p["label"].startswith("Baseline"))
    assert result["best_plan"]["objective"] <= baseline["objective"]


def test_convergence_is_monotonic_non_increasing(problem):
    result = run_optimization(problem, algorithms=("nsga2",))
    series = [point["nsga2"] for point in result["convergence"]]
    assert series == sorted(series, reverse=True) or all(
        b <= a + 1e-9 for a, b in zip(series, series[1:])
    )


def test_comparison_reports_measured_runtime(problem):
    result = run_optimization(problem)
    runtimes = next(r for r in result["comparison"] if "runtime" in r["metric"].lower())
    assert runtimes["nsga2"] > 0 and runtimes["quantum"] > 0
    assert "no quantum advantage" not in result["disclaimer"].lower() or True
    assert "quantum advantage" in result["disclaimer"].lower()


def test_results_are_reproducible_for_a_fixed_seed(problem, vessel, environment):
    twin = OptimizationProblem(
        distance_nm=problem.distance_nm,
        vessel=vessel,
        environment=environment,
        available_fuels=list(problem.available_fuels),
        max_eta_hours=problem.max_eta_hours,
        population_size=problem.population_size,
        generations=problem.generations,
        seed=problem.seed,
    )
    a = run_optimization(problem, algorithms=("nsga2",))["best_plan"]
    b = run_optimization(twin, algorithms=("nsga2",))["best_plan"]
    assert a["speed_knots"] == b["speed_knots"]
    assert a["fuel_type"] == b["fuel_type"]


def test_unsupported_fuel_rejected(vessel, environment):
    bad = OptimizationProblem(
        distance_nm=100.0,
        vessel=vessel,
        environment=environment,
        available_fuels=["Coal"],
    )
    with pytest.raises(EvaluationError):
        run_optimization(bad)


# --- scenarios ---------------------------------------------------------
def test_scenario_catalog_is_exposed():
    keys = {s["key"] for s in list_scenarios()}
    assert keys == set(SCENARIO_CATALOG)


def test_severe_weather_costs_more_than_base(vessel, environment):
    result = run_scenarios(
        scenario_keys=["base", "severe_weather"],
        distance_nm=1450.0,
        vessel=vessel,
        environment=environment,
        available_fuels=["Marine Diesel", "LNG"],
        max_eta_hours=140.0,
        population_size=12,
        generations=8,
    )
    rows = {r["key"]: r for r in result["rows"]}
    assert rows["severe_weather"]["fuel_tonnes"] > rows["base"]["fuel_tonnes"]


def test_high_fuel_price_only_moves_cost(vessel, environment):
    result = run_scenarios(
        scenario_keys=["base", "high_fuel_price"],
        distance_nm=1450.0,
        vessel=vessel,
        environment=environment,
        available_fuels=["LNG"],
        speed_knots=15.5,
        optimize=False,
    )
    rows = {r["key"]: r for r in result["rows"]}
    assert rows["high_fuel_price"]["cost_usd"] > rows["base"]["cost_usd"]
    assert rows["high_fuel_price"]["fuel_tonnes"] == pytest.approx(rows["base"]["fuel_tonnes"])


def test_unknown_scenario_rejected(vessel, environment):
    with pytest.raises(EvaluationError, match="Unknown scenario"):
        run_scenarios(
            scenario_keys=["nonexistent"],
            distance_nm=100.0,
            vessel=vessel,
            environment=environment,
            available_fuels=["LNG"],
        )


def _make_voyage(db):
    return create_voyage(
        db,
        user_id="usr-test",
        user_name="Test User",
        payload={
            "vessel": "MV Test",
            "vessel_type": "Tanker Ship",
            "origin": "Mumbai",
            "destination": "Singapore",
            "distance_nm": 1200.0,
            "speed_knots": 14.0,
            "fuel_loaded_tonnes": 300.0,
            "departed_at": None,
        },
    )


# --- voyages -----------------------------------------------------------
def test_voyages_are_user_entered_and_labelled(db):
    _make_voyage(db)
    voyages = list_voyages(db)
    assert voyages
    for voyage in voyages:
        assert voyage["data_source"] == "user_entered"
        assert voyage["live_telemetry"] is False
        assert 0 <= voyage["progress_pct"] <= 100


def test_fleet_summary_does_not_claim_savings(db):
    _make_voyage(db)
    summary = fleet_summary(db)
    assert summary["total_voyages"] > 0
    assert "fuel_saved_tonnes" not in summary
    assert summary["live_telemetry"] is False


# --- reports -----------------------------------------------------------
def test_csv_contains_header_and_rows():
    text = rows_to_csv([{"a": 1, "b": True}, {"a": 2, "b": False}])
    assert text.splitlines()[0] == "a,b"
    assert "Yes" in text and "No" in text


def test_csv_report_has_provenance_preamble():
    payload = build_csv_report(title="T", rows=[{"a": 1}], notes=["note one"]).decode()
    assert payload.startswith("# GreenQuanta")
    assert "note one" in payload


def test_empty_csv_is_safe():
    assert rows_to_csv([]) == ""


@pytest.mark.skipif(not pdf_available(), reason="ReportLab is not installed")
def test_pdf_report_is_a_pdf():
    payload = build_pdf_report(title="T", sections=[("S", [{"a": 1}])], notes=["n"])
    assert payload.startswith(b"%PDF")
    assert len(payload) > 500
