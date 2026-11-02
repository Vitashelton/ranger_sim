#!/usr/bin/env python3
"""Synthetic decision checks, explicitly not Gazebo performance evidence."""
import argparse
import json
from pathlib import Path

from rangermini_dynamic_semantic.command_semantics_mppi import MPPIConfig
from rangermini_dynamic_semantic.observation_mppi import ObservationMPPI
from rangermini_dynamic_semantic.task_revalidation import (
    ObservationKernel, TaskLoss, TaskRevalidationObjective)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    loss = TaskLoss(invalid_reuse=10, unnecessary_repair=4, abstain=3)
    kernel = ObservationKernel(sensitivity=0.9, false_support=0.1)

    def scenario(name, prior, visibility, sensor=kernel, query_cost=0.2):
        objective = TaskRevalidationObjective(prior, loss, sensor, 0.0, 0.0,
                                               query_cost, 0.0)
        terms = objective.evaluate([(0, 0, 0)], visibility, 0.0)
        selection = objective.choose([{"candidate_id": "registered_view",
                                       "feasible": True, "objective": terms["objective"]}])
        return {"name": name, "prior_valid_probability": prior,
                "informative_probability_proxy": visibility,
                "terms": terms, "selection": selection}

    cases = [scenario("ambiguous_memory_informative_view", 0.8, 0.9),
             scenario("same_memory_occluded_view", 0.8, 0.0),
             scenario("clear_view_cannot_change_task_decision", 0.99, 0.9),
             scenario("clear_view_uninformative_semantics", 0.8, 0.9,
                      ObservationKernel(0.9, 0.9))]
    assert [c["selection"]["action"] for c in cases] == [
        "observe", "no_new_observation", "no_new_observation", "no_new_observation"]
    assert abs(cases[0]["terms"]["expected_posterior_risk"]-0.668) < 1e-9

    # Belief expectation is preserved by Bayes updates; informative observation
    # cannot increase optimal expected decision loss when effort is omitted.
    invariant_checks = 0
    for p in (0.0, 0.1, 0.5, 0.8, 0.99, 1.0):
        for v in (0.0, 0.25, 0.9, 1.0):
            for sensor in (kernel, ObservationKernel(0.9, 0.9), ObservationKernel(1, 0)):
                terms = sensor.expected_risk(p, v, loss)
                outcomes = terms["outcomes"]
                assert abs(sum(o["probability"] for o in outcomes.values())-1) < 1e-9
                assert abs(sum(o["probability"]*o["posterior_valid_probability"]
                               for o in outcomes.values())-p) < 1e-9
                assert terms["expected_posterior_risk"] <= terms["prior_risk"]+1e-9
                assert outcomes["uncertain"]["posterior_valid_probability"] == p
                invariant_checks += 1

    cfg = MPPIConfig(samples_per_behavior=4, horizon=6, dt=0.15, allow_reverse=False)
    hypotheses = [(2.0, 0.0, 1.0, 1.0)]
    high_cost = TaskRevalidationObjective(0.8, loss, kernel, 0, 0, 10, 0)
    risk_planner = ObservationMPPI(config=cfg, revalidation=high_cost)
    risk_plan = risk_planner.plan((0, 0, 0), (0, 1), hypotheses=hypotheses,
                                  obstacle_points=[])
    assert risk_plan["objective_type"] == "task_risk"
    assert risk_plan["selected_behavior"] == "NO_NEW_OBSERVATION"
    assert risk_plan["action"] == (0.0, 0.0, 0.0)
    assert risk_plan["candidate_summary"]
    assert all(c["revalidation_terms"] is not None for c in risk_plan["candidate_summary"])
    affordable = TaskRevalidationObjective(0.8, loss, kernel, 0, 0, 0.01, 0)
    affordable_plan = ObservationMPPI(config=cfg, revalidation=affordable).plan(
        (0, 0, 0), (0, 1), hypotheses=hypotheses, obstacle_points=[])
    assert affordable_plan["revalidation_decision"]["action"] == "observe"
    default_plan = ObservationMPPI(config=cfg).plan(
        (0, 0, 0), (0, 1), hypotheses=hypotheses, obstacle_points=[])
    assert default_plan["objective_type"] == "observability"
    assert default_plan["revalidation_terms"] is None
    report = {"evidence_type": "synthetic_numerical_method_check",
              "gazebo_episode": False, "deepseek_api_called": False,
              "sensor_parameters": "illustrative_not_calibrated",
              "cases": cases, "bayes_invariant_checks": invariant_checks,
              "risk_mppi_high_query_cost": risk_plan,
              "risk_mppi_low_query_cost": affordable_plan,
              "default_mppi_objective": default_plan["objective_type"],
              "checks_passed": True}
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"PASS: 4 decision counterexamples, {invariant_checks} Bayes checks, MPPI integration")
    print(f"Evidence: {args.output / 'report.json'} (synthetic only)")


if __name__ == "__main__":
    main()
