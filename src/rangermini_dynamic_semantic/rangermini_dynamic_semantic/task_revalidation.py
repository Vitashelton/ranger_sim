"""One-step task-risk value of a feasible observation action.

Binary facts, Bayes updates and value of information are established tools.
This experimental objective couples them to existing feasible MPPI viewpoints;
it is neither an originality claim nor a calibrated Gazebo task result.
"""
from dataclasses import dataclass
import math


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _cost(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class TaskLoss:
    """Loss of reusing an invalid fact, repairing a valid fact, or abstaining.

    Costs share the same task-loss unit; they are task specifications, not
    model confidence or probabilities.
    """
    invalid_reuse: float
    unnecessary_repair: float
    abstain: float

    def __post_init__(self):
        for field in ("invalid_reuse", "unnecessary_repair", "abstain"):
            _cost(getattr(self, field), field)

    def decision(self, valid_probability):
        p = _probability(valid_probability, "valid_probability")
        risks = {"reuse": (1-p)*self.invalid_reuse,
                 "repair": p*self.unnecessary_repair,
                 "abstain": self.abstain}
        action = min(risks, key=risks.get)
        return {"action": action, "risk": risks[action], "risks": risks}


@dataclass(frozen=True)
class ObservationKernel:
    """Conditional evidence model; parameters require independent calibration.

    Informative support/contradiction must include adequate view coverage.
    A generic detector miss is an uncertain outcome, not contradiction.
    """
    sensitivity: float
    false_support: float

    def __post_init__(self):
        _probability(self.sensitivity, "sensitivity")
        _probability(self.false_support, "false_support")

    def outcomes(self, valid_probability, informative_probability):
        p = _probability(valid_probability, "valid_probability")
        v = _probability(informative_probability, "informative_probability")
        likelihoods = {
            "support": (v*self.sensitivity, v*self.false_support),
            "contradict": (v*(1-self.sensitivity), v*(1-self.false_support)),
            "uncertain": (1-v, 1-v),
        }
        result = {}
        for name, (given_valid, given_invalid) in likelihoods.items():
            probability = p*given_valid + (1-p)*given_invalid
            # Zero-probability outcomes cannot affect expected risk.
            posterior = (p*given_valid/probability
                         if probability and name != "uncertain" else p)
            result[name] = {"probability": probability,
                            "posterior_valid_probability": posterior}
        return result

    def expected_risk(self, valid_probability, informative_probability, loss):
        prior = loss.decision(valid_probability)
        outcomes = self.outcomes(valid_probability, informative_probability)
        expected = 0.0
        for outcome in outcomes.values():
            decision = loss.decision(outcome["posterior_valid_probability"])
            outcome["task_decision"] = decision["action"]
            outcome["posterior_risk"] = decision["risk"]
            expected += outcome["probability"]*decision["risk"]
        gain = prior["risk"]-expected
        if gain < -1e-9:
            raise ArithmeticError("expected Bayes risk unexpectedly exceeds prior risk")
        return {"prior_risk": prior["risk"], "expected_posterior_risk": expected,
                "expected_task_risk_reduction": max(0.0, gain), "outcomes": outcomes}


@dataclass(frozen=True)
class TaskRevalidationObjective:
    """Score one already feasible trajectory and one terminal observation.

    Predicted visibility is used as an experimental informative-probability
    proxy. It is not a calibrated success probability. Repeated correlated
    frames are deliberately not counted as independent observations.
    """
    valid_probability: float
    loss: TaskLoss
    kernel: ObservationKernel
    time_cost_per_s: float
    distance_cost_per_m: float
    query_cost: float
    query_latency_s: float

    def __post_init__(self):
        _probability(self.valid_probability, "valid_probability")
        if not isinstance(self.loss, TaskLoss) or not isinstance(self.kernel, ObservationKernel):
            raise TypeError("TaskLoss and ObservationKernel are required")
        for field in ("time_cost_per_s", "distance_cost_per_m", "query_cost", "query_latency_s"):
            _cost(getattr(self, field), field)

    def evaluate(self, trajectory, informative_probability, movement_duration_s):
        duration = _cost(movement_duration_s, "movement_duration_s")
        if not trajectory:
            raise ValueError("a feasible trajectory is required")
        for pose in trajectory:
            if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
                raise ValueError("trajectory requires finite (x, y, yaw) poses")
        distance = sum(math.hypot(b[0]-a[0], b[1]-a[1])
                       for a, b in zip(trajectory, trajectory[1:]))
        terms = self.kernel.expected_risk(self.valid_probability,
                                           informative_probability, self.loss)
        effort = (self.time_cost_per_s*(duration+self.query_latency_s)
                  + self.distance_cost_per_m*distance + self.query_cost)
        return {**terms, "movement_duration_s": duration,
                "path_distance_m": distance, "effort_cost": effort,
                "objective": terms["expected_posterior_risk"]+effort,
                "net_task_value": terms["expected_task_risk_reduction"]-effort}

    def choose(self, candidates):
        """Compare feasible scored observation actions to no new observation.

        Candidates must already pass the local geometry planner. This method
        returns semantic selection metadata, never a chassis command.
        """
        prior = self.loss.decision(self.valid_probability)
        best = {"action": "no_new_observation", "candidate_id": None,
                "objective": prior["risk"], "task_decision": prior["action"]}
        for candidate in candidates:
            if not candidate.get("feasible", False):
                continue
            cost = _cost(candidate["objective"], "candidate_objective")
            if cost < best["objective"]:
                best = {"action": "observe", "candidate_id": candidate["candidate_id"],
                        "objective": cost, "task_decision": "pending_current_evidence"}
        return best
