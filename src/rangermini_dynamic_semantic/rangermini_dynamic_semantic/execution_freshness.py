"""Timestamp contract shared by simulated and hardware sensor interfaces."""
import math


def observation_fault(now, wall_now, observations, required=('pose', 'scan', 'actuation'),
                      max_age=0.7, max_wall_age=1.5, future_tolerance=0.05):
    """Return the first invalid stream; ages use acquisition and receive clocks.

    The wall bound catches a stopped ROS clock. All required streams must
    recover before motion resumes; restarting one stream cannot release a stop.
    This is a command admission check, not a collision avoidance controller.
    """
    if not all(math.isfinite(v) for v in (now, wall_now)):
        return 'invalid_clock'
    for name in required:
        item = observations.get(name)
        if item is None:
            return name + ':missing'
        stamp, received_wall = item
        if not all(math.isfinite(v) for v in item):
            return name + ':invalid_stamp'
        age, wall_age = now-stamp, wall_now-received_wall
        if age < -future_tolerance or wall_age < 0:
            return name + ':clock_regression'
        if age > max_age or wall_age > max_wall_age:
            return name + ':stale'
    return None
