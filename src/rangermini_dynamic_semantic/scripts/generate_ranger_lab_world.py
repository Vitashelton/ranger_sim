#!/usr/bin/env python3
"""Generate the two fire-door states of the Ranger lab corridor world."""
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "worlds"
SOURCE = WORLDS / "corridor_902_904_906_908.sdf"

# The source layout was a generic corridor sketch.  Map the same topology into
# the longitudinal axis of two_room_elevator_candidate.pgm. The preview is
# cropped and cannot contain the full room depths while preserving the user's
# stated corridor width, so the transverse dimensions are deliberately metric.
# This is a layout proxy, not a surveyed reconstruction.
X_SCALE = 0.90
Y_SCALE = 1.50
Y_OFFSET = -7.20
DOOR_X = 11.45 * X_SCALE
DOOR_CLEAR_WIDTH = (4.6 - 2.2 - 0.12) * Y_SCALE
DOOR_LEAF_LENGTH = DOOR_CLEAR_WIDTH / 2.0
DOOR_HINGES = {
    'north': (DOOR_X, Y_OFFSET + 4.6 * Y_SCALE - 0.06 * Y_SCALE),
    'south': (DOOR_X, Y_OFFSET + 2.2 * Y_SCALE + 0.06 * Y_SCALE),
}


def door_leaf(side, angle_deg):
    import math
    x, y = DOOR_HINGES[side]
    direction = -1.0 if side == 'north' else 1.0
    yaw = math.radians(angle_deg) * (-direction)
    center = direction * DOOR_LEAF_LENGTH / 2.0
    geometry = f'<geometry><box><size>0.09 {DOOR_LEAF_LENGTH:.6f} 2.10</size></box></geometry>'
    # Model origin is the hinge. Rotation therefore never moves the hinge.
    return (f'<model name="fire_door_{side}_leaf"><static>true</static>'
            f'<pose>{x:.6f} {y:.6f} 0 0 0 {yaw:.8f}</pose><link name="leaf">'
            f'<collision name="panel_collision"><pose>0 {center:.6f} 1.05 0 0 0</pose>{geometry}</collision>'
            f'<visual name="panel"><pose>0 {center:.6f} 1.05 0 0 0</pose>{geometry}'
            '<material><ambient>0.72 0.22 0.16 1</ambient><diffuse>0.72 0.22 0.16 1</diffuse></material></visual>'
            '<visual name="hinge"><pose>0 0 1.05 0 0 0</pose><geometry><cylinder><radius>0.025</radius><length>2.1</length></cylinder></geometry>'
            '<material><ambient>0.15 0.15 0.15 1</ambient><diffuse>0.15 0.15 0.15 1</diffuse></material></visual>'
            '</link></model>')


def model(name, x, y, z, sx, sy, sz, color, *, collision=True, yaw=0.0):
    x *= X_SCALE
    y = Y_OFFSET + Y_SCALE * y
    sx *= X_SCALE
    sy *= Y_SCALE
    c = (f'<collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size>'
         f'</box></geometry></collision>') if collision else ""
    return (f'<model name="{name}"><static>true</static>'
            f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.5f}</pose><link name="link">{c}'
            f'<visual name="visual"><geometry><box><size>{sx} {sy} {sz}</size></box>'
            f'</geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse>'
            f'</material></visual></link></model>')


def wall_x(name, x0, x1, y, gaps=(), *, z=1.2, h=2.4, t=0.12):
    parts = []
    cursor = x0
    for center, width in sorted(gaps):
        left, right = center - width / 2, center + width / 2
        if left > cursor:
            parts.append(model(f"{name}_{len(parts)}", (cursor + left) / 2, y, z,
                               left - cursor, t, h, "0.72 0.74 0.76 1"))
        cursor = max(cursor, right)
    if cursor < x1:
        parts.append(model(f"{name}_{len(parts)}", (cursor + x1) / 2, y, z,
                           x1 - cursor, t, h, "0.72 0.74 0.76 1"))
    return "\n".join(parts)


def wall_y(name, x, y0, y1, gaps=(), *, z=1.2, h=2.4, t=0.12):
    parts = []
    cursor = y0
    for center, width in sorted(gaps):
        low, high = center - width / 2, center + width / 2
        if low > cursor:
            parts.append(model(f"{name}_{len(parts)}", x, (cursor + low) / 2, z,
                               t, low - cursor, h, "0.72 0.74 0.76 1"))
        cursor = max(cursor, high)
    if cursor < y1:
        parts.append(model(f"{name}_{len(parts)}", x, (cursor + y1) / 2, z,
                           t, y1 - cursor, h, "0.72 0.74 0.76 1"))
    return "\n".join(parts)


def build(door_state):
    source = SOURCE.read_text()
    robot_match = re.search(r'<model name="rangermini_2_0"[\s\S]*?</model>', source)
    if not robot_match:
        raise RuntimeError("Could not find the Ranger model in the source world")
    robot = robot_match.group(0).replace("<pose>1.2 2.3 0.01 0 0 0</pose>",
                                       "<pose>0.75 -2.10 0.01 0 0 0</pose>", 1)

    parts = [
        '<world name="ranger_lab_corridor">',
        '<physics name="1ms" type="ignored"><max_step_size>0.002</max_step_size>'
        '<real_time_factor>1.0</real_time_factor></physics>',
        '<plugin filename="libignition-gazebo-physics-system.so" name="ignition::gazebo::systems::Physics"/>',
        '<plugin filename="libignition-gazebo-user-commands-system.so" name="ignition::gazebo::systems::UserCommands"/>',
        '<plugin filename="libignition-gazebo-scene-broadcaster-system.so" name="ignition::gazebo::systems::SceneBroadcaster"/>',
        '<plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors"><render_engine>ogre2</render_engine></plugin>',
        '<scene><ambient>0.78 0.78 0.78 1</ambient><background>0.94 0.95 0.97 1</background></scene>',
        '<light type="directional" name="building_light"><pose>10 3 9 0 0 0</pose><diffuse>1 1 1 1</diffuse><direction>-0.3 0.2 -1</direction></light>',
        # Main floor and the corridor continuing beyond the fire door.
        model("building_floor", 10.2, 3.65, -0.02, 21.4, 8.8, 0.04, "0.88 0.89 0.90 1", collision=False),
        model("main_corridor_floor", 10.1, 3.4, 0.005, 20.6, 2.4, 0.02, "0.80 0.82 0.84 1", collision=False),
        model("north_rooms_floor", 9.8, 6.0, 0.005, 19.8, 2.8, 0.02, "0.91 0.90 0.87 1", collision=False),
        model("south_rooms_floor", 8.8, 0.6, 0.005, 17.6, 2.8, 0.02, "0.87 0.91 0.88 1", collision=False),
        model("passenger_lift_branch_floor", 18.9, 0.8, 0.006, 1.8, 2.8, 0.02, "0.80 0.82 0.84 1", collision=False),
        # Wall centerlines are 3.60 m apart; with 0.18 m walls, clear width
        # is 3.42 m (6.84 Ranger body widths).
        # Gaps are the doorways into rooms or lifts.
        wall_x("north_corridor_wall", 0.0, 20.4, 4.6,
               [(2.2, 1.0), (5.6, 1.0), (9.0, 1.0), (13.1, 1.0), (16.0, 1.0)]),
        wall_x("south_corridor_wall", 0.0, 20.4, 2.2,
               [(3.6, 1.0), (8.5, 1.0), (14.7, 1.2), (18.9, 1.4)]),
        # Three instructor offices and two student labs on the first corridor section.
        wall_x("north_office_outer_wall", 0.0, 11.2, 7.4),
        wall_x("south_lab_outer_wall", 0.0, 11.2, -0.8),
        wall_y("north_office_divider_1", 4.0, 4.6, 7.4),
        wall_y("north_office_divider_2", 7.6, 4.6, 7.4),
        wall_y("north_office_divider_3", 11.2, 4.6, 7.4),
        wall_y("south_lab_divider_1", 6.1, -0.8, 2.2),
        wall_y("south_lab_divider_2", 11.2, -0.8, 2.2),
        # Two more offices after the fire door, and the freight lift vestibule opposite.
        wall_x("far_office_outer_wall", 11.7, 20.4, 7.4),
        wall_y("far_office_divider", 15.0, 4.6, 7.4),
        wall_x("freight_lift_south_wall", 12.8, 16.7, -0.8),
        wall_y("freight_lift_left_wall", 12.8, -0.8, 2.2),
        wall_y("freight_lift_right_wall", 16.7, -0.8, 2.2),
        # Passenger lift is reached by turning right into a short side branch.
        wall_y("passenger_branch_left_wall", 18.1, -0.6, 2.2),
        wall_y("passenger_branch_right_wall", 19.7, -0.6, 2.2),
        wall_x("passenger_lift_back_wall", 18.1, 19.7, -0.6, gaps=[(18.9, 1.1)]),
        wall_y("passenger_lift_left_wall", 18.1, -0.6, 1.25),
        wall_y("passenger_lift_right_wall", 19.7, -0.6, 1.25),
        # Static elevator mockups are vestibules; no vertical lift motion is simulated.
        model("freight_elevator_cab_floor", 14.75, 0.8, 0.006, 3.8, 2.6, 0.02, "0.72 0.74 0.78 1", collision=False),
        model("passenger_elevator_cab_floor", 18.9, 0.32, 0.006, 1.8, 1.8, 0.02, "0.72 0.74 0.78 1", collision=False),
        # Furniture gives rooms visual identity but stays away from the door thresholds.
        model("desk_613", 1.9, 6.2, 0.38, 1.1, 0.55, 0.72, "0.48 0.34 0.23 1"),
        model("desk_615", 5.9, 6.2, 0.38, 1.1, 0.55, 0.72, "0.48 0.34 0.23 1"),
        model("desk_617_placeholder", 9.3, 6.2, 0.38, 1.1, 0.55, 0.72, "0.48 0.34 0.23 1"),
        model("lab_bench_614", 2.3, 0.4, 0.42, 1.6, 0.65, 0.8, "0.36 0.48 0.58 1"),
        model("lab_bench_616_placeholder", 8.8, 0.4, 0.42, 1.6, 0.65, 0.8, "0.36 0.48 0.58 1"),
        model("desk_far_office_A", 13.2, 6.2, 0.38, 1.0, 0.55, 0.72, "0.48 0.34 0.23 1"),
        model("desk_far_office_B", 17.0, 6.2, 0.38, 1.0, 0.55, 0.72, "0.48 0.34 0.23 1"),
        model("freight_elevator_marker", 14.75, 1.5, 0.35, 1.2, 0.08, 0.7, "0.24 0.36 0.48 1", collision=False),
        model("passenger_elevator_marker", 18.9, 2.0, 0.35, 1.1, 0.08, 0.7, "0.20 0.42 0.55 1", collision=False),
    ]

    angles = {'closed': (0, 0), 'half_open': (90, 0),
              'angled': (45, 45), 'open': (90, 90)}[door_state]
    parts.extend([door_leaf('north', angles[0]), door_leaf('south', angles[1])])

    parts.extend([robot, "</world>"])
    return '<sdf version="1.9">\n' + "\n".join(parts) + "\n</sdf>\n"


def main():
    for state in ("open", "half_open", "angled", "closed"):
        out = WORLDS / f"ranger_lab_corridor_firedoor_{state}.sdf"
        out.write_text(build(state))
        print(out)


if __name__ == "__main__":
    main()
