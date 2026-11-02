"""RGB-visible rollouts and an explicit body-frame view for near-field motion."""
import math
import cv2
import numpy as np


COLORS = [(28, 125, 153), (201, 139, 40), (123, 95, 165),
          (47, 149, 118), (184, 94, 74), (76, 116, 191)]


def body_coordinates(trajectory, pose):
    pts = np.asarray(trajectory, dtype=float).reshape(-1, 3)
    if not len(pts):
        return pts
    dx, dy = pts[:, 0] - pose[0], pts[:, 1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy,
                            np.arctan2(np.sin(pts[:, 2]-pose[2]), np.cos(pts[:, 2]-pose[2]))])


def camera_segments(local, width, height, fov=1.396):
    focal = width / (2 * math.tan(fov / 2))
    segments, current, count = [], [], 0
    for forward, lateral, _ in local:
        depth = forward - .34
        uv = None
        if depth > .08:
            u, v = width / 2 - lateral * focal / depth, height / 2 + .39 * focal / depth
            if 0 <= u < width and 0 <= v < height:
                uv = (round(u), round(v))
        if uv is None:
            if len(current) > 1:
                segments.append(current)
            current = []
        else:
            current.append(uv)
            count += 1
    if len(current) > 1:
        segments.append(current)
    return segments, count


def render_candidate_view(rgb, pose, candidates, objects=(), selected=None):
    """No synthetic camera paths: off-image motion is shown only in body frame.

    candidates contain behavior_id, motion_behavior, motion_sequence, feasible, trajectory.
    This rendering does not change the selected control or candidate cost.
    """
    h, w = rgb.shape[:2]
    side, header = h, 38
    canvas = np.full((h+header, w+side, 3), 255, dtype=np.uint8)
    canvas[header:, :w] = rgb
    ink = (40, 58, 71)
    def label(text, xy, color=ink, scale=.48, thickness=1):
        cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)
    label('RGB: visible projected segments', (12, 25), scale=.55)
    label('Body-frame motion candidates', (w+12, 25), scale=.55)
    cv2.line(canvas, (w, 0), (w, h+header), (210, 218, 223), 1)
    origin = (w+side//2, header+int(h*.58))
    scale = min(side/4.8, h/4.8)
    def uv(x, y):
        return (round(origin[0]-y*scale), round(origin[1]-x*scale))
    plot_bottom = header+int(h*.73)
    cv2.arrowedLine(canvas, uv(-.3, 0), uv(2.35, 0), (203, 211, 216), 1, tipLength=.035)
    label('Forward', (origin[0]+9, header+28), scale=.52)
    cv2.arrowedLine(canvas, uv(0, -1.8), uv(0, 1.8), (203, 211, 216), 1, tipLength=.035)
    def footprint(point, color, thick=1):
        x, y, yaw = point
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [uv(x+c*a-s*b, y+s*a+c*b) for a,b in
                   [(-.3755,-.25),(.3755,-.25),(.3755,.25),(-.3755,.25)]]
        cv2.polylines(canvas, [np.asarray(corners, np.int32)], True, color, thick, cv2.LINE_AA)
        cv2.arrowedLine(canvas, uv(x,y), uv(x+.28*c,y+.28*s), color, thick, tipLength=.3)
    for obj in objects:
        x,y = obj['position'][:2]
        q = body_coordinates([[x,y,pose[2]]],pose)[0]
        depth=q[0]-.34
        if depth>.08:
            focal=w/(2*math.tan(1.396/2))
            u,v=w/2-q[1]*focal/depth,h/2+.39*focal/depth
            if 0<=u<w and 0<=v<h:
                label(obj.get('evidence_id',''),(max(0,round(u)-22),max(header+15,round(v)+header-12)),(240,190,45),scale=.52)
        center = uv(*q[:2])
        if w+10 < center[0] < w+side-10 and header+20 < center[1] < plot_bottom:
            cv2.circle(canvas, center, max(3, round(obj.get('radius',.2)*scale)), (210,199,190), 1, cv2.LINE_AA)
            label(obj.get('evidence_id',''), (center[0]+5, center[1]-8), scale=.52)
    records=[]
    for i, candidate in enumerate(candidates):
        local = body_coordinates(candidate['trajectory'],pose)
        segments,count = camera_segments(local,w,h)
        color = COLORS[i%len(COLORS)]
        identifier = f'C{i}'
        records.append(dict(candidate_id=identifier, behavior_id=candidate['behavior_id'],
                            visible_points=count, visible_segments=len(segments),
                            motion_behavior=candidate['motion_behavior'],
                            motion_sequence=candidate.get('motion_sequence', []),
                            color_rgb=color,
                            feasible=bool(candidate['feasible'])))
        if not candidate['feasible']:
            continue
        for seg in segments:
            pts=np.asarray([(u,v+header) for u,v in seg],np.int32)
            cv2.polylines(canvas,[pts],False,color,3,cv2.LINE_AA)
            label(identifier,tuple(pts[-1]+[4,-5]),color)
        if len(local):
            pts=np.asarray([uv(x,y) for x,y,_ in local],np.int32)
            # Clip to the body-view region; do not draw into the camera panel.
            layer=canvas[:,w:].copy()
            cv2.polylines(layer,[pts-np.array([w,0])],False,color,
                          3 if candidate['behavior_id']==selected else 2,cv2.LINE_AA)
            canvas[:,w:]=layer
            footprint(local[-1],color)
            if np.linalg.norm(local[-1,:2]-local[0,:2]) < .05:
                cv2.circle(canvas, uv(*local[-1,:2]), 7, color, 1, cv2.LINE_AA)
    footprint((0,0,0),ink,2)
    visible=[(i,c) for i,c in enumerate(candidates) if c['feasible']]
    for row,(i,c) in enumerate(visible):
        family={'ACKERMANN':'Curved','PARALLEL':'Parallel', 'SIDE_SLIP':'Lateral',
                'SPIN_THEN_DRIVE':'Rotate-drive','STOP':'Hold'}.get(c['motion_behavior'],c['motion_behavior'])
        prefix='> ' if c['behavior_id']==selected else '  '
        sequence='-'.join(c.get('motion_sequence', []))
        description=(sequence if sequence else family)
        label(f"{prefix}C{i} {description}",
              (w+12,plot_bottom+19+row*18),COLORS[i%len(COLORS)],scale=.52)
    return canvas, records
