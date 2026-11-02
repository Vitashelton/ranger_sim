"""Metric vehicle glyphs from logged poses and wheel states, without rescaling."""
import math
import numpy as np
from matplotlib.patches import Polygon

MODE_COLORS = {'DUAL_ACKERMANN':'#426786', 'PARALLEL':'#16858D',
               'SPINNING':'#B28A3E', 'STOPPED':'#D8DCDF'}


def vehicle(ax, row, color, width=.7):
    x,y,yaw=row['pose']
    c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.array([[c,-s],[s,c]])
    def world(points):
        return np.asarray(points)@rotation.T+np.array([x,y])
    body=world([[-.3755,-.25],[.3755,-.25],[.3755,.25],[-.3755,.25]])
    ax.add_patch(Polygon(body,fill=False,ec=color,lw=width,zorder=6))
    nose=world([[.13,0],[.32,0]])
    ax.annotate('',xy=nose[1],xytext=nose[0],
                arrowprops=dict(arrowstyle='->',color=color,lw=width,mutation_scale=5),zorder=7)
    wheels=row.get('actuation',{}).get('wheel_steering_rad',[])
    positions=[(.247,.182),(.247,-.182),(-.247,.182),(-.247,-.182)]
    for (wx,wy),delta in zip(positions,wheels):
        dx,dy=.075*math.cos(delta),.075*math.sin(delta)
        segment=world([[wx-dx,wy-dy],[wx+dx,wy+dy]])
        ax.plot(segment[:,0],segment[:,1],c=color,lw=1.5,zorder=7)


def mode_strip(ax, rows, y, label):
    times=[r['elapsed'] for r in rows]
    modes=[r.get('actuation',{}).get('mode','STOPPED') for r in rows]
    start=0
    for i in range(1,len(rows)+1):
        if i==len(rows) or modes[i]!=modes[start]:
            end=times[min(i,len(rows)-1)]
            ax.broken_barh([(times[start],max(0,end-times[start]))],(y-.27,.54),
                          fc=MODE_COLORS.get(modes[start],'#D8DCDF'),ec='none')
            start=i
    ax.plot(times[-1],y,marker='|',ms=5,c='#26333B',mew=.6)


def draw_map(ax, boxes, limits):
    from plot_nav2_matched_cases import corners,clipped_polygon
    for box in boxes:
        poly=clipped_polygon(corners(box[:3],*box[3:]),limits)
        if poly:
            ax.add_patch(Polygon(poly,fc='#E0E2E3',ec='#A7ADB1',lw=.35,zorder=0))
    ax.set(xlim=limits[:2],ylim=limits[2:])
    ax.set_aspect('equal',adjustable='box')
