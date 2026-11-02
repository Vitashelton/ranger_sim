#pragma once
#include <cmath>
#include <algorithm>
// Closest member of Ranger's translation / curvature-bounded arc / spin set.
// This projects a free 4WIDS twist; it does not choose a semantic behavior.
inline int ranger_mppi_seed = 623;
inline void ranger_project(double& vx,double& vy,double& w) {
 const double arc_w=std::clamp(w,-std::abs(vx)/.4764,std::abs(vx)/.4764);
 const double a=.247*.247;
 const double parallel_error=a*w*w;
 const double arc_error=vy*vy+a*(w-arc_w)*(w-arc_w);
 const double spin_error=vx*vx+vy*vy;
 if(spin_error<std::min(parallel_error,arc_error)){vx=0;vy=0;}
 else if(arc_error<parallel_error){vy=0;w=arc_w;}
 else w=0;
 const double speed=std::hypot(vx,vy);
 if(speed>.35){double scale=.35/speed;vx*=scale;vy*=scale;w*=scale;}
}
