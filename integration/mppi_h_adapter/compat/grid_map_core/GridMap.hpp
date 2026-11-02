#pragma once
// ROS-independent view of a row-major metric grid. Only the GridMap API
// consumed by the upstream solver is provided; no planner logic is here.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>
namespace grid_map {
using Position = Eigen::Vector2d;
enum class InterpolationMethods { INTER_NEAREST, INTER_LINEAR };
class GridMap {
 public:
 const double* data = nullptr; int nx=0,ny=0; double x0=0,y0=0,res=.05;
 bool isInside(const Position& p) const {
 return p.x()>=x0 && p.y()>=y0 && p.x()<x0+(nx-1)*res && p.y()<y0+(ny-1)*res;
 }
 double atPosition(const std::string&, const Position& p,
 InterpolationMethods method=InterpolationMethods::INTER_NEAREST) const {
 double x=std::clamp((p.x()-x0)/res,0.,double(nx-1));
 double y=std::clamp((p.y()-y0)/res,0.,double(ny-1));
 if(method==InterpolationMethods::INTER_NEAREST) return data[int(std::round(y))*nx+int(std::round(x))];
 int i=std::min(nx-2,int(x)),j=std::min(ny-2,int(y)); double a=x-i,b=y-j;
 return (1-b)*((1-a)*data[j*nx+i]+a*data[j*nx+i+1])+b*((1-a)*data[(j+1)*nx+i]+a*data[(j+1)*nx+i+1]);
 }
};
}
