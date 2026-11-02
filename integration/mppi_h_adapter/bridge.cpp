#include <chrono>
#include <numeric>
#include <omp.h>
#include "mppi_h/mppi_h_core.hpp"
struct Solver {
 controller_mppi_h::MPPIHybridCore* h;
 controller_mppi_3d::MPPI3DCore* d3;
 controller_mppi_4d::MPPI4DCore* d4;
 int mode,horizon;
};
template<class P> void settings(P& p,int n, bool four) {
 auto& c=p.controller;
 c.name=four?"MPPI-4D":"MPPI-3D(b)";c.num_samples=n;
 c.param_exploration=.1;c.param_lambda=four?100.:200.;c.param_alpha=.975;
 c.sigma=four?std::vector<double>{.78,.78,.175,.175}:std::vector<double>{.09625,.09625,.55};
 c.reduce_computation=false;c.weight_cmd_change={0,0,0};
 c.weight_vehicle_cmd_change={1.4,1.4,1.4,1.4,.1,.1,.1,.1};
 c.ref_velocity=.30;c.weight_velocity_error=10;c.weight_angular_error=30;
 c.weight_collision_penalty=50;c.weight_distance_error_penalty=40;c.weight_terminal_state_penalty=100;
 c.use_sg_filter=true;c.sg_filter_half_window_size=4;c.sg_filter_poly_order=6;
}
extern "C" {
void* mppi_create(int mode,int samples,int horizon,double dt,int seed) {
 ranger_mppi_seed=seed;
 // One thread avoids the upstream shared RNG data race, retaining seed 623.
 omp_set_num_threads(1);
 param::CommonParam c{}; c.navigation={.03,.15};
 c.target_system={.247,.247,.182,.182,.09};c.controller={.12,horizon,dt};c.mode_selector={.5,.5};
 param::MPPI3DParam p3{};settings(p3,samples,false);
 param::MPPI4DParam p4{};settings(p4,samples,true);
 auto* s=new Solver{nullptr,nullptr,nullptr,mode,horizon};
 if(mode==0)s->h=new controller_mppi_h::MPPIHybridCore(std::make_tuple(c,p3,p4));
 if(mode==1)s->d3=new controller_mppi_3d::MPPI3DCore(c,p3);
 if(mode==2)s->d4=new controller_mppi_4d::MPPI4DCore(c,p4);
 return s;
}
void mppi_destroy(void* ptr){auto* s=(Solver*)ptr;delete s->h;delete s->d3;delete s->d4;delete s;}
int mppi_solve(void* ptr,const double* pose,const double* goal,int nx,int ny,double x0,double y0,double res,
 const double* collision,const double* distance,const double* yaw,double* result,double* trajectory) {
 auto* s=(Solver*)ptr;grid_map::GridMap cm{collision,nx,ny,x0,y0,res},dm{distance,nx,ny,x0,y0,res},ym{yaw,nx,ny,x0,y0,res};
 common_type::XYYaw x{},g{};x.x=pose[0];x.y=pose[1];x.yaw=pose[2];g.x=goal[0];g.y=goal[1];g.yaw=goal[2];
 common_type::VxVyOmega u;std::vector<common_type::XYYaw> tr;
 if(s->mode==0){u=s->h->solveMPPI(x,cm,dm,ym,g);tr=s->h->getOptimalTrajectory();result[3]=s->h->getStateCost();result[4]=s->h->getCalcTime();result[5]=s->h->getControllerName().find("3D")!=std::string::npos?3:4;}
 if(s->mode==1){u=s->d3->solveMPPI(x,cm,dm,ym,g);tr=s->d3->getOptimalTrajectory();result[3]=s->d3->getStateCost();result[4]=s->d3->getCalcTime();result[5]=3;}
 if(s->mode==2){u=s->d4->solveMPPI(x,cm,dm,ym,g);tr=s->d4->getOptimalTrajectory();result[3]=s->d4->getStateCost();result[4]=s->d4->getCalcTime();result[5]=4;}
 result[0]=u.vx;result[1]=u.vy;result[2]=u.omega;
 for(size_t i=0;i<tr.size();++i){trajectory[3*i]=tr[i].x;trajectory[3*i+1]=tr[i].y;trajectory[3*i+2]=tr[i].yaw;}
 return tr.size();
}
}
