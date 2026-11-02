// Adapter only: links the installed, unmodified Nav2 MPPI optimizer and critics.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <xtensor/xrandom.hpp>
#include "rclcpp/rclcpp.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_mppi_controller/optimizer.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"
#include "nav_msgs/msg/path.hpp"

static thread_local std::string error_message;
class ObservedCostmap : public nav2_costmap_2d::Costmap2DROS {
public:
  ObservedCostmap() : Costmap2DROS("ranger_observed_costmap") {
    global_frame_="map"; robot_base_frame_="base_link"; footprint_padding_=0.;
    layered_costmap_=std::make_unique<nav2_costmap_2d::LayeredCostmap>("map",false,false);
    layered_costmap_->resizeMap(161,161,.05,-4.025,-4.025);
    for (auto xy : std::vector<std::pair<double,double>>{{-.3755,-.25},{.3755,-.25},{.3755,.25},{-.3755,.25}}) {
      geometry_msgs::msg::Point p;p.x=xy.first;p.y=xy.second;padded_footprint_.push_back(p);
    }
    unpadded_footprint_=padded_footprint_;layered_costmap_->setFootprint(padded_footprint_);
  }
  void attach(const std::shared_ptr<nav2_util::LifecycleNode>& parent) {
    inflation=std::make_shared<nav2_costmap_2d::InflationLayer>();
    layered_costmap_->addPlugin(inflation);
    inflation->initialize(layered_costmap_.get(),"inflation_layer",nullptr,parent,
                         parent->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));
    inflation->matchSize();layered_costmap_->setFootprint(padded_footprint_);
  }
  void set(unsigned char* data,int width,int height,double res,double x0,double y0) {
    layered_costmap_->resizeMap(width,height,res,x0,y0);
    inflation->matchSize();
    std::memcpy(getCostmap()->getCharMap(),data,width*height);
    inflation->updateCosts(*getCostmap(),0,0,width,height);
  }
  std::shared_ptr<nav2_costmap_2d::InflationLayer> inflation;
};

class RecordedOptimizer : public mppi::Optimizer {
public:
 double minimumSampleCost() const {return *std::min_element(costs_.begin(),costs_.end());}
};

struct Engine {
  std::shared_ptr<nav2_util::LifecycleNode> node;
  std::shared_ptr<ObservedCostmap> map;
  std::unique_ptr<mppi::ParametersHandler> parameters;
  RecordedOptimizer optimizer;
  int horizon;
  Engine(int samples,int steps,double dt,double speed,int seed):horizon(steps) {
    xt::random::seed(seed);
    if(!rclcpp::ok())rclcpp::init(0,nullptr);
    rclcpp::NodeOptions opts;
    opts.parameter_overrides({
      rclcpp::Parameter("controller_frequency",1./dt),
      rclcpp::Parameter("FollowPath.motion_model","Omni"),
      rclcpp::Parameter("FollowPath.batch_size",samples),
      rclcpp::Parameter("FollowPath.time_steps",steps),
      rclcpp::Parameter("FollowPath.model_dt",dt),
      rclcpp::Parameter("FollowPath.vx_max",speed),
      rclcpp::Parameter("FollowPath.vx_min",-speed),
      rclcpp::Parameter("FollowPath.vy_max",speed),
      rclcpp::Parameter("FollowPath.wz_max",.9),
      rclcpp::Parameter("FollowPath.vx_std",.18),
      rclcpp::Parameter("FollowPath.vy_std",.18),
      rclcpp::Parameter("FollowPath.wz_std",.4),
      rclcpp::Parameter("FollowPath.iteration_count",1),
      rclcpp::Parameter("FollowPath.regenerate_noises",false),
      rclcpp::Parameter("FollowPath.critics",std::vector<std::string>{"ConstraintCritic","CostCritic","GoalCritic","GoalAngleCritic","PathFollowCritic","PathAlignCritic","PathAngleCritic","TwirlingCritic"}),
      // Task preferences add nonlethal graded costs. CostCritic evaluates
      // these directly; ObstaclesCritic would invert them as exponential
      // obstacle distances, which is invalid for this additional layer.
      rclcpp::Parameter("FollowPath.CostCritic.consider_footprint",true),
      rclcpp::Parameter("FollowPath.CostCritic.cost_weight",3.81),
      rclcpp::Parameter("FollowPath.TwirlingCritic.cost_weight",10.),
      rclcpp::Parameter("FollowPath.GoalCritic.threshold_to_consider",10.),
      rclcpp::Parameter("FollowPath.GoalAngleCritic.threshold_to_consider",10.),
      rclcpp::Parameter("FollowPath.PathAlignCritic.use_path_orientations",true),
      rclcpp::Parameter("FollowPath.PathAngleCritic.forward_preference",false),
      rclcpp::Parameter("inflation_layer.inflation_radius",.6),
      rclcpp::Parameter("inflation_layer.cost_scaling_factor",5.)});
    node=std::make_shared<nav2_util::LifecycleNode>("ranger_official_mppi","",opts);
    map=std::make_shared<ObservedCostmap>();map->attach(node);
    parameters=std::make_unique<mppi::ParametersHandler>(node);
    optimizer.initialize(node,"FollowPath",map,parameters.get());
  }
};
extern "C" {
const char* nav2_mppi_error(){return error_message.c_str();}
void* nav2_mppi_create(int samples,int steps,double dt,double speed,int seed) {
  try{return new Engine(samples,steps,dt,speed,seed);}catch(const std::exception& e){error_message=e.what();return nullptr;}
}
void nav2_mppi_destroy(void* p){delete static_cast<Engine*>(p);}
int nav2_mppi_solve(void* ptr,const double* state,const double* velocity,unsigned char* grid,
 int width,int height,double res,double x0,double y0,const double* path,int count,
 double* command,double* trajectory) {
 try {
  auto& e=*static_cast<Engine*>(ptr);e.map->set(grid,width,height,res,x0,y0);
  geometry_msgs::msg::PoseStamped pose;pose.header.frame_id="map";pose.header.stamp=e.node->now();
  pose.pose.position.x=state[0];pose.pose.position.y=state[1];
  pose.pose.orientation.z=std::sin(state[2]*.5);pose.pose.orientation.w=std::cos(state[2]*.5);
  geometry_msgs::msg::Twist twist;twist.linear.x=velocity[0];twist.linear.y=velocity[1];twist.angular.z=velocity[2];
  nav_msgs::msg::Path plan;plan.header=pose.header;
  for(int i=0;i<count;++i){geometry_msgs::msg::PoseStamped p;p.header=plan.header;
    p.pose.position.x=path[3*i];p.pose.position.y=path[3*i+1];
    p.pose.orientation.z=std::sin(path[3*i+2]*.5);p.pose.orientation.w=std::cos(path[3*i+2]*.5);
    plan.poses.push_back(p);}
  auto result=e.optimizer.evalControl(pose,twist,plan,nullptr);
  command[0]=result.twist.linear.x;command[1]=result.twist.linear.y;command[2]=result.twist.angular.z;command[3]=e.optimizer.minimumSampleCost();
  auto best=e.optimizer.getOptimizedTrajectory();int n=std::min(e.horizon,static_cast<int>(best.shape(0)));
  for(int i=0;i<n;++i)for(int j=0;j<3;++j)trajectory[i*3+j]=best(i,j);
  return n;
 }catch(const std::exception& e){error_message=e.what();return -1;}
}
}
