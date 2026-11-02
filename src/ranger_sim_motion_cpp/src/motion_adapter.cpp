#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/create_timer.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/string.hpp"

extern "C" void ranger_wheel_step(const double*,const double*,double*);
namespace {
constexpr double pi=3.14159265358979323846;
using Twist=geometry_msgs::msg::Twist;
using Steady=std::chrono::steady_clock;
const char* mode_name(int m){
  static const char* names[]={"STOPPED","DUAL_ACKERMANN","PARALLEL","SPINNING"};
  return names[std::clamp(m,0,3)];
}
class MotionAdapter final: public rclcpp::Node {
public:
  MotionAdapter():Node("ranger_mini_v2_motion_adapter_cpp") {
    auto input=declare_parameter<std::string>("input_topic","/cmd_vel_safe");
    auto output=declare_parameter<std::string>("output_topic","/cmd_vel_gazebo");
    auto debug=declare_parameter<std::string>("debug_topic","/motion_mode_debug");
    const double hz=declare_parameter<double>("rate_hz",50.);
    audit_period_=1./declare_parameter<double>("audit_rate_hz",10.);
    timeout_=declare_parameter<double>("command_timeout",.35);
    wall_timeout_=declare_parameter<double>("wall_command_timeout",1.5);
    p_[0]=declare_parameter<double>("wheelbase",.494);
    p_[1]=declare_parameter<double>("track",.364);
    p_[2]=declare_parameter<double>("max_linear_speed",1.5);
    p_[3]=declare_parameter<double>("max_steer_rate",2.5);
    p_[4]=declare_parameter<double>("max_ackermann_angle",40.*pi/180.);
    p_[5]=declare_parameter<double>("max_parallel_angle",pi/2.);p_[6]=1e-4;
    angular_max_=declare_parameter<double>("max_angular_speed",4.8);
    min_radius_=declare_parameter<double>("min_turn_radius",.4764);
    output_=create_publisher<Twist>(output,10);
    debug_=create_publisher<std_msgs::msg::String>(debug,10);
    command_=create_subscription<Twist>(input,10,[this](const Twist& msg){
      if(!std::isfinite(msg.linear.x)||!std::isfinite(msg.linear.y)||!std::isfinite(msg.angular.z)){
        received_=false;latest_=Twist();output_->publish(Twist());return;
      }
      latest_=msg;received_=true;command_ros_=now().seconds();command_wall_=Steady::now();
    });
    last_step_=now().seconds();
    timer_=rclcpp::create_timer(this,get_clock(),rclcpp::Duration::from_seconds(1./std::max(1.,hz)),[this]{tick();});
    watchdog_=create_wall_timer(std::chrono::milliseconds(50),[this]{
      if(wall_stale())output_->publish(Twist());
    });
    RCLCPP_INFO(get_logger(),"C++ Ranger simulation execution; shared native wheel model; %s -> %s",input.c_str(),output.c_str());
  }
private:
  bool wall_stale()const{
    return !received_||std::chrono::duration<double>(Steady::now()-command_wall_).count()>wall_timeout_;
  }
  std::array<double,3> native_command(const Twist& u){
    double vx=u.linear.x,vy=u.linear.y,w=u.angular.z;
    double speed=std::hypot(vx,vy);
    if(speed>p_[2]){vx*=p_[2]/speed;vy*=p_[2]/speed;}
    w=std::clamp(w,-angular_max_,angular_max_);
    if(std::hypot(vx,vy)<=p_[6]&&std::abs(w)<=p_[6])return {0.,0.,0.};
    if(std::abs(vy)>p_[6])return {vx,vy,0.};
    if(std::abs(vx)<=p_[6])return {0.,0.,w};
    if(std::abs(w)<=p_[6])return {vx,0.,0.};
    double radius=std::abs(vx/w);
    if(radius<min_radius_){w=std::copysign(std::abs(vx)/min_radius_,w);radius=min_radius_;}
    double inner=std::copysign(std::min(std::atan((p_[0]/2.)/radius),40.*pi/180.),vx*w);
    double a=std::abs(inner);
    double central=std::copysign(std::atan2(p_[0]*std::sin(a),p_[0]*std::cos(a)+p_[1]*std::sin(a)),inner);
    return {vx,0.,2.*vx*std::sin(central)/p_[0]};
  }
  void tick(){
    const double stamp=now().seconds();double dt=std::clamp(stamp-last_step_,0.,.08);last_step_=stamp;
    const bool stale=wall_stale()||!received_||stamp-command_ros_>timeout_;
    Twist input=stale?Twist():latest_;auto requested=native_command(input);
    std::array<double,10> state={requested[0],requested[1],requested[2],dt,
      static_cast<double>(mode_),pending_?1.:0.,angles_[0],angles_[1],angles_[2],angles_[3]};
    std::array<double,10> out{};ranger_wheel_step(p_.data(),state.data(),out.data());
    mode_=static_cast<int>(out[3]);pending_=out[4]!=0.;bool hold=out[5]!=0.;
    for(int i=0;i<4;++i)angles_[i]=out[6+i];
    Twist velocity;velocity.linear.x=out[0];velocity.linear.y=out[1];velocity.angular.z=out[2];
    output_->publish(velocity);
    std::string signature=std::to_string(mode_)+std::to_string(stale)+std::to_string(hold);
    if(signature==last_signature_&&stamp-last_debug_<audit_period_)return;
    std::ostringstream json;json<<std::setprecision(17)<<std::boolalpha;
    json<<"{\"mode\":\""<<mode_name(mode_)<<"\",\"reason\":\""<<(stale?"command timeout; safe stop":"native Ranger command")
      <<"\",\"stale\":"<<stale<<",\"wall_stale\":"<<wall_stale()
      <<",\"requested\":{\"vx\":"<<input.linear.x<<",\"vy\":"<<input.linear.y<<",\"wz\":"<<input.angular.z
      <<"},\"executed\":{\"vx\":"<<out[0]<<",\"vy\":"<<out[1]<<",\"wz\":"<<out[2]
      <<"},\"wheel_steering_rad\":["<<angles_[0]<<","<<angles_[1]<<","<<angles_[2]<<","<<angles_[3]
      <<"],\"steering_transition_stop\":"<<hold<<",\"kinematic_model\":\"4WS rigid-body pure-rolling wheel constraints\",\"implementation\":\"rclcpp_shared_native_core\"}";
    std_msgs::msg::String message;message.data=json.str();debug_->publish(message);
    last_signature_=signature;last_debug_=stamp;
  }
  std::array<double,7> p_{};std::array<double,4> angles_{};
  int mode_=0;bool pending_=false,received_=false;
  double timeout_,wall_timeout_,angular_max_,min_radius_,audit_period_,last_step_,command_ros_=0,last_debug_=-1e9;
  Steady::time_point command_wall_{};Twist latest_;std::string last_signature_;
  rclcpp::Publisher<Twist>::SharedPtr output_;rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_;
  rclcpp::Subscription<Twist>::SharedPtr command_;rclcpp::TimerBase::SharedPtr timer_,watchdog_;
};
}
int main(int argc,char** argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<MotionAdapter>());rclcpp::shutdown();}
