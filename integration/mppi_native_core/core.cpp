#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
constexpr double pi=3.14159265358979323846;
double wrap(double a){return std::atan2(std::sin(a),std::cos(a));}
double clamp(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
}
extern "C" {
int ranger_native_abi(){return 1;}
// params: wheelbase, track, speed limit, steering rate, Ackermann limit,
// parallel limit, deadband. state: vx,vy,wz,dt,mode,pending,angles[4].
// result: twist[3],mode,pending,hold,angles[4]. No global mutable state.
void ranger_wheel_step(const double* p,const double* state,double* out){
 double vx=state[0],vy=state[1],w=state[2];
 double speed=std::hypot(vx,vy);
 if(speed>p[2]){vx*=p[2]/speed;vy*=p[2]/speed;}
 int mode=(std::hypot(vx,vy)<=p[6] && std::abs(w)<=p[6])?0:
          std::abs(vy)>p[6]?2:(std::abs(vx)<=p[6] && std::abs(w)>p[6])?3:1;
 bool pending=state[5]!=0;
 if(mode!=static_cast<int>(state[4]) && mode!=0)pending=true;
 const double xs[4]={p[0]/2,p[0]/2,-p[0]/2,-p[0]/2};
 const double ys[4]={p[1]/2,-p[1]/2,p[1]/2,-p[1]/2};
 double angles[4];bool aligned=true;
 double max_step=p[3]*std::max(0.,state[3]);
 for(int i=0;i<4;++i){
  double wx=vx-w*ys[i],wy=vy+w*xs[i],previous=state[6+i];
  double a=std::hypot(wx,wy)>p[6]?std::atan2(wy,wx):previous;
  double alt=wrap(a+pi);
  if(std::abs(wrap(alt-previous))<std::abs(wrap(a-previous)))a=alt;
  double limit=mode==2?p[5]:mode==1?p[4]:pi/2;
  a=clamp(a,-limit,limit);
  double error=wrap(a-previous);
  if(std::abs(error)>.025)aligned=false;
  angles[i]=wrap(previous+clamp(error,-max_step,max_step));
  out[6+i]=angles[i];
 }
 out[0]=out[1]=out[2]=0;out[3]=mode;out[5]=0;
 if(mode==0){out[4]=pending;return;}
 if(pending){out[4]=!aligned;out[5]=1;return;}
 double sx=0,sy=0,sw=0,radius=0;
 for(int i=0;i<4;++i){
  double c=std::cos(angles[i]),s=std::sin(angles[i]);
  double v=clamp((vx-w*ys[i])*c+(vy+w*xs[i])*s,-p[2],p[2]);
  double wx=v*c,wy=v*s;
  sx+=wx;sy+=wy;sw+=-ys[i]*wx+xs[i]*wy;
  radius+=xs[i]*xs[i]+ys[i]*ys[i];
 }
 if(radius>=1e-12){out[0]=sx/4;out[1]=sy/4;out[2]=sw/radius;}
 double peak=0;
 for(int i=0;i<4;++i)peak=std::max(peak,std::hypot(out[0]-out[2]*ys[i],out[1]+out[2]*xs[i]));
 if(peak>p[2])for(int i=0;i<3;++i)out[i]*=p[2]/peak;
 out[4]=0;
}
// Exact existing scan-point/rectangle cost; this is not a signed mesh distance.
void ranger_clearance(const double* points,std::size_t n,double x,double y,
                      double yaw,double length,double width,double* out){
 if(n==0){out[0]=0;out[1]=20;return;}
 double c=std::cos(yaw),s=std::sin(yaw),sum=0,mind=INFINITY;
 for(std::size_t i=0;i<n;++i){
  double dx=points[2*i]-x,dy=points[2*i+1]-y;
  double ox=std::max(std::abs(c*dx+s*dy)-length/2,0.);
  double oy=std::max(std::abs(-s*dx+c*dy)-width/2,0.);
  double d=std::hypot(ox,oy),q=std::max(.55-d,0.);
  sum+=q*q;mind=std::min(mind,d);
 }
 out[0]=sum*3;out[1]=mind;
}
}
