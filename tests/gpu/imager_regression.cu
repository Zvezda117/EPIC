// Artificial, bounded GPU tests for the production imager. No observations are read.
#include "fft_dx.cuh"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EPIC_HAS_AUTOCORR_FLAG
#define EPIC_HAS_AUTOCORR_FLAG 1
#endif

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
#define CUDA(call) check((call), #call)
using C = std::complex<double>;
static_assert(LWA_SV_NSTANDS==16 && LWA_SV_NPOLS==2,
              "These fixtures require the compiled 16-slot dual-polarization array");
struct Point { int x,y,ant; C ex,ey; };
struct Case {
  std::string name; int n,support,nseq; bool gain; int offset,count,active;
  int boundary=0;
  bool remove_auto=false;
  int gulps=1;
};

template<class T> struct Device {
  T* ptr=nullptr;
  explicit Device(size_t count) { CUDA(cudaMalloc(&ptr,count*sizeof(T))); }
  ~Device() { if(ptr) cudaFree(ptr); }
  void upload(const std::vector<T>& v) { CUDA(cudaMemcpy(ptr,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
};

// Reference is a sparse direct DFT, independent of all CUDA gridding/FFT helpers.
// The production two-dimensional inverse FFT leaves its result transposed and
// divides voltage by 64 once. Products then receive the production factor N.
static std::vector<double> reference(const Case& t, const std::vector<CNib2>& voltage,
                                     const std::vector<float3>& positions,
                                     const std::vector<float4>& gain,
                                     const std::vector<float>& kernels) {
  std::vector<double> result(2*t.n*t.n*4,0.0);
  const int ne=t.support*t.support;
  const double pi=std::acos(-1.0), scale=t.n/(64.0*64.0);
  for(int ch=t.offset;ch<t.offset+t.count;++ch) {
    for(int seq=0;seq<t.nseq;++seq) {
      std::vector<Point> points;
      for(int ant=0;ant<16;++ant) {
        const auto v=voltage[(ch*t.nseq+seq)*16+ant];
        const auto g=gain[ch*16+ant];
        const C ex=C(v.X.re,v.X.im)*C(g.x,g.y), ey=C(v.Y.re,v.Y.im)*C(g.z,g.w);
        if(ex==C(0,0)&&ey==C(0,0)) continue;
        for(int elem=0;elem<ne;++elem) {
          const int x=int(positions[ch*16+ant].x+elem%t.support-t.support/2);
          const int y=int(positions[ch*16+ant].y+t.support/2-elem/t.support);
          if(x<0||y<0||x>=t.n||y>=t.n) continue;
          const double weight=kernels[(ch*16+ant)*ne+elem];
          if(weight) points.push_back({x,y,ant,ex*weight,ey*weight});
        }
      }
      for(int py=0;py<t.n;++py) for(int px=0;px<t.n;++px) {
        C ex(0,0),ey(0,0);
        C ax[16]{}, ay[16]{};
        for(const auto& p:points) {
          const double phase=2*pi*(py*p.x+px*p.y)/t.n;
          const C phasor(std::cos(phase),std::sin(phase));
          ex+=p.ex*phasor; ey+=p.ey*phasor;
          ax[p.ant]+=p.ex*phasor; ay[p.ant]+=p.ey*phasor;
        }
        const C xy=ex*std::conj(ey);
        const size_t i=(ch*t.n*t.n+py*t.n+px)*4;
        result[i]+=std::norm(ex)*scale;
        result[i+1]+=std::norm(ey)*scale;
        result[i+2]+=xy.real()*scale;
        result[i+3]+=xy.imag()*scale;
        if(t.remove_auto) for(int ant=0;ant<16;++ant) {
          const C self= ax[ant]*std::conj(ay[ant]);
          result[i]-=std::norm(ax[ant])*scale;
          result[i+1]-=std::norm(ay[ant])*scale;
          result[i+2]-=self.real()*scale;
          result[i+3]-=self.imag()*scale;
        }
      }
    }
  }
  return result;
}

template<class FFT> static bool run(const Case& t, int repeat) {
  const int ne=t.support*t.support, channels=2;
  std::vector<CNib2> voltage(channels*t.nseq*16);
  std::vector<float3> positions(channels*16);
  std::vector<float4> gain(channels*16);
  std::vector<float> kernels(channels*16*ne,0), correction(channels*t.n*t.n,1);
  for(int ch=0;ch<channels;++ch) for(int ant=0;ant<16;++ant) {
    positions[ch*16+ant]={float(t.n/2-2+4*(ant%2)),float(t.n/2-1+3*(ant%2)),0};
    gain[ch*16+ant]=t.gain?float4{1,.25f,.75f,-.25f}:float4{1,0,1,0};
    const int center=ne/2;
    if(t.support==1) kernels[(ch*16+ant)*ne]=ch?.5f:1.0f;
    else if(t.support==5) {
      // Reach the outer support cells and cross the shared half-grid boundary.
      kernels[(ch*16+ant)*ne+center]=.5f;
      kernels[(ch*16+ant)*ne+center-2]=ch?.25f:.125f;
      kernels[(ch*16+ant)*ne+center+2]=ch?.25f:.125f;
      if(!ch) {
        kernels[(ch*16+ant)*ne+center-2*t.support]=.125f;
        kernels[(ch*16+ant)*ne+center+2*t.support]=.125f;
      }
    } else {
      kernels[(ch*16+ant)*ne+center]=ch?.5f:1.0f;
      if(ch) { kernels[(ch*16+ant)*ne+center-1]=.25f; kernels[(ch*16+ant)*ne+center+1]=.25f; }
    }
    if(ant<t.active) for(int seq=0;seq<t.nseq;++seq) {
      auto& v=voltage[(ch*t.nseq+seq)*16+ant];
      v.X.re=1+seq%3+ant; v.X.im=ch;
      v.Y.re=1+ch+ant; v.Y.im=(seq+ch)%2;
    }
  }
  if(t.boundary) for(int ch=0;ch<channels;++ch)
    positions[ch*16].y=t.boundary<0?-1.0f:float(t.n+1);
  auto expected=reference(t,voltage,positions,gain,kernels);
  for(auto& value:expected) value*=t.gulps;
  Case inclusive=t;inclusive.remove_auto=false;
  const auto unremoved=reference(inclusive,voltage,positions,gain,kernels);
  Device<CNib2> vin(voltage.size());vin.upload(voltage);
  Device<float3> pos(positions.size());pos.upload(positions);
  Device<float4> phases(gain.size());phases.upload(gain);
  Device<float> gcf(kernels.size());gcf.upload(kernels);
  Device<float> corr(correction.size());corr.upload(correction);
  Device<float> output(expected.size());
  auto* input=reinterpret_cast<uint8_t*>(vin.ptr);
  auto* positions_ptr=reinterpret_cast<float*>(pos.ptr);
  auto* phase_ptr=reinterpret_cast<float*>(phases.ptr);
  cudaTextureObject_t texture=0;
  int nseq=t.nseq,nchan=channels,offset=t.offset;bool first=true;
#if EPIC_HAS_AUTOCORR_FLAG
  bool remove_auto=t.remove_auto;
  Device<float> self_response(channels*16*t.n*t.n);
  ComputeAutoCorrPower<<<dim3((t.n*t.n+255)/256,channels*16),256>>>(
      gcf.ptr,positions_ptr,self_response.ptr,t.n,t.support,16);
  CUDA(cudaDeviceSynchronize());
  const float* auto_power=self_response.ptr;
#endif
  void* args[]={&input,&positions_ptr,&phase_ptr,&nseq,&nchan,&texture,
                &output.ptr,&offset,&first,&gcf.ptr,&corr.ptr
#if EPIC_HAS_AUTOCORR_FLAG
                ,&remove_auto,&auto_power
#endif
  };
  // Use exactly the production kernel selector and production shared-memory size.
  void* kernel=GetImagingKernel<FFT,float2>(t.support);
  const int smem=t.n==64?FFT::shared_memory_size*2:FFT::shared_memory_size*3/2;
  CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  double plane_scale[4]={0,0,0,0};
  for(int ch=t.offset;ch<t.offset+t.count;++ch) for(int i=0;i<t.n*t.n;++i)
    for(int p=0;p<4;++p) plane_scale[p]=std::max(plane_scale[p],std::abs(unremoved[(ch*t.n*t.n+i)*4+p])*t.gulps);
  bool passed=true;
  for(int trial=0;trial<repeat;++trial) {
    const std::vector<float> sentinel(expected.size(),-999);
    output.upload(sentinel);
    for(int gulp=0;gulp<t.gulps;++gulp) {
      first=gulp==0;
      CUDA(cudaLaunchKernel(kernel,dim3(t.count),FFT::block_dim,args,smem,nullptr));
    }
    CUDA(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    CUDA(cudaMemcpy(actual.data(),output.ptr,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double maxerr[4]={0,0,0,0};size_t mismatches=0,untouched_bad=0;
    for(int ch=0;ch<channels;++ch) for(int i=0;i<t.n*t.n;++i) for(int p=0;p<4;++p) {
      const size_t idx=(ch*t.n*t.n+i)*4+p;
      if(ch<t.offset||ch>=t.offset+t.count) { if(actual[idx]!=-999) ++untouched_bad; continue; }
      const double err=std::abs(actual[idx]-expected[idx]);
      maxerr[p]=std::max(maxerr[p],err);
      // Half FFT roundoff is compared to the plane peak, including null pixels.
      const double tolerance=2e-5+0.0075*plane_scale[p];
      if(!std::isfinite(actual[idx])||err>tolerance) ++mismatches;
    }
    const bool okay=!mismatches&&!untouched_bad;
    passed &= okay;
    std::printf("%s trial=%d %s bad=%zu untouched_bad=%zu maxerr=[%.6g,%.6g,%.6g,%.6g] peak=[%.6g,%.6g,%.6g,%.6g]\n",
        t.name.c_str(),trial,okay?"PASS":"FAIL",mismatches,untouched_bad,
        maxerr[0],maxerr[1],maxerr[2],maxerr[3],plane_scale[0],plane_scale[1],plane_scale[2],plane_scale[3]);
  }
  return passed;
}

int main(int argc,char** argv) {
  int device=0,repeat=1;std::string selected;
  for(int i=1;i<argc;++i) {
    const std::string arg=argv[i];
    if(arg=="--device"&&i+1<argc) device=std::stoi(argv[++i]);
    else if(arg=="--case"&&i+1<argc) selected=argv[++i];
    else if(arg=="--repeat"&&i+1<argc) repeat=std::stoi(argv[++i]);
    else { std::fprintf(stderr,"Usage: %s [--device 0] [--case NAME] [--repeat N]\n",argv[0]); return 2; }
  }
  if(repeat<1||repeat>100) return 2;
  try {
    CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,device));
    std::printf("device=%d %s autocorr_flag=%d\n",device,prop.name,EPIC_HAS_AUTOCORR_FLAG);
    const std::vector<Case> cases={
      {"64-single-s1-t32",64,1,32,false,0,2,1},
      {"64-two-s1-t1",64,1,1,false,0,2,2},
      {"64-two-s3-t4-gain",64,3,4,true,0,2,2},
      {"64-offset-s3-t4",64,3,4,true,1,1,2},
      {"128-two-s1-t1",128,1,1,false,0,2,2},
      {"128-two-s3-t4-gain",128,3,4,true,0,2,2},
      {"128-offset-s3-t4",128,3,4,true,1,1,2},
      {"64-zero-s5-t4",64,5,4,true,0,2,0},
      {"128-zero-s5-t4",128,5,4,true,0,2,0},
      {"64-two-s5-t4-gain",64,5,4,true,0,2,2},
      {"128-two-s5-t4-gain",128,5,4,true,0,2,2},
      {"64-auto-single-s1",64,1,32,false,0,2,1,0,true},
      {"64-auto-single-s3-gain",64,3,4,true,0,2,1,0,true},
      {"128-auto-single-s3-gain",128,3,4,true,0,2,1,0,true},
      {"64-auto-two-s3-gain",64,3,4,true,0,2,2,0,true},
      {"128-auto-offset-s3",128,3,4,true,1,1,2,0,true},
      {"64-auto-two-gulps",64,3,4,true,0,2,2,0,true,2},
      {"64-auto-zero-s5",64,5,4,true,0,2,0,0,true},
      {"128-auto-zero-s5",128,5,4,true,0,2,0,0,true},
      {"64-auto-single-s5-gain",64,5,4,true,0,2,1,0,true},
      {"128-auto-single-s5-gain",128,5,4,true,0,2,1,0,true},
      {"128-auto-two-s5-gain",128,5,4,true,0,2,2,0,true},
      {"boundary-low",64,1,1,false,0,2,1,-1},
      {"boundary-high",64,1,1,false,0,2,1,1}};
    bool found=false,passed=true;
    for(const auto& t:cases) {
      if(selected.empty()&&t.boundary) continue; // Run unsafe boundary cases in isolated processes.
      if(!selected.empty()&&selected!=t.name) continue;
      found=true;
      passed &= t.n==64?run<FFT64x64>(t,repeat):run<FFT128x128>(t,repeat);
    }
    if(!found) { std::fprintf(stderr,"Unknown case: %s\n",selected.c_str());return 2; }
    return passed?0:1;
  } catch(const std::exception& e) { std::fprintf(stderr,"CUDA/test failure: %s\n",e.what());return 3; }
}
