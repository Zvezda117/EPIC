#include "lband_fx.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Z = cuComplex;
thread_local std::string error_text;
void cuda_check(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char* where) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(where) + ": cuBLAS status " + std::to_string(status));
}
#define CUDA(call) cuda_check((call), #call)
#define BLAS(call) blas_check((call), #call)
struct DeviceGuard {
  int previous;
  explicit DeviceGuard(int device) { CUDA(cudaGetDevice(&previous)); CUDA(cudaSetDevice(device)); }
  ~DeviceGuard() { cudaSetDevice(previous); }
};
size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::runtime_error("dimension product overflow");
  return a * b;
}
size_t checked_sum(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b)
    throw std::runtime_error("buffer byte sum overflow");
  return a+b;
}
template<class T> void allocate(T*& ptr, size_t count) {
  CUDA(cudaMalloc(&ptr, product(count, sizeof(T))));
}
void finite(const double* ptr, size_t count, const char* name) {
  if (!ptr) throw std::runtime_error(std::string(name) + " pointer is null");
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(ptr[i])) throw std::runtime_error(std::string(name) + " must be finite");
}
void finite(const Z* ptr, size_t count, const char* name) {
  if (!ptr) throw std::runtime_error(std::string(name) + " pointer is null");
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(ptr[i].x) || !std::isfinite(ptr[i].y))
      throw std::runtime_error(std::string(name) + " must be finite");
}
void units(const double* ptr, size_t count, const char* name) {
  finite(ptr, product(count, 3), name);
  for (size_t i = 0; i < count; ++i) {
    double norm = ptr[3*i]*ptr[3*i] + ptr[3*i+1]*ptr[3*i+1] + ptr[3*i+2]*ptr[3*i+2];
    if (std::abs(norm - 1.) > 2e-6) throw std::runtime_error(std::string(name) + " must contain unit vectors");
  }
}
__global__ void make_steering(Z* a, const double* pos, const double* freq,
                              const double* dirs, const double* center,
                              int n, int c, int p) {
  size_t idx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (idx >= size_t(c)*n*p) return;
  int pixel = idx % p, ant = (idx/p) % n, channel = idx/(size_t(p)*n);
  double delay = 0.;
  for (int axis=0; axis<3; ++axis)
    delay += pos[3*ant+axis] * (dirs[3*pixel+axis] - center[axis]);
  double phase = 6.283185307179586476925286766559 * freq[channel] * delay / 299792458.;
  double sine, cosine; sincos(phase, &sine, &cosine);
  a[idx] = make_cuComplex(float(cosine), float(sine));
}
__global__ void pack_gain(const Z* input, const Z* gains, Z* x, int n, int c, int t) {
  size_t idx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (idx >= size_t(t)*c*n*2) return;
  int pol=idx%2, ant=(idx/2)%n, channel=(idx/(size_t(n)*2))%c;
  int time=idx/(size_t(c)*n*2);
  Z v=input[idx], g=gains[(size_t(channel)*n+ant)*2+pol];
  x[((size_t(channel)*2+pol)*t+time)*n+ant] =
      make_cuComplex(g.x*v.x-g.y*v.y, g.x*v.y+g.y*v.x);
}
__global__ void zero_physical_diagonal(Z* r, int n, int c) {
  size_t idx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (idx >= size_t(c)*3*n) return;
  int ant=idx%n; size_t matrix=idx/n;
  r[matrix*n*n + size_t(ant)*n+ant] = make_cuComplex(0.f,0.f);
}
__global__ void image_diagonal(const Z* z, const Z* a, float* output,
                                int n, int c, int p, int component) {
  size_t idx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (idx >= size_t(c)*p) return;
  int pixel=idx%p, channel=idx/p;
  float re=0.f, im=0.f;
  for (int ant=0; ant<n; ++ant) {
    size_t offset=(size_t(channel)*n+ant)*p+pixel;
    Z v=z[offset], b=a[offset];
    re += v.x*b.x + v.y*b.y;
    im += v.y*b.x - v.x*b.y;
  }
  output[idx*4+component]=re;
  if (component==2) output[idx*4+3]=im;
}
struct Context {
  int n=0,c=0,p=0,t=0,device=-1;
  bool remove=false;
  cublasHandle_t handle=nullptr;
  Z *input=nullptr,*x=nullptr,*a=nullptr,*gains=nullptr,*r=nullptr,*z=nullptr;
  float* output=nullptr;
  double *pos=nullptr,*freq=nullptr,*dirs=nullptr,*center=nullptr;
  size_t inputs=0;
  ~Context() {
    if(device<0) return;
    int old=-1; cudaGetDevice(&old); cudaSetDevice(device);
    if(handle) cublasDestroy(handle);
    for(void* ptr:{(void*)input,(void*)x,(void*)a,(void*)gains,(void*)r,(void*)z,
                  (void*)output,(void*)pos,(void*)freq,(void*)dirs,(void*)center})
      if(ptr) cudaFree(ptr);
    if(old>=0) cudaSetDevice(old);
  }
  void init(int nn,int cc,int pp,int tt,int dev,bool rm,const double* hp,
             const double* hf,const double* hd,const double* hc,const Z* hg) {
    if(nn<1||cc<1||pp<1||tt<1) throw std::runtime_error("all dimensions must be positive");
    n=nn;c=cc;p=pp;t=tt;remove=rm;
    inputs=product(product(product(size_t(n),c),t),2);
    size_t steering=product(product(size_t(n),c),p);
    size_t matrices=product(product(product(size_t(n),n),c),3);
    size_t gaincount=product(product(size_t(n),c),2);
    size_t outcount=product(product(size_t(c),p),4);
    // Leaves room for CUDA/cuBLAS runtime allocations under a 1GB case budget.
    size_t complex_count=checked_sum(checked_sum(product(inputs,2),product(steering,2)),
                                     checked_sum(matrices,gaincount));
    size_t geometry=checked_sum(checked_sum(product(size_t(n),3),size_t(c)),
                                 checked_sum(product(size_t(p),3),3));
    size_t bytes=checked_sum(checked_sum(product(complex_count,sizeof(Z)),
                                         product(outcount,sizeof(float))),
                               product(geometry,sizeof(double)));
    if(bytes>900ull*1024*1024) throw std::runtime_error("FX buffers exceed 900 MiB; reduce caller channel tile");
    finite(hp,product(size_t(n),3),"positions_m");finite(hf,c,"frequencies_hz");
    for(int i=0;i<c;++i) if(hf[i]<=0.) throw std::runtime_error("frequencies_hz must be positive");
    units(hd,p,"directions");units(hc,1,"phase_center");finite(hg,gaincount,"gains");
    int count=0;CUDA(cudaGetDeviceCount(&count));
    if(dev<0||dev>=count) throw std::runtime_error("invalid CUDA device index");
    device=dev;DeviceGuard guard(device);
    BLAS(cublasCreate(&handle));BLAS(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    BLAS(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
    allocate(input,inputs);allocate(x,inputs);allocate(a,steering);allocate(z,steering);
    allocate(r,matrices);allocate(gains,gaincount);allocate(output,outcount);
    allocate(pos,product(size_t(n),3));allocate(freq,c);allocate(dirs,product(size_t(p),3));allocate(center,3);
    CUDA(cudaMemcpy(pos,hp,size_t(n)*3*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(freq,hf,size_t(c)*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dirs,hd,size_t(p)*3*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(center,hc,3*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(gains,hg,gaincount*sizeof(Z),cudaMemcpyHostToDevice));
    make_steering<<<(steering+255)/256,256>>>(a,pos,freq,dirs,center,n,c,p);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    for(double** ptr:{&pos,&freq,&dirs,&center}) {CUDA(cudaFree(*ptr));*ptr=nullptr;}
  }
  void image(const Z* host,float* result) {
    if(!result) throw std::runtime_error("output pointer is null");
    finite(host,inputs,"voltages");DeviceGuard guard(device);
    CUDA(cudaMemcpy(input,host,inputs*sizeof(Z),cudaMemcpyHostToDevice));
    pack_gain<<<(inputs+255)/256,256>>>(input,gains,x,n,c,t);CUDA(cudaGetLastError());
    Z one=make_cuComplex(1.f,0.f),zero=make_cuComplex(0.f,0.f),scale=make_cuComplex(1.f/t,0.f);
    long long xs=2ll*n*t,rs=3ll*n*n,as=1ll*n*p;
    for(int component=0;component<3;++component) {
      Z* left=x+(component==1?size_t(n)*t:0);
      Z* right=x+(component>0?size_t(n)*t:0);
      BLAS(cublasCgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_C,n,n,t,
            &scale,left,n,xs,right,n,xs,&zero,r+size_t(component)*n*n,n,rs,c));
    }
    if(remove) {size_t count=size_t(c)*3*n;zero_physical_diagonal<<<(count+255)/256,256>>>(r,n,c);CUDA(cudaGetLastError());}
    for(int component=0;component<3;++component) {
      BLAS(cublasCgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,n,n,
            &one,a,p,as,r+size_t(component)*n*n,n,rs,&zero,z,p,as,c));
      size_t count=size_t(c)*p;
      image_diagonal<<<(count+255)/256,256>>>(z,a,output,n,c,p,component);
      CUDA(cudaGetLastError());
    }
    CUDA(cudaMemcpy(result,output,size_t(c)*p*4*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<size_t(c)*p*4;++i)
      if(!std::isfinite(result[i])) throw std::runtime_error("nonfinite FX output; input/gain power may overflow FP32");
  }
};
}
extern "C" const char* epic_fx_error() {return error_text.c_str();}
extern "C" void* epic_fx_create(int n,int c,int p,int t,int device,int remove,
    const double* pos,const double* freq,const double* dirs,const double* center,const void* gains) {
  error_text.clear();try {std::unique_ptr<Context> ctx(new Context);
    ctx->init(n,c,p,t,device,remove!=0,pos,freq,dirs,center,static_cast<const Z*>(gains));
    return ctx.release();
  }catch(const std::exception& e) {error_text=e.what();return nullptr;}
}
extern "C" int epic_fx_image(void* context,const void* voltages,float* output) {
  error_text.clear();try {if(!context) throw std::runtime_error("FX context is null/closed");
    static_cast<Context*>(context)->image(static_cast<const Z*>(voltages),output);return 0;
  }catch(const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" void epic_fx_destroy(void* context) {delete static_cast<Context*>(context);}
