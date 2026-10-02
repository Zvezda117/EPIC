// Exercise the actual GCF precomputation with a constant-one CUDA texture.
#include "cu_helpers.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
static void check(cudaError_t e,const char* what) {
  if(e!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
#define CUDA(call) check((call),#call)
int main(int argc,char**argv) {
  int device=0;
  if(argc==3 && std::string(argv[1])=="--device") device=std::stoi(argv[2]);
  else if(argc!=1) return 2;
  try {
    CUDA(cudaSetDevice(device));
    constexpr int channels=2,ants=16,grid=64,dim=4;
    static_assert(LWA_SV_NSTANDS==ants,"Fixture requires 16-slot array");
    cudaArray_t texture_array=nullptr;
    const auto channel_desc=cudaCreateChannelDesc<float>();
    CUDA(cudaMallocArray(&texture_array,&channel_desc,dim,dim));
    std::vector<float> ones(dim*dim,1);
    CUDA(cudaMemcpy2DToArray(texture_array,0,0,ones.data(),dim*sizeof(float),
                            dim*sizeof(float),dim,cudaMemcpyHostToDevice));
    cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;
    resource.res.array.array=texture_array;
    cudaTextureDesc description{};
    description.addressMode[0]=description.addressMode[1]=cudaAddressModeClamp;
    description.filterMode=cudaFilterModeLinear;
    description.readMode=cudaReadModeElementType;
    cudaTextureObject_t texture=0;
    CUDA(cudaCreateTextureObject(&texture,&resource,&description,nullptr));
    std::vector<float3> positions(channels*ants,float3{32.25f,32.25f,0});
    float3* gpu_positions=nullptr;
    CUDA(cudaMalloc(&gpu_positions,positions.size()*sizeof(float3)));
    CUDA(cudaMemcpy(gpu_positions,positions.data(),positions.size()*sizeof(float3),cudaMemcpyHostToDevice));
    bool passed=true;
    for(int support:{1,3,5,9,10}) {
      const int ne=support*support,block=(1024/ne)*ne,groups=block/ne;
      std::vector<float> elements(channels*ants*ne);
      float* gpu_elements=nullptr;CUDA(cudaMalloc(&gpu_elements,elements.size()*sizeof(float)));
      CUDA(cudaMemset(gpu_elements,0xff,elements.size()*sizeof(float)));
      ComputeGcfElements<<<channels,block,ants*sizeof(float)>>>(gpu_elements,
          reinterpret_cast<float*>(gpu_positions),54136,1,texture,grid,support,ants);
      CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
      CUDA(cudaMemcpy(elements.data(),gpu_elements,elements.size()*sizeof(float),cudaMemcpyDeviceToHost));
      double worst_sum=0,worst_cell=0;bool okay=true;
      for(int ca=0;ca<channels*ants;++ca) {
        double sum=0;
        for(int e=0;e<ne;++e) {
          const float v=elements[ca*ne+e];sum+=v;
          if(!std::isfinite(v)||v<0) okay=false;
          worst_cell=std::max(worst_cell,std::abs(double(v)-1.0/ne));
        }
        worst_sum=std::max(worst_sum,std::abs(sum-1));
      }
      okay &= worst_sum<2e-6 && worst_cell<2e-7;
      passed &= okay;
      std::printf("gcf support=%d groups=%d rounds=%d %s sum_error=%.9g cell_error=%.9g\n",
          support,groups,(ants+groups-1)/groups,okay?"PASS":"FAIL",worst_sum,worst_cell);
      CUDA(cudaFree(gpu_elements));
    }
    CUDA(cudaFree(gpu_positions));CUDA(cudaDestroyTextureObject(texture));
    CUDA(cudaFreeArray(texture_array));return passed?0:1;
  } catch(const std::exception&e) {std::fprintf(stderr,"GCF/CUDA failure: %s\n",e.what());return 3;}
}
