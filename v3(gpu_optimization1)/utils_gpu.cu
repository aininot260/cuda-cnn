#include "utils_gpu.cuh"

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));
}

__device__ float _get_rand(int _rand,float fan_in)
{
    float sum=0;
    for(int i=0;i<12;i++)
        sum+=(float)_rand/RAND_MAX;
    sum-=6;
    sum*=1/sqrt(fan_in);
    return sum;
}

__device__ float _sigmoid(float x)
{
    return (1/(1+exp(-1*x)));
}