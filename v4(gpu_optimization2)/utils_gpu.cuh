#ifndef _UTILS_GPU_H_
#define _UTILS_GPU_H_

#include "stdio.h"
#include "global_gpu.cuh"

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

void initDevice(int);

__device__ float _get_rand(int,float);
__device__ float _sigmoid(float);

#endif