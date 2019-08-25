#ifndef _UTILS_H_
#define _UITLS_H_

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

#include "stdio.h"

void initDevice(int);
int swap_endian(int);
float get_rand(float);

__host__ __device__ float sigmoid(float);

#endif