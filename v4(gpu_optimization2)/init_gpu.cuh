#ifndef _INIT_GPU_H_
#define _INIT_GPU_H_

#include "global.cuh"
#include "global_gpu.cuh"
#include "utils_gpu.cuh"
#include <curand_kernel.h>
#include <omp.h>

void init_data_gpu();
void init_params_gpu();

__global__ void init_conv_b(int);
__global__ void init_conv_w(int);
__global__ void init_fc1_b(int);
__global__ void init_fc1_w(int,int);
__global__ void init_fc2_b(int);
__global__ void init_fc2_w(int);

#endif