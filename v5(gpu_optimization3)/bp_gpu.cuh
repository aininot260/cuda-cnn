#ifndef _BP_GPU_H_
#define _BP_GPU_H_

#include "omp.h"

#include "global.cuh"
#include "global_gpu.cuh"

__global__ void _update_fc2_b(int);
__global__ void _update_fc2_w(int);
__global__ void _update_fc1_b(int);
__global__ void _update_fc1_w(int,int);
__global__ void _update_conv_delta(int);
__global__ void _update_conv_b(int);
__global__ void _update_conv_w(int);
__global__ void assign_fc2_b(int);
__global__ void assign_fc2_w(int);
__global__ void assign_fc1_b(int);
__global__ void assign_fc1_w(int,int);
__global__ void assign_conv_b(int);
__global__ void assign_conv_w(int);

void update_fc2_b_gpu(int);
void update_fc2_w_gpu(int);
void update_fc1_b_gpu(int);
void update_fc1_w_gpu(int);
void update_conv_b_gpu(int);
void update_conv_w_gpu(int);
void assign_grads_gpu(int);

#endif