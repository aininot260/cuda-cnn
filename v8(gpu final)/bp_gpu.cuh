#ifndef _BP_GPU_H_
#define _BP_GPU_H_

#include "global.cuh"
#include "global_gpu.cuh"

__global__ void bp_update_fc(int);
__global__ void bp_update_conv(int);
__global__ void bp_assign_grads_fc(int,int);
__global__ void bp_assign_grads_conv(int,int);

void bp_update_gpu(int);
void bp_assign_grads_gpu(int);

#endif