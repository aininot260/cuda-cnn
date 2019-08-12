#ifndef _BP_GPU_H_
#define _BP_GPU_H_

#include "stdio.h"
#include "global_gpu.cuh"
#include "utils_gpu.cuh"

void update_fc2_b_gpu(int idx);
void update_fc2_w_gpu(int idx);
void update_fc1_b_gpu(int idx);
void update_fc1_w_gpu(int idx);
void update_conv_b_gpu(int idx);
void update_conv_w_gpu(int idx);
void assign_grads_gpu(int idx);

#endif