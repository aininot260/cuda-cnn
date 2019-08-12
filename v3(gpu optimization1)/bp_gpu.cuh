#ifndef _BP_GPU_H_
#define _BP_GPU_H_

#include "stdio.h"
#include "global_gpu.cuh"
#include "utils_gpu.cuh"

void update_fc2_b_gpu();
void update_fc2_w_gpu();
void update_fc1_b_gpu();
void update_fc1_w_gpu();
void update_conv_b_gpu();
void update_conv_w_gpu();
void assign_grads_gpu();

#endif