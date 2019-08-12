#ifndef _FP_GPU_H_
#define _FP_GPU_H_

#include "stdio.h"
#include "global_gpu.cuh"
#include "utils_gpu.cuh"

void set_input_gpu_train(int idx);
void set_input_gpu_test(int idx);
void input_conv_gpu();
void conv_pool_gpu();
void pool_fc1_gpu();
void fc1_fc2_gpu();
void set_answer_gpu_train(int idx);
void set_answer_gpu_test(int idx);
void check_answer_get_error_gpu();
#endif