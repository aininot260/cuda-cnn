#ifndef _FP_GPU_H_
#define _FP_GPU_H_

#include "utils.cuh"
#include "global.cuh"
#include "global_gpu.cuh"

__global__ void _set_input_train(int);
__global__ void _set_input_test(int);
__global__ void _input_conv(int);
__global__ void _conv_pool(int);
__global__ void _pool_fc1(int);
__global__ void _fc1_fc2(int);
__global__ void _set_answer_train(int);
__global__ void _set_answer_test(int);
__global__ void _check_answer_get_error(int);

void set_input_gpu_train(int);
void set_input_gpu_test(int);
void input_conv_gpu(int);
void conv_pool_gpu(int);
void pool_fc1_gpu(int);
void fc1_fc2_gpu(int);
void set_answer_gpu_train(int);
void set_answer_gpu_test(int);
void check_answer_get_error_gpu(int);

#endif