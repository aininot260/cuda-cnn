#ifndef _FP_GPU_H_
#define _FP_GPU_H_

#include "utils.cuh"
#include "global.cuh"
#include "global_gpu.cuh"

__global__ void fp_conv_pool(int,bool);
__global__ void fp_fc_answer(int,bool);

void fp_conv_pool_gpu(int,bool);
void fp_fc_answer_gpu(int,bool);

#endif