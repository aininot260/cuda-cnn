#include "global_gpu.cuh"

cudaStream_t *stream;

__device__ float _input[N_STREAM][ROW][COL];
__device__ int _pool_pos[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _pool[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc1_a[N_STREAM][FC1_SIZE];
__device__ float _fc2_a[N_STREAM][FC2_SIZE];

__device__ float _conv_dw[N_STREAM][CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__device__ float _conv_db[N_STREAM][CONV_W_NUM];
__device__ float _fc1_db[N_STREAM][FC1_SIZE];
__device__ float _fc1_dw[N_STREAM][FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc2_db[N_STREAM][FC2_SIZE];
__device__ float _fc2_dw[N_STREAM][FC2_SIZE][FC1_SIZE];
__device__ float _C[N_STREAM][FC2_SIZE];
__device__ float _fc1_delta[N_STREAM][FC1_SIZE];