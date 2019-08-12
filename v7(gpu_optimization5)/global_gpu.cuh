#ifndef _GLOBAL_GPU_H_
#define _GLOBAL_GPU_H_

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define ROW 28
#define COL 28
#define CONV_SIZE 24
#define POOL_SIZE 12
#define FC1_SIZE 45
#define FC2_SIZE 10
#define CONV_W_SIZE 5
#define CONV_W_NUM 6
#define N_STREAM 16

extern cudaStream_t *stream;

extern __constant__ int _minibatch;

extern __device__ float _input[N_STREAM][ROW][COL];
extern __device__ int _pool_pos[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _pool[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _fc1_a[N_STREAM][FC1_SIZE];
extern __device__ float _fc2_a[N_STREAM][FC2_SIZE];

extern __device__ float _conv_dw[N_STREAM][CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
extern __device__ float _conv_db[N_STREAM][CONV_W_NUM];
extern __device__ float _fc1_db[N_STREAM][FC1_SIZE];
extern __device__ float _fc1_dw[N_STREAM][FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _fc2_db[N_STREAM][FC2_SIZE];
extern __device__ float _fc2_dw[N_STREAM][FC2_SIZE][FC1_SIZE];
extern __device__ float _C[N_STREAM][FC2_SIZE];
extern __device__ float _fc1_delta[N_STREAM][FC1_SIZE];

#endif