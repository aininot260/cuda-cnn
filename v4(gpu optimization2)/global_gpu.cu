#include "global_gpu.cuh"

cudaStream_t *stream;

__constant__ float _alpha;
__constant__ int _minibatch;
__constant__ int _epochs;

__device__ int _correct_cnt;
__device__ float _avg_error;

__device__ float _train_image[TRAIN_NUM][ROW][COL];
__device__ int _train_label[TRAIN_NUM];
__device__ float _test_image[TEST_NUM][ROW][COL];
__device__ int _test_label[TEST_NUM];

__device__ float _conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__device__ float _conv_b[CONV_W_NUM];
__device__ float _fc1_b[FC1_SIZE];
__device__ float _fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE]; 
__device__ float _fc2_b[FC2_SIZE];
__device__ float _fc2_w[FC2_SIZE][FC1_SIZE];

__device__ float _input[N_STREAM][ROW][COL];
__device__ float _conv_z[N_STREAM][CONV_W_NUM][CONV_SIZE][CONV_SIZE];
__device__ float _conv_a[N_STREAM][CONV_W_NUM][CONV_SIZE][CONV_SIZE];
__device__ int _pool_pos[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _pool[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc1_z[N_STREAM][FC1_SIZE];
__device__ float _fc1_a[N_STREAM][FC1_SIZE];
__device__ float _fc2_z[N_STREAM][FC2_SIZE];
__device__ float _fc2_a[N_STREAM][FC2_SIZE];
__device__ float _output[N_STREAM][FC2_SIZE];
__device__ int _answer[N_STREAM][FC2_SIZE];

__device__ float _conv_dw[N_STREAM][CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__device__ float _conv_db[N_STREAM][CONV_W_NUM];
__device__ float _fc1_db[N_STREAM][FC1_SIZE];
__device__ float _fc1_dw[N_STREAM][FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc2_db[N_STREAM][FC2_SIZE];
__device__ float _fc2_dw[N_STREAM][FC2_SIZE][FC1_SIZE];
__device__ float _C[N_STREAM][FC2_SIZE];
__device__ float _fc2_delta[N_STREAM][FC2_SIZE];
__device__ float _fc1_delta[N_STREAM][FC1_SIZE];
__device__ float _conv_sigma_delta[N_STREAM][CONV_W_NUM];
__device__ float _conv_delta[N_STREAM][CONV_W_NUM][POOL_SIZE][POOL_SIZE];

__device__ int tmp;