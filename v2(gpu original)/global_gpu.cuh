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

extern __constant__ float _alpha;
extern __constant__ int _minibatch;
extern __constant__ int _epochs;

extern __device__ int _correct_cnt;
extern __device__ float _avg_error;

extern __device__ float _train_image[TRAIN_NUM][ROW][COL];
extern __device__ int _train_label[TRAIN_NUM];
extern __device__ float _test_image[TEST_NUM][ROW][COL];
extern __device__ int _test_label[TEST_NUM];

extern __device__ float _conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
extern __device__ float _conv_b[CONV_W_NUM];
extern __device__ float _fc1_b[FC1_SIZE];
extern __device__ float _fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE]; 
extern __device__ float _fc2_b[FC2_SIZE];
extern __device__ float _fc2_w[FC2_SIZE][FC1_SIZE];

extern __device__ float _input[ROW][COL];
extern __device__ float _conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
extern __device__ float _conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
extern __device__ int _pool_pos[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _pool[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _fc1_z[FC1_SIZE];
extern __device__ float _fc1_a[FC1_SIZE];
extern __device__ float _fc2_z[FC2_SIZE];
extern __device__ float _fc2_a[FC2_SIZE];
extern __device__ float _output[FC2_SIZE];
extern __device__ int _answer[FC2_SIZE];

extern __device__ float _conv_dw[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
extern __device__ float _conv_db[CONV_W_NUM];
extern __device__ float _fc1_db[FC1_SIZE];
extern __device__ float _fc1_dw[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
extern __device__ float _fc2_db[FC2_SIZE];
extern __device__ float _fc2_dw[FC2_SIZE][FC1_SIZE];
extern __device__ float _C[FC2_SIZE];
extern __device__ float _fc2_delta[FC2_SIZE];
extern __device__ float _fc1_delta[FC1_SIZE];
extern __device__ float _conv_sigma_delta[CONV_W_NUM];
extern __device__ float _conv_delta[CONV_W_NUM][POOL_SIZE][POOL_SIZE];

extern __device__ int tmp;

#endif