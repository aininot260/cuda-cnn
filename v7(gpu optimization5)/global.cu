#include "global.cuh"

int epochs=20;
int minibatch=1;

__managed__ float alpha=0.2;
__managed__ int correct_cnt=0;
__managed__ float avg_error=0;

__managed__ float train_image[TRAIN_NUM][ROW][COL];
__managed__ int train_label[TRAIN_NUM];
__managed__ float test_image[TEST_NUM][ROW][COL];
__managed__ int test_label[TEST_NUM];

__managed__ float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__managed__ float conv_b[CONV_W_NUM];
__managed__ float fc1_b[FC1_SIZE];
__managed__ float fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE]; 
__managed__ float fc2_b[FC2_SIZE];
__managed__ float fc2_w[FC2_SIZE][FC1_SIZE];

float input[ROW][COL];
float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
float conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
int pool_pos[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float pool[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float fc1_z[FC1_SIZE];
float fc1_a[FC1_SIZE];
float fc2_z[FC2_SIZE];
float fc2_a[FC2_SIZE];
float output[FC2_SIZE];
int answer[FC2_SIZE];

float conv_dw[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
float conv_db[CONV_W_NUM];
float fc1_db[FC1_SIZE];
float fc1_dw[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float fc2_db[FC2_SIZE];
float fc2_dw[FC2_SIZE][FC1_SIZE];
float C[FC2_SIZE];
float fc2_delta[FC2_SIZE];
float fc1_delta[FC1_SIZE];
float conv_sigma_delta[CONV_W_NUM];
float conv_delta[CONV_W_NUM][POOL_SIZE][POOL_SIZE];