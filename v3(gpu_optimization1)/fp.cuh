#ifndef _FP_H_
#define _FP_H_

#include "global.cuh"
#include "utils.cuh"

void set_input(int,float [TRAIN_NUM][ROW][COL]);
void input_conv();
void conv_pool();
void pool_fc1();
void fc1_fc2();
void set_answer(int,int [TRAIN_NUM]);
void check_answer(int &);
void get_error(float &);

#endif