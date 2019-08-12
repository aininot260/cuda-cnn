#ifndef _FP_H_
#define _FP_H_

#include "utils.cuh"
#include "global.cuh"

void set_input(int,float[][ROW][COL]);
void input_conv();
void conv_pool();
void pool_fc1();
void fc1_fc2();
void set_answer(int,int[]);
void check_answer(int&);
void get_error(float&);

#endif