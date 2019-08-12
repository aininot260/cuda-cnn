#ifndef _BP_H_
#define _BP_H_

#include "utils.cuh"
#include "global.cuh"

void update_fc2_b();
void update_fc2_w();
void update_fc1_b();
void update_fc1_w();
void update_conv_b();
void update_conv_w();
void assign_grads();

#endif