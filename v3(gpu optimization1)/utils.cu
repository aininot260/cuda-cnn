#include "utils.cuh"

int swap_endian(int val)
{
    unsigned char c1,c2,c3,c4;
    c1=val&255;
    c2=(val>>8)&255;
    c3=(val>>16)&255;
    c4=(val>>24)&255;
    return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}

float get_rand(float fan_in)
{
    float sum=0;
    for(int i=0;i<12;i++)
        sum+=(float)rand()/RAND_MAX;
    sum-=6;
    sum*=1/sqrt(fan_in);
    return sum;
}

float sigmoid(float x)
{
    return (1/(1+exp(-1*x)));
}