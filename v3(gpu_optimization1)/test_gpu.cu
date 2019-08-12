#include "test_gpu.cuh"

__global__ void test_gpu()
{
    printf("%f %d %d\n",_alpha,_epochs,_minibatch);
    printf("%d\n",tmp);
    tmp=18;
    printf("%d\n",tmp);
}

__global__ void test_gpu1()
{
    printf("====\n");
    printf("%d\n",tmp);
    tmp=19;
    printf("%d\n",tmp);
}