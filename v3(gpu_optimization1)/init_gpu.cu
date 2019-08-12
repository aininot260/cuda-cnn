#include "init_gpu.cuh"

void init_data_gpu()
{
    CHECK(cudaMemcpyToSymbol(_train_image,train_image,TRAIN_NUM*ROW*COL*sizeof(float)));
    CHECK(cudaMemcpyToSymbol(_train_label,train_label,sizeof(train_label)));
    CHECK(cudaMemcpyToSymbol(_test_image,test_image,TEST_NUM*ROW*COL*sizeof(float)));
    CHECK(cudaMemcpyToSymbol(_test_label,test_label,sizeof(test_label)));
}

__global__ void init_conv_b(int seed)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    curandState state;
    curand_init(seed,ix,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,CONV_W_SIZE*CONV_W_SIZE);
    if(ix<CONV_W_NUM)
        _conv_b[ix]=rn;
}

__global__ void init_conv_w(int seed)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int iz=threadIdx.z+blockDim.z*blockIdx.z;
    int idx=ix+iy*CONV_W_SIZE+iz*CONV_W_SIZE*CONV_W_SIZE;
    curandState state;
    curand_init(seed,idx,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,CONV_W_SIZE*CONV_W_SIZE);
    if(ix<CONV_W_NUM&&iy<CONV_W_SIZE&&iz<CONV_W_SIZE)
        _conv_w[ix][iy][iz]=rn;
}

__global__ void init_fc1_b(int seed)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    curandState state;
    curand_init(seed,ix,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,POOL_SIZE*POOL_SIZE*CONV_W_NUM);
    if(ix<FC1_SIZE)
        _fc1_b[ix]=rn;
}

__global__ void init_fc1_w(int seed,int i)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int iz=threadIdx.z+blockDim.z*blockIdx.z;
    int idx=ix+iy*POOL_SIZE+iz*POOL_SIZE*POOL_SIZE;
    curandState state;
    curand_init(seed,idx,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,POOL_SIZE*POOL_SIZE*CONV_W_NUM);
    if(ix<CONV_W_NUM&&iy<POOL_SIZE&&iz<POOL_SIZE)
        _fc1_w[i][ix][iy][iz]=rn;
}

__global__ void init_fc2_b(int seed)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    curandState state;
    curand_init(seed,ix,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,FC1_SIZE);
    if(ix<FC2_SIZE)
        _fc2_b[ix]=rn;
}

__global__ void init_fc2_w(int seed)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*FC1_SIZE;
    curandState state;
    curand_init(seed,idx,0,&state);
    float rn=_get_rand(abs((int)curand(&state))%RAND_MAX,FC1_SIZE);
    if(ix<FC2_SIZE&&iy<FC1_SIZE)
        _fc2_w[ix][iy]=rn;
}

void init_params_gpu()
{
    srand((unsigned)time(NULL));

    dim3 block1(32);
    dim3 grid1((CONV_W_NUM-1)/block1.x+1);
    dim3 block2(32,32,32);
    dim3 grid2((CONV_W_NUM-1)/block2.x+1,(CONV_W_SIZE-1)/block2.y+1,(CONV_W_SIZE-1)/block2.z+1);
    dim3 block3(32);
    dim3 grid3((FC1_SIZE-1)/block3.x+1);
    dim3 block4(32,32,32);
    dim3 grid4((CONV_W_NUM-1)/block4.x+1,(POOL_SIZE-1)/block4.y+1,(POOL_SIZE-1)/block4.z+1);
    dim3 block5(32);
    dim3 grid5((FC2_SIZE-1)/block5.x+1);
    dim3 block6(32,32);
    dim3 grid6((FC2_SIZE-1)/block6.x+1,(FC1_SIZE-1)/block6.y+1);

    init_conv_b<<<block1,grid1>>>(rand());
    init_conv_w<<<block2,grid2>>>(rand());
    init_fc1_b<<<block3,grid3>>>(rand());
    
    // #pragma omp parallel for
    for(int i=0;i<FC1_SIZE;i++)
        init_fc1_w<<<block4,grid4>>>(rand(),i);
    init_fc2_b<<<block5,grid5>>>(rand());
    init_fc2_w<<<block6,grid6>>>(rand());
}