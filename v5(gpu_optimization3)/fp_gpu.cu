#include "fp_gpu.cuh"

__global__ void _set_input_train(int idx)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if(ix<ROW&&iy<COL)
    {
        _input[idx%N_STREAM][ix][iy]=train_image[idx][ix][iy];
    }
}

__global__ void _set_input_test(int idx)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if(ix<ROW&&iy<COL)
    {
        _input[idx%N_STREAM][ix][iy]=test_image[idx][ix][iy];
    }
}

void set_input_gpu_train(int idx)
{
    dim3 block(16,16);
    dim3 grid((ROW-1)/block.x+1,(COL-1)/block.y+1);
    _set_input_train<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

void set_input_gpu_test(int idx)
{
    dim3 block(16,16);
    dim3 grid((ROW-1)/block.x+1,(COL-1)/block.y+1);
    _set_input_test<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _input_conv(int idx)
{
    __shared__ float tile[CONV_SIZE][CONV_SIZE];
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int iz=threadIdx.z+blockDim.z*blockIdx.z;
    if(ix<CONV_W_NUM&&iy<CONV_SIZE&&iz<CONV_SIZE)
    {
        tile[iy][iz]=0;
        #pragma unroll
        for(int l=0;l<CONV_W_SIZE;l++)
        #pragma unroll
        for(int m=0;m<CONV_W_SIZE;m++)
            tile[iy][iz]+=__ldg(&_input[idx%N_STREAM][iy+l][iz+m])*conv_w[ix][l][m];
        tile[iy][iz]+=conv_b[ix];
        _conv_z[idx%N_STREAM][ix][iy][iz]=tile[iy][iz];
        _conv_a[idx%N_STREAM][ix][iy][iz]=sigmoid(tile[iy][iz]);
    }
}

void input_conv_gpu(int idx)
{
    dim3 block(1,32,32);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(CONV_SIZE-1)/block.y+1,(CONV_SIZE-1)/block.z+1);
    _input_conv<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _conv_pool(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int k=threadIdx.z+blockDim.z*blockIdx.z;
    __shared__ float tile[POOL_SIZE][POOL_SIZE];
    if(i<CONV_W_NUM&&j<POOL_SIZE&&k<POOL_SIZE)
    {
        float _max=_conv_a[idx%N_STREAM][i][j*2][k*2];
        tile[j][k]=0;
        if(_conv_a[idx%N_STREAM][i][j*2][k*2+1]>_max)
        {
            _max=_conv_a[idx%N_STREAM][i][j*2][k*2+1];
            tile[j][k]=1;
        }
        if(_conv_a[idx%N_STREAM][i][j*2+1][k*2]>_max)
        {
            _max=_conv_a[idx%N_STREAM][i][j*2+1][k*2];
            tile[j][k]=2;
        }
        if(_conv_a[idx%N_STREAM][i][j*2+1][k*2+1]>_max)
        {
            _max=_conv_a[idx%N_STREAM][i][j*2+1][k*2+1];
            tile[j][k]=3;
        }
        _pool_pos[idx%N_STREAM][i][j][k]=tile[j][k];
        _pool[idx%N_STREAM][i][j][k]=_max;
    }
}

void conv_pool_gpu(int idx)
{
    dim3 block(1,16,16);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(POOL_SIZE-1)/block.y+1,(POOL_SIZE-1)/block.z+1);
    _conv_pool<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _pool_fc1(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ float tile[FC1_SIZE];
    if(i<FC1_SIZE)
    {
        tile[i]=0;
        #pragma unroll
        for(int j=0;j<CONV_W_NUM;j++)
        #pragma unroll
        for(int k=0;k<POOL_SIZE;k++)
        #pragma unroll
        for(int l=0;l<POOL_SIZE;l++)
            tile[i]+=_pool[idx%N_STREAM][j][k][l]*fc1_w[i][j][k][l];
        tile[i]+=fc1_b[i];
        _fc1_z[idx%N_STREAM][i]=tile[i];
        _fc1_a[idx%N_STREAM][i]=sigmoid(tile[i]);
    }
}

void pool_fc1_gpu(int idx)
{
    dim3 block(64);
    dim3 grid((FC1_SIZE-1)/block.x+1);
    _pool_fc1<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _fc1_fc2(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ float tile[FC2_SIZE];
    if(i<FC2_SIZE)
    {
        tile[i]=0;
        #pragma unroll
        for(int j=0;j<FC1_SIZE;j++)
            tile[i]+=_fc1_a[idx%N_STREAM][j]*fc2_w[i][j];
        tile[i]+=fc2_b[i];
        _fc2_z[idx%N_STREAM][i]=tile[i];
        _fc2_a[idx%N_STREAM][i]=sigmoid(tile[i]);
    }
}

void fc1_fc2_gpu(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _fc1_fc2<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _set_answer_train(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _output[idx%N_STREAM][i]=_fc2_a[idx%N_STREAM][i];
        _answer[idx%N_STREAM][i]=(train_label[idx]==i)?1:0;
    }
}

__global__ void _set_answer_test(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _output[idx%N_STREAM][i]=_fc2_a[idx%N_STREAM][i];
        _answer[idx%N_STREAM][i]=(test_label[idx]==i)?1:0;
    }
}

void set_answer_gpu_train(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _set_answer_train<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

void set_answer_gpu_test(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _set_answer_test<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _check_answer_get_error(int idx)
{
    float _max=_output[idx%N_STREAM][0];
    int max_pos=0;
    #pragma unroll
    for(int i=0;i<FC2_SIZE;i++)
    {
        if(_max<_output[idx%N_STREAM][i])
        {
            _max=_output[idx%N_STREAM][i];
            max_pos=i;
        }
    }
    if(_answer[idx%N_STREAM][max_pos])
        atomicAdd(&correct_cnt,1);
    #pragma unroll
    for(int i=0;i<FC2_SIZE;i++)
    {
        _C[idx%N_STREAM][i]=_output[idx%N_STREAM][i]-_answer[idx%N_STREAM][i];
        atomicExch(&avg_error,avg_error+_C[idx%N_STREAM][i]*_C[idx%N_STREAM][i]*0.5);
    }
}

void check_answer_get_error_gpu(int idx)
{
    _check_answer_get_error<<<1,1,0,stream[idx%N_STREAM]>>>(idx);
}