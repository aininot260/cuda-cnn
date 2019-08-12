#include "fp_gpu.cuh"

__global__ void _set_input_train(int idx)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if(ix<ROW&&iy<COL)
    {
        _input[ix][iy]=_train_image[idx][ix][iy];
    }
}

__global__ void _set_input_test(int idx)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if(ix<ROW&&iy<COL)
    {
        _input[ix][iy]=_test_image[idx][ix][iy];
    }
}

void set_input_gpu_train(int idx)
{
    dim3 block(32,32);
    dim3 grid((ROW-1)/block.x+1,(COL-1)/block.y+1);
    _set_input_train<<<block,grid>>>(idx);
}

void set_input_gpu_test(int idx)
{
    dim3 block(32,32);
    dim3 grid((ROW-1)/block.x+1,(COL-1)/block.y+1);
    _set_input_test<<<block,grid>>>(idx);
}

__global__ void _input_conv()
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int iz=threadIdx.z+blockDim.z*blockIdx.z;
    if(ix<CONV_W_NUM&&iy<CONV_SIZE&&iz<CONV_SIZE)
    {
        _conv_z[ix][iy][iz]=0;
        // #pragma unroll
        for(int l=0;l<CONV_W_SIZE;l++)
        for(int m=0;m<CONV_W_SIZE;m++)
            _conv_z[ix][iy][iz]+=_input[iy+l][iz+m]*_conv_w[ix][l][m];
        _conv_z[ix][iy][iz]+=_conv_b[ix];
        _conv_a[ix][iy][iz]=_sigmoid(_conv_z[ix][iy][iz]);
    }
}

void input_conv_gpu()
{
    dim3 block(8,8,8);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(CONV_SIZE-1)/block.y+1,(CONV_SIZE-1)/block.z+1);
    _input_conv<<<block,grid>>>();
}

__global__ void _conv_pool()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<POOL_SIZE&&k<POOL_SIZE)
    {
        float _max=_conv_a[i][j*2][k*2];
        _pool_pos[i][j][k]=0;
        if(_conv_a[i][j*2][k*2+1]>_max)
        {
            _max=_conv_a[i][j*2][k*2+1];
            _pool_pos[i][j][k]=1;
        }
        if(_conv_a[i][j*2+1][k*2]>_max)
        {
            _max=_conv_a[i][j*2+1][k*2];
            _pool_pos[i][j][k]=2;
        }
        if(_conv_a[i][j*2+1][k*2+1]>_max)
        {
            _max=_conv_a[i][j*2+1][k*2+1];
            _pool_pos[i][j][k]=3;
        }
        _pool[i][j][k]=_max;
    }
}

void conv_pool_gpu()
{
    dim3 block(8,8,8);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(POOL_SIZE-1)/block.y+1,(POOL_SIZE-1)/block.z+1);
    _conv_pool<<<block,grid>>>();
}

__global__ void _pool_fc1()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        _fc1_z[i]=0;
        for(int j=0;j<CONV_W_NUM;j++)
        for(int k=0;k<POOL_SIZE;k++)
        for(int l=0;l<POOL_SIZE;l++)
            _fc1_z[i]+=_pool[j][k][l]*_fc1_w[i][j][k][l];
        _fc1_z[i]+=_fc1_b[i];
        _fc1_a[i]=_sigmoid(_fc1_z[i]);
    }
}

void pool_fc1_gpu()
{
    dim3 block(32);
    dim3 grid((FC1_SIZE-1)/block.x+1);
    _pool_fc1<<<block,grid>>>();
}

__global__ void _fc1_fc2()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _fc2_z[i]=0;
        for(int j=0;j<FC1_SIZE;j++)
            _fc2_z[i]+=_fc1_a[j]*_fc2_w[i][j];
        _fc2_z[i]+=_fc2_b[i];
        _fc2_a[i]=_sigmoid(_fc2_z[i]);
    }
}

void fc1_fc2_gpu()
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _fc1_fc2<<<block,grid>>>();
}

__global__ void _set_answer_train(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _output[i]=_fc2_a[i];
        _answer[i]=(_train_label[idx]==i)?1:0;
    }
}

__global__ void _set_answer_test(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _output[i]=_fc2_a[i];
        _answer[i]=(_test_label[idx]==i)?1:0;
    }
}

void set_answer_gpu_train(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _set_answer_train<<<block,grid>>>(idx);
}

void set_answer_gpu_test(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _set_answer_test<<<block,grid>>>(idx);
}

__global__ void _check_answer_get_error()
{
    float _max=_output[0];
    int max_pos=0;
    for(int i=0;i<FC2_SIZE;i++)
    {
        if(_max<_output[i])
        {
            _max=_output[i];
            max_pos=i;
        }
    }
    if(_answer[max_pos])
        _correct_cnt++;
    for(int i=0;i<FC2_SIZE;i++)
    {
        _C[i]=_output[i]-_answer[i];
        _avg_error+=_C[i]*_C[i]*0.5;
    }
}

void check_answer_get_error_gpu()
{
    _check_answer_get_error<<<1,1>>>();
}