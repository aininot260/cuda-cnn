#include "bp_gpu.cuh"

__global__ void _update_fc2_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _fc2_delta[i]=_alpha*_C[i]*(_fc2_a[i]*(1.0-_fc2_a[i]));
        _fc2_db[i]+=_fc2_delta[i];
    }
}

void update_fc2_b_gpu()
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _update_fc2_b<<<block,grid>>>();
}

__global__ void _update_fc2_w()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j<FC1_SIZE)
        _fc2_dw[i][j]+=_fc2_delta[i]*_fc1_a[j];
}

void update_fc2_w_gpu()
{
    dim3 block(32,32);
    dim3 grid((FC2_SIZE-1)/block.x+1,(FC1_SIZE-1)/block.x+1);
    _update_fc2_w<<<block,grid>>>();
}

__global__ void _update_fc1_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        float error=0;
        for(int j=0;j<FC2_SIZE;j++)
            error+=_fc2_delta[j]*_fc2_w[j][i];
        _fc1_delta[i]=error*(_fc1_a[i]*(1.0-_fc1_a[i]));
        _fc1_db[i]+=_fc1_delta[i];
    }
}

void update_fc1_b_gpu()
{
    dim3 block(32);
    dim3 grid((FC1_SIZE-1)/block.x+1);
    _update_fc1_b<<<block,grid>>>();
}

__global__ void _update_fc1_w(int j)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    int l=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<FC1_SIZE&&k<POOL_SIZE&&l<POOL_SIZE)
        _fc1_dw[i][j][k][l]+=_fc1_delta[i]*_pool[j][k][l];
}

void update_fc1_w_gpu()
{
    dim3 block(8,8,8);
    dim3 grid((FC1_SIZE-1)/block.x+1,(POOL_SIZE-1)/block.y+1,(POOL_SIZE-1)/block.z+1);

    // #pragma omp parallel for
    for(int j=0;j<CONV_W_NUM;j++)
        _update_fc1_w<<<block,grid>>>(j);
}

__global__ void _update_conv_b_inner(int i)
{
    int j=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    if(j<POOL_SIZE&&k<POOL_SIZE)
    {
        float error=0;
        _conv_delta[i][j][k]=0;
        for(int l=0;l<FC1_SIZE;l++)
            error+=_fc1_delta[l]*_fc1_w[l][i][j][k];
        _conv_delta[i][j][k]=error*(_pool[i][j][k]*(1.0-_pool[i][j][k]));
        _conv_sigma_delta[i]+=error*(_pool[i][j][k]*(1.0-_pool[i][j][k]));
    }
}

__global__ void _update_conv_b_final()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<CONV_W_NUM)
        _conv_db[i]+=_conv_sigma_delta[i];
}

__global__ void _update_conv_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<CONV_W_NUM)
    {
        _conv_sigma_delta[i]=0;
        dim3 block(32,32);
        dim3 grid((POOL_SIZE-1)/block.x+1,(POOL_SIZE-1)/block.y+1);
        _update_conv_b_inner<<<block,grid>>>(i);
    }
}

void update_conv_b_gpu()
{
    dim3 block(32);
    dim3 grid((CONV_W_NUM-1)/block.x+1);
    _update_conv_b<<<block,grid>>>();
    _update_conv_b_final<<<block,grid>>>();
}

__global__ void _update_conv_w()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<CONV_W_SIZE&&k<CONV_W_SIZE)
    {
        float error=0;
        for(int m=0;m<POOL_SIZE;m++)
        for(int n=0;n<POOL_SIZE;n++)
        {
            int x=_pool_pos[i][m][n]/2;
            int y=_pool_pos[i][m][n]%2;
            error+=_conv_delta[i][m][n]*_input[2*m+j+x][2*n+k+y];
        }
        _conv_dw[i][j][k]+=error;
    }
}

void update_conv_w_gpu()
{
    dim3 block(8,8,8);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(CONV_W_SIZE-1)/block.y+1,(CONV_W_SIZE-1)/block.z+1);
    _update_conv_w<<<block,grid>>>();
}

__global__ void assign_fc2_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _fc2_b[i]-=(_fc2_db[i]/_minibatch);
        _fc2_db[i]=0;
    }
}

__global__ void assign_fc2_w()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j<FC1_SIZE)
    {
        _fc2_w[i][j]-=(_fc2_dw[i][j]/_minibatch);
        _fc2_dw[i][j]=0;
    }
}

__global__ void assign_fc1_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        _fc1_b[i]-=(_fc1_db[i]/_minibatch);
        _fc1_db[i]=0;
    }
}

__global__ void assign_fc1_w(int j)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    int l=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<FC1_SIZE&&k<POOL_SIZE&&l<POOL_SIZE)
    {
        _fc1_w[i][j][k][l]-=(_fc1_dw[i][j][k][l]/_minibatch);
        _fc1_dw[i][j][k][l]=0;
    }
}

__global__ void assign_conv_b()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<CONV_W_NUM)
    {
        _conv_b[i]-=(_conv_db[i]/_minibatch);
        _conv_db[i]=0;
    }
}

__global__ void assign_conv_w()
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int l=threadIdx.y+blockDim.y*blockIdx.y;
    int m=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&l<CONV_W_SIZE&&m<CONV_W_SIZE)
    {
        _conv_w[i][l][m]-=(_conv_dw[i][l][m]/_minibatch);
        _conv_dw[i][l][m]=0;
    }
}

void assign_grads_gpu()
{
    dim3 block1(32);
    dim3 grid1((FC2_SIZE-1)/block1.x+1);
    assign_fc2_b<<<block1,grid1>>>();

    dim3 block2(32,32);
    dim3 grid2((FC2_SIZE-1)/block2.x+1,(FC1_SIZE-1)/block2.y+1);
    assign_fc2_w<<<block2,grid2>>>();

    dim3 block3(32);
    dim3 grid3((FC1_SIZE-1)/block3.x+1);
    assign_fc1_b<<<block3,grid3>>>();

    dim3 block4(8,8,8);
    dim3 grid4((FC1_SIZE-1)/block4.x+1,(POOL_SIZE-1)/block4.y+1,(POOL_SIZE-1)/block4.z+1);
    for(int j=0;j<CONV_W_NUM;j++)
        assign_fc1_w<<<block4,grid4>>>(j);

    dim3 block5(32);
    dim3 grid5((CONV_W_NUM-1)/block5.x+1);
    assign_conv_b<<<block5,grid5>>>();

    dim3 block6(8,8,8);
    dim3 grid6((CONV_W_NUM-1)/block6.x+1,(CONV_W_SIZE-1)/block6.y+1,(CONV_W_SIZE-1)/block6.z+1);
    assign_conv_w<<<block6,grid6>>>();
}