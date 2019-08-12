#include "bp_gpu.cuh"

__global__ void _update_fc2_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        _fc2_delta[idx%N_STREAM][i]=alpha*_C[idx%N_STREAM][i]*(_fc2_a[idx%N_STREAM][i]*(1.0-_fc2_a[idx%N_STREAM][i]));
        _fc2_db[idx%N_STREAM][i]+=_fc2_delta[idx%N_STREAM][i];
    }
}

void update_fc2_b_gpu(int idx)
{
    dim3 block(32);
    dim3 grid((FC2_SIZE-1)/block.x+1);
    _update_fc2_b<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _update_fc2_w(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j<FC1_SIZE)
        _fc2_dw[idx%N_STREAM][i][j]+=_fc2_delta[idx%N_STREAM][i]*_fc1_a[idx%N_STREAM][j];
}

void update_fc2_w_gpu(int idx)
{
    dim3 block(5,64);
    dim3 grid((FC2_SIZE-1)/block.x+1,(FC1_SIZE-1)/block.x+1);
    _update_fc2_w<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _update_fc1_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        float error=0;
        #pragma unroll
        for(int j=0;j<FC2_SIZE;j++)
            error+=_fc2_delta[idx%N_STREAM][j]*fc2_w[j][i];
        _fc1_delta[idx%N_STREAM][i]=error*(_fc1_a[idx%N_STREAM][i]*(1.0-_fc1_a[idx%N_STREAM][i]));
        _fc1_db[idx%N_STREAM][i]+=_fc1_delta[idx%N_STREAM][i];
    }
}

void update_fc1_b_gpu(int idx)
{
    dim3 block(64);
    dim3 grid((FC1_SIZE-1)/block.x+1);
    _update_fc1_b<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _update_fc1_w(int j,int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    int l=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<FC1_SIZE&&k<POOL_SIZE&&l<POOL_SIZE)
        _fc1_dw[idx%N_STREAM][i][j][k][l]+=_fc1_delta[idx%N_STREAM][i]*_pool[idx%N_STREAM][j][k][l];
}

void update_fc1_w_gpu(int idx)
{
    dim3 block(16,4,4);
    dim3 grid((FC1_SIZE-1)/block.x+1,(POOL_SIZE-1)/block.y+1,(POOL_SIZE-1)/block.z+1);
    #pragma omp parallel for
    for(int j=0;j<CONV_W_NUM;j++)
        _update_fc1_w<<<block,grid,0,stream[idx%N_STREAM]>>>(j,idx);
}

__global__ void _update_conv_delta(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<CONV_W_NUM&&j<POOL_SIZE&&k<POOL_SIZE)
    {
        float error=0;
        _conv_delta[idx%N_STREAM][i][j][k]=0;
        #pragma unroll
        for(int l=0;l<FC1_SIZE;l++)
            error+=_fc1_delta[idx%N_STREAM][l]*fc1_w[l][i][j][k];
        _conv_delta[idx%N_STREAM][i][j][k]=error*(_pool[idx%N_STREAM][i][j][k]*(1.0-_pool[idx%N_STREAM][i][j][k]));
    }
}

__global__ void _update_conv_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<CONV_W_NUM)
    {
        _conv_sigma_delta[idx%N_STREAM][i]=0;
        #pragma unroll
        for(int j=0;j<POOL_SIZE;j++)
        #pragma unroll
        for(int k=0;k<POOL_SIZE;k++)
            _conv_sigma_delta[idx%N_STREAM][i]+=_conv_delta[idx%N_STREAM][i][j][k];
        _conv_db[idx%N_STREAM][i]+=_conv_sigma_delta[idx%N_STREAM][i];
    }
}

void update_conv_b_gpu(int idx)
{
    dim3 block1(1,16,16);
    dim3 grid1((CONV_W_NUM-1)/block1.x+1,(POOL_SIZE-1)/block1.y+1,(POOL_SIZE-1)/block1.z+1);
    dim3 block2(32);
    dim3 grid2((CONV_W_NUM-1)/block2.x+1);
    _update_conv_delta<<<block1,grid1,0,stream[idx%N_STREAM]>>>(idx);
    _update_conv_b<<<block2,grid2,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void _update_conv_w(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<CONV_W_SIZE&&k<CONV_W_SIZE)
    {
        float error=0;
        #pragma unroll
        for(int m=0;m<POOL_SIZE;m++)
        #pragma unroll
        for(int n=0;n<POOL_SIZE;n++)
        {
            int x=_pool_pos[idx%N_STREAM][i][m][n]/2;
            int y=_pool_pos[idx%N_STREAM][i][m][n]%2;
            error+=_conv_delta[idx%N_STREAM][i][m][n]*_input[idx%N_STREAM][2*m+j+x][2*n+k+y];
        }
        _conv_dw[idx%N_STREAM][i][j][k]+=error;
    }
}

void update_conv_w_gpu(int idx)
{
    dim3 block(5,8,8);
    dim3 grid((CONV_W_NUM-1)/block.x+1,(CONV_W_SIZE-1)/block.y+1,(CONV_W_SIZE-1)/block.z+1);
    _update_conv_w<<<block,grid,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void assign_fc2_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC2_SIZE)
    {
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            fc2_b[i]-=(_fc2_db[(idx-j)%N_STREAM][i]/_minibatch);
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            _fc2_db[(idx-j)%N_STREAM][i]=0;
    }
}

__global__ void assign_fc2_w(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j<FC1_SIZE)
    {
        #pragma unroll
        for(int k=0;k<_minibatch;k++)
            fc2_w[i][j]-=(_fc2_dw[(idx-k)%N_STREAM][i][j]/_minibatch);
        #pragma unroll
        for(int k=0;k<_minibatch;k++)
            _fc2_dw[(idx-k)%N_STREAM][i][j]=0;
    }
}

__global__ void assign_fc1_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            fc1_b[i]-=(_fc1_db[(idx-j)%N_STREAM][i]/_minibatch);
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            _fc1_db[(idx-j)%N_STREAM][i]=0;
    }
}

__global__ void assign_fc1_w(int j,int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int k=threadIdx.y+blockDim.y*blockIdx.y;
    int l=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<FC1_SIZE&&k<POOL_SIZE&&l<POOL_SIZE)
    {
        #pragma unroll
        for(int m=0;m<_minibatch;m++)
            fc1_w[i][j][k][l]-=(_fc1_dw[(idx-m)%N_STREAM][i][j][k][l]/_minibatch);
        #pragma unroll
        for(int m=0;m<_minibatch;m++)
            _fc1_dw[(idx-m)%N_STREAM][i][j][k][l]=0;
    }
}

__global__ void assign_conv_b(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<CONV_W_NUM)
    {
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            conv_b[i]-=(_conv_db[(idx-j)%N_STREAM][i]/_minibatch);
        #pragma unroll
        for(int j=0;j<_minibatch;j++)
            _conv_db[(idx-j)%N_STREAM][i]=0;
    }
}

__global__ void assign_conv_w(int idx)
{
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<CONV_W_SIZE&&k<CONV_W_SIZE)
    {
        #pragma unroll
        for(int l=0;l<_minibatch;l++)
            conv_w[i][j][k]-=(_conv_dw[(idx-l)%N_STREAM][i][j][k]/_minibatch);
        #pragma unroll
        for(int l=0;l<_minibatch;l++)
            _conv_dw[(idx-l)%N_STREAM][i][j][k]=0;
    }
}

void assign_grads_gpu(int idx)
{
    dim3 block1(32);
    dim3 grid1((FC2_SIZE-1)/block1.x+1);

    dim3 block2(5,64);
    dim3 grid2((FC2_SIZE-1)/block2.x+1,(FC1_SIZE-1)/block2.y+1);

    dim3 block3(64);
    dim3 grid3((FC1_SIZE-1)/block3.x+1);

    dim3 block4(16,4,4);
    dim3 grid4((FC1_SIZE-1)/block4.x+1,(POOL_SIZE-1)/block4.y+1,(POOL_SIZE-1)/block4.z+1);

    dim3 block5(32);
    dim3 grid5((CONV_W_NUM-1)/block5.x+1);

    dim3 block6(8,8,8);
    dim3 grid6((CONV_W_NUM-1)/block6.x+1,(CONV_W_SIZE-1)/block6.y+1,(CONV_W_SIZE-1)/block6.z+1);
    
    assign_fc2_b<<<block1,grid1>>>(idx);
    assign_fc2_w<<<block2,grid2>>>(idx);
    assign_fc1_b<<<block3,grid3>>>(idx);
    #pragma omp parallel for
    for(int j=0;j<CONV_W_NUM;j++)
        assign_fc1_w<<<block4,grid4>>>(j,idx);
    assign_conv_b<<<block5,grid5>>>(idx);
    assign_conv_w<<<block6,grid6>>>(idx);
}