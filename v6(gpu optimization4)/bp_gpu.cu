#include "bp_gpu.cuh"

__global__ void bp_update_fc(int idx)
{
    int i,j,k,l,m,n;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j==0)
    {
        _fc2_delta[idx%N_STREAM][i]=alpha*_C[idx%N_STREAM][i]*(_fc2_a[idx%N_STREAM][i]*(1.0-_fc2_a[idx%N_STREAM][i]));
        _fc2_db[idx%N_STREAM][i]+=_fc2_delta[idx%N_STREAM][i];
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    __syncthreads();
    if(i<FC2_SIZE&&j<FC1_SIZE)
        _fc2_dw[idx%N_STREAM][i][j]+=_fc2_delta[idx%N_STREAM][i]*_fc1_a[idx%N_STREAM][j];

    j=threadIdx.x+blockDim.x*blockIdx.x;
    i=threadIdx.y+blockDim.y*blockIdx.y;
    __syncthreads();
    if(i<FC1_SIZE&&j==0)
    {
        float error=0;
        for(j=0;j<FC2_SIZE;j++)
            error+=_fc2_delta[idx%N_STREAM][j]*fc2_w[j][i];
        _fc1_delta[idx%N_STREAM][i]=error*(_fc1_a[idx%N_STREAM][i]*(1.0-_fc1_a[idx%N_STREAM][i]));
        _fc1_db[idx%N_STREAM][i]+=_fc1_delta[idx%N_STREAM][i];
    }

    j=threadIdx.x+blockDim.x*blockIdx.x;
    i=threadIdx.y+blockDim.y*blockIdx.y;
    __syncthreads();
    if(i<FC1_SIZE&&j<CONV_W_NUM)
    {
        for(k=0;k<POOL_SIZE;k++)
        for(l=0;l<POOL_SIZE;l++)
            _fc1_dw[idx%N_STREAM][i][j][k][l]+=_fc1_delta[idx%N_STREAM][i]*_pool[idx%N_STREAM][j][k][l];
    }
    __syncthreads();
}

__global__ void bp_update_conv(int idx)
{
    int i,j,k,l,m,n;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<POOL_SIZE&&k<POOL_SIZE)
    {
        float error=0;
        _conv_delta[idx%N_STREAM][i][j][k]=0;
        for(l=0;l<FC1_SIZE;l++)
            error+=_fc1_delta[idx%N_STREAM][l]*fc1_w[l][i][j][k];
        _conv_delta[idx%N_STREAM][i][j][k]=error*(_pool[idx%N_STREAM][i][j][k]*(1.0-_pool[idx%N_STREAM][i][j][k]));
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    k=threadIdx.z+blockDim.z*blockIdx.z;
    __syncthreads();
    if(i<CONV_W_NUM&&j==0&&k==0)
    {
        _conv_sigma_delta[idx%N_STREAM][i]=0;
        for(j=0;j<POOL_SIZE;j++)
        for(k=0;k<POOL_SIZE;k++)
            _conv_sigma_delta[idx%N_STREAM][i]+=_conv_delta[idx%N_STREAM][i][j][k];
        _conv_db[idx%N_STREAM][i]+=_conv_sigma_delta[idx%N_STREAM][i];
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    k=threadIdx.z+blockDim.z*blockIdx.z;
    __syncthreads();
    if(i<CONV_W_NUM&&j<CONV_W_SIZE&&k<CONV_W_SIZE)
    {
        float error=0;
        for(m=0;m<POOL_SIZE;m++)
        for(n=0;n<POOL_SIZE;n++)
        {
            int x=_pool_pos[idx%N_STREAM][i][m][n]/2;
            int y=_pool_pos[idx%N_STREAM][i][m][n]%2;
            error+=_conv_delta[idx%N_STREAM][i][m][n]*_input[idx%N_STREAM][2*m+j+x][2*n+k+y];
        }
        _conv_dw[idx%N_STREAM][i][j][k]+=error;
    }
    __syncthreads();
}

void bp_update_gpu(int idx)
{
    dim3 block1(16,64);
    dim3 grid1(1,1);
    dim3 block2(6,12,12);
    dim3 grid2(1,1,1);

    bp_update_fc<<<grid1,block1,0,stream[idx%N_STREAM]>>>(idx);
    bp_update_conv<<<grid2,block2,0,stream[idx%N_STREAM]>>>(idx);
}

__global__ void bp_assign_grads_fc(int idx)
{
    int i,j,k,l,m,n;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j==0)
    {
        for(j=0;j<_minibatch;j++)
            fc2_b[i]-=(_fc2_db[(idx-j)%N_STREAM][i]/_minibatch);
        for(j=0;j<_minibatch;j++)
            _fc2_db[(idx-j)%N_STREAM][i]=0;
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC2_SIZE&&j<FC1_SIZE)
    {
        for(k=0;k<_minibatch;k++)
            fc2_w[i][j]-=(_fc2_dw[(idx-k)%N_STREAM][i][j]/_minibatch);
        for(k=0;k<_minibatch;k++)
            _fc2_dw[(idx-k)%N_STREAM][i][j]=0;
    }

    j=threadIdx.x+blockDim.x*blockIdx.x;
    i=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC1_SIZE&&j==0)
    {
        for(j=0;j<_minibatch;j++)
            fc1_b[i]-=(_fc1_db[(idx-j)%N_STREAM][i]/_minibatch);
        for(j=0;j<_minibatch;j++)
            _fc1_db[(idx-j)%N_STREAM][i]=0;
    }

    j=threadIdx.x+blockDim.x*blockIdx.x;
    i=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<FC1_SIZE&&j<CONV_W_NUM)
    {
        for(k=0;k<POOL_SIZE;k++)
        for(l=0;l<POOL_SIZE;l++)
        {
            for(int m=0;m<_minibatch;m++)
                fc1_w[i][j][k][l]-=(_fc1_dw[(idx-m)%N_STREAM][i][j][k][l]/_minibatch);
            for(int m=0;m<_minibatch;m++)
                _fc1_dw[(idx-m)%N_STREAM][i][j][k][l]=0;
        }
    }
}

__global__ void bp_assign_grads_conv(int idx)
{
    int i,j,k,l,m,n;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j==0&&k==0)
    {
        for(j=0;j<_minibatch;j++)
            conv_b[i]-=(_conv_db[(idx-j)%N_STREAM][i]/_minibatch);
        for(j=0;j<_minibatch;j++)
            _conv_db[(idx-j)%N_STREAM][i]=0;
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    k=threadIdx.z+blockDim.z*blockIdx.z;
    if(i<CONV_W_NUM&&j<CONV_W_SIZE&&k<CONV_W_SIZE)
    {
        for(l=0;l<_minibatch;l++)
            conv_w[i][j][k]-=(_conv_dw[(idx-l)%N_STREAM][i][j][k]/_minibatch);
        for(l=0;l<_minibatch;l++)
            _conv_dw[(idx-l)%N_STREAM][i][j][k]=0;
    }
}

void bp_assign_grads_gpu(int idx)
{
    dim3 block1(16,64);
    dim3 grid1(1,1);
    dim3 block2(6,12,12);
    dim3 grid2(1,1,1);
    bp_assign_grads_fc<<<grid1,block1,0,stream[idx%N_STREAM]>>>(idx);
    bp_assign_grads_conv<<<grid2,block2,0,stream[idx%N_STREAM]>>>(idx);
}