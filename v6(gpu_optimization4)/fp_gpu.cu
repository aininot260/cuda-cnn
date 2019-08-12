#include "fp_gpu.cuh"

__global__ void fp_conv_pool(int idx,bool flag)
{
    int i,j,k,l,m;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    if(i<ROW&&j<COL)
    {
        if(flag)
            _input[idx%N_STREAM][i][j]=train_image[idx][i][j];
        else
            _input[idx%N_STREAM][i][j]=test_image[idx][i][j];
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    __syncthreads();
    if(i<CONV_W_NUM&&j<CONV_SIZE)
    {
        for(k=0;k<CONV_SIZE;k++)
        {
            _conv_z[idx%N_STREAM][i][j][k]=0;
            for(l=0;l<CONV_W_SIZE;l++)
            for(m=0;m<CONV_W_SIZE;m++)
                _conv_z[idx%N_STREAM][i][j][k]+=_input[idx%N_STREAM][j+l][k+m]*conv_w[i][l][m];
            _conv_z[idx%N_STREAM][i][j][k]+=conv_b[i];
            _conv_a[idx%N_STREAM][i][j][k]=sigmoid(_conv_z[idx%N_STREAM][i][j][k]);
        }
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    j=threadIdx.y+blockDim.y*blockIdx.y;
    __syncthreads();
    if(i<CONV_W_NUM&&j<POOL_SIZE)
    {
        for(k=0;k<POOL_SIZE;k++)
        {
            float _max=_conv_a[idx%N_STREAM][i][j*2][k*2];
            _pool_pos[idx%N_STREAM][i][j][k]=0;
            if(_conv_a[idx%N_STREAM][i][j*2][k*2+1]>_max)
            {
                _max=_conv_a[idx%N_STREAM][i][j*2][k*2+1];
                _pool_pos[idx%N_STREAM][i][j][k]=1;
            }
            if(_conv_a[idx%N_STREAM][i][j*2+1][k*2]>_max)
            {
                _max=_conv_a[idx%N_STREAM][i][j*2+1][k*2];
                _pool_pos[idx%N_STREAM][i][j][k]=2;
            }
            if(_conv_a[idx%N_STREAM][i][j*2+1][k*2+1]>_max)
            {
                _max=_conv_a[idx%N_STREAM][i][j*2+1][k*2+1];
                _pool_pos[idx%N_STREAM][i][j][k]=3;
            }
            _pool[idx%N_STREAM][i][j][k]=_max;
        }
    }
    __syncthreads();
}

void fp_conv_pool_gpu(int idx,bool flag)
{
    dim3 block(32,32);
    dim3 grid(1,1);
    fp_conv_pool<<<grid,block,0,stream[idx%N_STREAM]>>>(idx,flag);
}

__global__ void fp_fc_answer(int idx,bool flag)
{
    int i,j,k,l;
    i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<FC1_SIZE)
    {
        _fc1_z[idx%N_STREAM][i]=0;
        for(j=0;j<CONV_W_NUM;j++)
        for(k=0;k<POOL_SIZE;k++)
        for(l=0;l<POOL_SIZE;l++)
            _fc1_z[idx%N_STREAM][i]+=_pool[idx%N_STREAM][j][k][l]*fc1_w[i][j][k][l];
        _fc1_z[idx%N_STREAM][i]+=fc1_b[i];
        _fc1_a[idx%N_STREAM][i]=sigmoid(_fc1_z[idx%N_STREAM][i]);
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    __syncthreads();
    if(i<FC2_SIZE)
    {
        _fc2_z[idx%N_STREAM][i]=0;
        for(j=0;j<FC1_SIZE;j++)
            _fc2_z[idx%N_STREAM][i]+=_fc1_a[idx%N_STREAM][j]*fc2_w[i][j];
        _fc2_z[idx%N_STREAM][i]+=fc2_b[i];
        _fc2_a[idx%N_STREAM][i]=sigmoid(_fc2_z[idx%N_STREAM][i]);
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    __syncthreads();
    if(i<FC2_SIZE)
    {
        _output[idx%N_STREAM][i]=_fc2_a[idx%N_STREAM][i];
        if(flag)
            _answer[idx%N_STREAM][i]=(train_label[idx]==i)?1:0;
        else
            _answer[idx%N_STREAM][i]=(test_label[idx]==i)?1:0;
    }

    i=threadIdx.x+blockDim.x*blockIdx.x;
    __syncthreads();
    if(i==0)
    {
        float _max=_output[idx%N_STREAM][0];
        int max_pos=0;
        for(i=0;i<FC2_SIZE;i++)
        {
            if(_max<_output[idx%N_STREAM][i])
            {
                _max=_output[idx%N_STREAM][i];
                max_pos=i;
            }
        }
        if(_answer[idx%N_STREAM][max_pos])
            atomicAdd(&correct_cnt,1);
        for(i=0;i<FC2_SIZE;i++)
        {
            _C[idx%N_STREAM][i]=_output[idx%N_STREAM][i]-_answer[idx%N_STREAM][i];
            atomicExch(&avg_error,avg_error+_C[idx%N_STREAM][i]*_C[idx%N_STREAM][i]*0.5);
        }
    }
}

void fp_fc_answer_gpu(int idx,bool flag)
{
    dim3 block(64);
    dim3 grid(1);
    fp_fc_answer<<<grid,block,0,stream[idx%N_STREAM]>>>(idx,flag);
}
