#include "bp.cuh"

void update_fc2_b()
{
    for(int i=0;i<FC2_SIZE;i++)
    {
        fc2_delta[i]=alpha*C[i]*(fc2_a[i]*(1.0-fc2_a[i]));
        fc2_db[i]+=fc2_delta[i];
    }
}

void update_fc2_w()
{
    for(int i=0;i<FC2_SIZE;i++)
    for(int j=0;j<FC1_SIZE;j++)
        fc2_dw[i][j]+=fc2_delta[i]*fc1_a[j];
}

void update_fc1_b()
{
    for(int i=0;i<FC1_SIZE;i++)
    {
        float error=0;
        for(int j=0;j<FC2_SIZE;j++)
            error+=fc2_delta[j]*fc2_w[j][i];
        fc1_delta[i]=error*(fc1_a[i]*(1.0-fc1_a[i]));
        fc1_db[i]+=fc1_delta[i];
    }
}

void update_fc1_w()
{
    for(int i=0;i<FC1_SIZE;i++)
    for(int j=0;j<CONV_W_NUM;j++)
    for(int k=0;k<POOL_SIZE;k++)
    for(int l=0;l<POOL_SIZE;l++)
        fc1_dw[i][j][k][l]+=fc1_delta[i]*pool[j][k][l];
}

void update_conv_b()
{
    for(int i=0;i<CONV_W_NUM;i++)
    {
        conv_sigma_delta[i]=0;
        for(int j=0;j<POOL_SIZE;j++)
        for(int k=0;k<POOL_SIZE;k++)
        {
            float error=0;
            conv_delta[i][j][k]=0;
            for(int l=0;l<FC1_SIZE;l++)
                error+=fc1_delta[l]*fc1_w[l][i][j][k];
            conv_delta[i][j][k]=error*(pool[i][j][k]*(1.0-pool[i][j][k]));
            conv_sigma_delta[i]+=error*(pool[i][j][k]*(1.0-pool[i][j][k]));
        }
        conv_db[i]+=conv_sigma_delta[i];
    }
}

void update_conv_w()
{
    for(int i=0;i<CONV_W_NUM;i++)
    for(int j=0;j<CONV_W_SIZE;j++)
    for(int k=0;k<CONV_W_SIZE;k++)
    {
        float error=0;
        for(int m=0;m<POOL_SIZE;m++)
        for(int n=0;n<POOL_SIZE;n++)
        {
            int x=pool_pos[i][m][n]/2;
            int y=pool_pos[i][m][n]%2;
            error+=conv_delta[i][m][n]*input[2*m+j+x][2*n+k+y];
        }
        conv_dw[i][j][k]+=error;
    }
}

void assign_grads()
{
    for(int i=0;i<FC2_SIZE;i++)
    {
        fc2_b[i]-=(fc2_db[i]/minibatch);
        fc2_db[i]=0;
    }

    for(int i=0;i<FC2_SIZE;i++)
    for(int j=0;j<FC1_SIZE;j++)
    {
        fc2_w[i][j]-=(fc2_dw[i][j]/minibatch);
        fc2_dw[i][j]=0;
    }

    for(int i=0;i<FC1_SIZE;i++)
    {
        fc1_b[i]-=(fc1_db[i]/minibatch);
        fc1_db[i]=0;
    }

    for(int i=0;i<FC1_SIZE;i++)
    for(int j=0;j<CONV_W_NUM;j++)
    for(int k=0;k<POOL_SIZE;k++)
    for(int l=0;l<POOL_SIZE;l++)
    {
        fc1_w[i][j][k][l]-=(fc1_dw[i][j][k][l]/minibatch);
        fc1_dw[i][j][k][l]=0;
    }

    for(int i=0;i<CONV_W_NUM;i++)
    {
        conv_b[i]-=(conv_db[i]/minibatch);
        conv_db[i]=0;
    }

    for(int i=0;i<CONV_W_NUM;i++)
    for(int j=0;j<CONV_W_SIZE;j++)
    for(int k=0;k<CONV_W_SIZE;k++)
    {
        conv_w[i][j][k]-=(conv_dw[i][j][k]/minibatch);
        conv_dw[i][j][k]=0;
    }
}