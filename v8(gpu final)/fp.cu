#include "fp.cuh"

void set_input(int idx,float image[TRAIN_NUM][ROW][COL])
{
    for(int i=0;i<ROW;i++)
    for(int j=0;j<COL;j++)
        input[i][j]=image[idx][i][j];
}

void input_conv()
{
    for(int i=0;i<CONV_W_NUM;i++)
    for(int j=0;j<CONV_SIZE;j++)
    for(int k=0;k<CONV_SIZE;k++)
    {
        conv_z[i][j][k]=0;
        for(int l=0;l<CONV_W_SIZE;l++)
        for(int m=0;m<CONV_W_SIZE;m++)
            conv_z[i][j][k]+=input[j+l][k+m]*conv_w[i][l][m];
        conv_z[i][j][k]+=conv_b[i];
        conv_a[i][j][k]=sigmoid(conv_z[i][j][k]);
    }
}

void conv_pool()
{
    for(int i=0;i<CONV_W_NUM;i++)
    for(int j=0;j<POOL_SIZE;j++)
    for(int k=0;k<POOL_SIZE;k++)
    {
        float _max=conv_a[i][j*2][k*2];
        pool_pos[i][j][k]=0;
        if(conv_a[i][j*2][k*2+1]>_max)
        {
            _max=conv_a[i][j*2][k*2+1];
            pool_pos[i][j][k]=1;
        }
        if(conv_a[i][j*2+1][k*2]>_max)
        {
            _max=conv_a[i][j*2+1][k*2];
            pool_pos[i][j][k]=2;
        }
        if(conv_a[i][j*2+1][k*2+1]>_max)
        {
            _max=conv_a[i][j*2+1][k*2+1];
            pool_pos[i][j][k]=3;
        }
        pool[i][j][k]=_max;
    }
}

void pool_fc1()
{
    for(int i=0;i<FC1_SIZE;i++)
    {
        fc1_z[i]=0;
        for(int j=0;j<CONV_W_NUM;j++)
        for(int k=0;k<POOL_SIZE;k++)
        for(int l=0;l<POOL_SIZE;l++)
            fc1_z[i]+=pool[j][k][l]*fc1_w[i][j][k][l];
        fc1_z[i]+=fc1_b[i];
        fc1_a[i]=sigmoid(fc1_z[i]);
    }
}

void fc1_fc2()
{
    for(int i=0;i<FC2_SIZE;i++)
    {
        fc2_z[i]=0;
        for(int j=0;j<FC1_SIZE;j++)
            fc2_z[i]+=fc1_a[j]*fc2_w[i][j];
        fc2_z[i]+=fc2_b[i];
        fc2_a[i]=sigmoid(fc2_z[i]);
    }
}

void set_answer(int idx,int label[TRAIN_NUM])
{
    for(int i=0;i<FC2_SIZE;i++)
    {
        output[i]=fc2_a[i];
        answer[i]=(label[idx]==i)?1:0;
    }
}

void check_answer(int &correct_cnt)
{
    float _max=output[0];
    int max_pos=0;
    for(int i=0;i<FC2_SIZE;i++)
    {
        if(_max<output[i])
        {
            _max=output[i];
            max_pos=i;
        }
    }
    if(answer[max_pos])
        correct_cnt++;
}

void get_error(float &avg_error)
{
    for(int i=0;i<FC2_SIZE;i++)
    {
        C[i]=output[i]-answer[i];
        avg_error+=C[i]*C[i]*0.5;
    }
}