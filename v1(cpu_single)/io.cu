#include "io.cuh"

void load_data()
{
    FILE *f_images=fopen("../data/train-images.idx3-ubyte","rb");
    FILE *f_labels=fopen("../data/train-labels.idx1-ubyte","rb");

    int tmp;

    int magic_num;
    fread(&magic_num,sizeof(int),1,f_images);
    fread(&magic_num,sizeof(int),1,f_labels);

    // printf("debug:%d\n",swap_endian(magic_num));

    int train_size;
    fread(&train_size,sizeof(int),1,f_images);
    fread(&train_size,sizeof(int),1,f_labels);
    train_size=swap_endian(train_size);

    // printf("debug:%d\n",swap_endian(train_size));

    int rows,cols;
    fread(&rows,sizeof(int),1,f_images);
    fread(&cols,sizeof(int),1,f_images);
    rows=swap_endian(rows);
    cols=swap_endian(cols);

    // printf("debug:%d\n",swap_endian(rows));
    // printf("debug:%d\n",swap_endian(cols));

    for(int i=0;i<train_size;i++)
    {
        fread(&train_label[i],1,1,f_labels);
        if(i%100==0)
            printf("Training labels : Already read %5d labels\r",i);
        // printf("%d:debug:%d\r",i,train_label[i]);
        // system("pause");
    }
    printf("Training labels : Already read %5d labels\n",train_size);

    for(int i=0;i<train_size;i++)
    {
        for(int j=0;j<rows;j++)
        for(int k=0;k<cols;k++)
        {
            tmp=0;
            fread(&tmp,1,1,f_images);
            train_image[i][j][k]=tmp;
            train_image[i][j][k]/=255;
            // printf("%d %d %d debug: %f\n",i,j,k,train_image[i][j][k]);
            // system("pause");
        }
        if(i%100==0)
            printf("Training images : Already read %5d images\r",i);
    }
    printf("Training images : Already read %5d images\n",train_size);

    fclose(f_images);
    fclose(f_labels);

    f_images=fopen("../data/t10k-images.idx3-ubyte","rb");
    f_labels=fopen("../data/t10k-labels.idx1-ubyte","rb");

    fread(&magic_num,sizeof(int),1,f_images);
    fread(&magic_num,sizeof(int),1,f_labels);

    int test_size;
    fread(&test_size,sizeof(int),1,f_images);
    fread(&test_size,sizeof(int),1,f_labels);
    test_size=swap_endian(test_size);

    fread(&rows,sizeof(int),1,f_images);
    fread(&cols,sizeof(int),1,f_images);
    rows=swap_endian(rows);
    cols=swap_endian(cols);

    for(int i=0;i<test_size;i++)
    {
        fread(&test_label[i],1,1,f_labels);
        if(i%100==0)
            printf("Testing labels : Already read %5d labels\r",i);
    }
    printf("Testing labels : Already read %5d labels\n",test_size);

    for(int i=0;i<test_size;i++)
    {
        for(int j=0;j<rows;j++)
        for(int k=0;k<cols;k++)
        {
            tmp=0;
            fread(&tmp,1,1,f_images);
            test_image[i][j][k]=tmp;
            test_image[i][j][k]/=255;
        }
        if(i%100==0)
            printf("Testing images : Already read %5d images\r",i);
    }
    printf("Testing images : Already read %5d images\n\n",test_size);

    fclose(f_images);
    fclose(f_labels);
}

void export_params()
{
    FILE *f_params=fopen("./params.txt","w");

    fprintf(f_params,"6\n");

    fprintf(f_params,"conv1bias 0 6 ");
    for(int i=0;i<CONV_W_NUM;i++)
        fprintf(f_params,"%X ", *(int *)&conv_b[i]);
    fprintf(f_params,"\n");

    fprintf(f_params,"conv1filter 0 150 ");
    for(int i=0;i<CONV_W_NUM;i++)
    for(int j=0;j<CONV_W_SIZE;j++)
    for(int k=0;k<CONV_W_SIZE;k++)
        fprintf(f_params,"%X ", *(int *)&conv_w[i][j][k]);
    fprintf(f_params,"\n");

    fprintf(f_params,"ip1bias 0 45 ");
    for(int i=0;i<FC1_SIZE;i++)
        fprintf(f_params,"%X ", *(int *)&fc1_b[i]);
    fprintf(f_params,"\n");

    fprintf(f_params,"ip1filter 0 38880 ");
    for(int i=0;i<FC1_SIZE;i++)
    for(int j=0;j<CONV_W_NUM;j++)
    for(int k=0;k<POOL_SIZE;k++)
    for(int l=0;l<POOL_SIZE;l++)
        fprintf(f_params,"%X ", *(int *)&fc1_w[i][j][k][l]);
    fprintf(f_params,"\n");

    fprintf(f_params,"ip2bias 0 10 ");
    for(int i=0;i<FC2_SIZE;i++)
        fprintf(f_params,"%X ", *(int *)&fc2_b[i]);
    fprintf(f_params,"\n");

    fprintf(f_params,"ip2filter 0 450 ");
    for(int i=0;i<FC2_SIZE;i++)
    for(int j=0;j<FC1_SIZE;j++)
        fprintf(f_params,"%X ", *(int *)&fc2_w[i][j]);
    
    fclose(f_params);

}