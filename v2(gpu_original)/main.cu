#include "stdio.h"
#include "global.cuh"
#include "io.cuh"
#include "init.cuh"
#include "fp.cuh"
#include "bp.cuh"

#include "global_gpu.cuh"
#include "utils_gpu.cuh"
#include "init_gpu.cuh"
#include "test_gpu.cuh"
#include "fp_gpu.cuh"
#include "bp_gpu.cuh"

int correct_cnt;
float avg_error;
float max_acc;

__global__ void _test()
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int iz=threadIdx.z+blockDim.z*blockIdx.z;

    for(int i=5000;i<5001;i++)
    for(int j=0;j<ROW;j++)
    {
        for(int k=0;k<COL;k++)
            printf("%f ",_test_image[i][j][k]);
        printf("\n");
    }
    printf("%d",_test_label[5000]);

    // printf("%f ",_test_image[ix][iy][iz]);
}

void test()
{
    puts("");
    puts("debug1");
    dim3 block(1,1,1);
    dim3 grid(1,1,1);
    _test<<<block,grid>>>();
    puts("debug2");
    cudaDeviceSynchronize();
    puts("debug3");
}

int main(int argc,char *argv[])
{
    printf("====== aininot260 gh@ysucloud.com ======\n");
    printf("         Processor used : %s\n",argv[1]);
    printf("         Learning rate  : %.2f\n",alpha);
    printf("         Epochs         : %d\n",epochs);
    printf("         Batch size     : %d\n",minibatch);
    printf("========================================\n");
    printf("\n");

    load_data();
    
    clock_t t=clock();

    if(strcmp(argv[1],"CPU")==0)
    {
        init_params();

        for(int i=1;i<=epochs;i++)
        {
            correct_cnt=0;
            avg_error=0;
    
            for(int j=0;j<TRAIN_NUM;j++)
            {
                set_input(j,train_image);
                input_conv();
                conv_pool();
                pool_fc1();
                fc1_fc2();
                set_answer(j,train_label);
                check_answer(correct_cnt);
                get_error(avg_error);
    
                update_fc2_b();
                update_fc2_w();
                update_fc1_b();
                update_fc1_w();
                update_conv_b();
                update_conv_w();
                if((j+1)%minibatch==0)
                    assign_grads();
    
                if(j&&j%100==0)
                    printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100,i);
            }
            printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TRAIN_NUM,((float)correct_cnt/TRAIN_NUM)*100,(avg_error/TRAIN_NUM)*100,i);
        
            correct_cnt=0;
            avg_error=0;
    
            for(int j=0;j<TEST_NUM;j++)
            {
                set_input(j,test_image);
                input_conv();
                conv_pool();
                pool_fc1();
                fc1_fc2();
                set_answer(j,test_label);
                check_answer(correct_cnt);
                get_error(avg_error);
    
                if(j&&j%100==0)
                    printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100);
            }
            printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TEST_NUM,((float)correct_cnt/TEST_NUM)*100,(avg_error/TEST_NUM)*100);
        
            if((float)correct_cnt/TEST_NUM*100>max_acc)
            {
                max_acc=(float)correct_cnt/TEST_NUM*100;
                export_params();
                printf("The new model has been exported.Accuracy has reached to %0.5f%%\n\n",max_acc);
            }
            else
            {
                alpha=alpha-(alpha/3);
                printf("Learning rate has been reduced to %f\n\n",alpha);
            }
        }
    }

    else if(strcmp(argv[1],"GPU")==0)
    {
        initDevice(0);
        CHECK(cudaMemcpyToSymbol(_alpha,&alpha,sizeof(float)));
        CHECK(cudaMemcpyToSymbol(_minibatch,&minibatch,sizeof(int)));
        CHECK(cudaMemcpyToSymbol(_epochs,&epochs,sizeof(int)));
        init_data_gpu();
        init_params_gpu();
        for(int i=1;i<=epochs;i++)
        {

            int value1=0;
            float value2=0;
            CHECK(cudaMemcpyToSymbol(_correct_cnt,&value1,sizeof(int)));
            CHECK(cudaMemcpyToSymbol(_avg_error,&value2,sizeof(float)));
            cudaDeviceSynchronize();

            for(int j=0;j<TRAIN_NUM;j++)
            {
                set_input_gpu_train(j);
                input_conv_gpu();
                conv_pool_gpu();
                pool_fc1_gpu();
                fc1_fc2_gpu();
                set_answer_gpu_train(j);
                check_answer_get_error_gpu();

                update_fc2_b_gpu();
                update_fc2_w_gpu();
                update_fc1_b_gpu();
                update_fc1_w_gpu();
                update_conv_b_gpu();
                update_conv_w_gpu();
                if((j+1)%minibatch==0)
                    assign_grads_gpu();

                if(j&&j%100==0)
                {
                    cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int));
                    cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float));
                    printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100,i);
                }
            }

            cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int));
            cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float));
            printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TRAIN_NUM,((float)correct_cnt/TRAIN_NUM)*100,(avg_error/TRAIN_NUM)*100,i);

            correct_cnt=0;
            avg_error=0;
            cudaMemcpyToSymbol(_correct_cnt,&correct_cnt,sizeof(int));
            cudaMemcpyToSymbol(_avg_error,&avg_error,sizeof(float));

            for(int j=0;j<TEST_NUM;j++)
            {
                set_input_gpu_test(j);
                input_conv_gpu();
                conv_pool_gpu();
                pool_fc1_gpu();
                fc1_fc2_gpu();
                set_answer_gpu_test(j);
                check_answer_get_error_gpu();
    
                if(j&&j%100==0)
                {
                    cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int));
                    cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float));
                    printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100);
                }
            }
            cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int));
            cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float));
            printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TEST_NUM,((float)correct_cnt/TEST_NUM)*100,(avg_error/TEST_NUM)*100);
        
            if((float)correct_cnt/TEST_NUM*100>max_acc)
            {
                max_acc=(float)correct_cnt/TEST_NUM*100;
                //export_params();
                printf("The new model has been exported.Accuracy has reached to %0.5f%%\n\n",max_acc);
            }
            else
            {
                alpha=alpha-(alpha/3);
                cudaMemcpyToSymbol(_alpha,&alpha,sizeof(float));
                printf("Learning rate has been reduced to %f\n\n",alpha);
            }
        }
    }
    else
    {
        printf("The parameter can only be GPU or CPU!\n");
        return 0;
    }
    return 0;
}