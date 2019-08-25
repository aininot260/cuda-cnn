#include "stdio.h"

#include "io.cuh"
#include "fp.cuh"
#include "bp.cuh"
#include "init.cuh"
#include "utils.cuh"
#include "global.cuh"

#include "fp_gpu.cuh"
#include "bp_gpu.cuh"
#include "global_gpu.cuh"

float max_acc;
clock_t t;

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
    init_params();

    if(strcmp(argv[1],"CPU")==0)
    {
        for(int i=1;i<=epochs;i++)
        {
            t=clock();
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
            }
            printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TEST_NUM,((float)correct_cnt/TEST_NUM)*100,(avg_error/TEST_NUM)*100);
        
            if((float)correct_cnt/TEST_NUM*100>max_acc)
            {
                max_acc=(float)correct_cnt/TEST_NUM*100;
                export_params();
                printf("The new model has been exported. Accuracy has reached to %0.5f%%\n\n",max_acc);
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
        int n_stream=N_STREAM;

        stream=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
        for(int i=0;i<n_stream;i++)
            cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking);

        for(int i=1;i<=epochs;i++)
        {
            t=clock();
            correct_cnt=0;
            avg_error=0;
    
            for(int j=0;j<TRAIN_NUM;j++)
            {
                fp_conv_pool_gpu(j,1);
                fp_fc_answer_gpu(j,1);
    
                bp_update_gpu(j);
                if((j+1)%minibatch==0)
                    bp_assign_grads_gpu(j);
            }
    
            cudaDeviceSynchronize();
            printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TRAIN_NUM,((float)correct_cnt/TRAIN_NUM)*100,(avg_error/TRAIN_NUM)*100,i);
    
            correct_cnt=0;
            avg_error=0;
    
            for(int j=0;j<TEST_NUM;j++)
            {
                fp_conv_pool_gpu(j,0);
                fp_fc_answer_gpu(j,0);
            }
    
            cudaDeviceSynchronize();
            printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TEST_NUM,((float)correct_cnt/TEST_NUM)*100,(avg_error/TEST_NUM)*100);
        
            if((float)correct_cnt/TEST_NUM*100>max_acc)
            {
                max_acc=(float)correct_cnt/TEST_NUM*100;
                export_params();
                printf("The new model has been exported. Accuracy has reached to %0.5f%%\n\n",max_acc);
            }
            else
            {
                alpha=alpha-(alpha/10);
                printf("Learning rate has been reduced to %f\n\n",alpha);
            }
        }

        for(int i=0;i<n_stream;i++)
            cudaStreamDestroy(stream[i]);
        free(stream);
    }
    else
    {
        printf("The parameter can only be GPU or CPU!\n");
        return 0;
    }
    return 0;
}