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
clock_t t;

void work_cpu()
{
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

            if(j&&j%5000==0)
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

            if(j&&j%5000==0)
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

void work_gpu()
{
    for(int i=1;i<=epochs;i++)
    {
        int value1=0;
        float value2=0;
        CHECK(cudaMemcpyToSymbol(_correct_cnt,&value1,sizeof(int),0,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(_avg_error,&value2,sizeof(float),0,cudaMemcpyHostToDevice));

        for(int j=0;j<TRAIN_NUM;j++)
        {
            set_input_gpu_train(j);
            input_conv_gpu(j);
            conv_pool_gpu(j);
            pool_fc1_gpu(j);
            fc1_fc2_gpu(j);
            set_answer_gpu_train(j);
            check_answer_get_error_gpu(j);

            update_fc2_b_gpu(j);
            update_fc2_w_gpu(j);
            update_fc1_b_gpu(j);
            update_fc1_w_gpu(j);
            update_conv_b_gpu(j);
            update_conv_w_gpu(j);
            if((j+1)%minibatch==0)
                assign_grads_gpu(j);
            if(j&&j%5000==0)
            {
                CHECK(cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int),0,cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float),0,cudaMemcpyDeviceToHost));
                printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100,i);
            }
        }

        CHECK(cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int),0,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float),0,cudaMemcpyDeviceToHost));
        printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n",floor(((float)(clock()-t))/CLOCKS_PER_SEC),TRAIN_NUM,((float)correct_cnt/TRAIN_NUM)*100,(avg_error/TRAIN_NUM)*100,i);

        correct_cnt=0;
        avg_error=0;
        CHECK(cudaMemcpyToSymbol(_correct_cnt,&correct_cnt,sizeof(int),0,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(_avg_error,&avg_error,sizeof(float),0,cudaMemcpyHostToDevice));

        for(int j=0;j<TEST_NUM;j++)
        {
            set_input_gpu_test(j);
            input_conv_gpu(j);
            conv_pool_gpu(j);
            pool_fc1_gpu(j);
            fc1_fc2_gpu(j);
            set_answer_gpu_test(j);
            check_answer_get_error_gpu(j);

            if(j&&j%5000==0)
            {
                CHECK(cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int),0,cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float),0,cudaMemcpyDeviceToHost));
                printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r",floor(((float)(clock()-t))/CLOCKS_PER_SEC),j,((float)correct_cnt/j)*100,(avg_error/j)*100);
            }
        }
        CHECK(cudaMemcpyFromSymbol(&correct_cnt,_correct_cnt,sizeof(int),0,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpyFromSymbol(&avg_error,_avg_error,sizeof(float),0,cudaMemcpyDeviceToHost));
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
            CHECK(cudaMemcpyToSymbol(_alpha,&alpha,sizeof(float),0,cudaMemcpyHostToDevice));
            printf("Learning rate has been reduced to %f\n\n",alpha);
        }
    }
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
    

    if(strcmp(argv[1],"CPU")==0)
    {
        init_params();
        t=clock();
        work_cpu();
    }

    else if(strcmp(argv[1],"GPU")==0)
    {
        initDevice(0);

        CHECK(cudaMemcpyToSymbol(_alpha,&alpha,sizeof(float)));
        CHECK(cudaMemcpyToSymbol(_minibatch,&minibatch,sizeof(int)));
        CHECK(cudaMemcpyToSymbol(_epochs,&epochs,sizeof(int)));

        // setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
        /*
            five SMs,each SM has two 1024-blocks.
        */
        int n_stream=16;

        stream=(cudaStream_t*)malloc(n_stream*sizeof(cudaStream_t));
        for(int i=0;i<n_stream;i++)
            // cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking);
            cudaStreamCreate(&stream[i]);

        init_data_gpu();
        t=clock();
        init_params_gpu();

        work_gpu();

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