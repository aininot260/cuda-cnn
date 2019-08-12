#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

int main()
{
    #pragma omp parallel for
    for(int i=0;i<16;i++)
        printf("%d\n",i);
    return 0;
}