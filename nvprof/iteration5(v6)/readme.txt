多个核函数串行调用并且有数据相关时，__syncthreads()加在结尾
<<<grid,block>>>，not <<<block,grid>>>

