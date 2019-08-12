NVCC = nvcc
SRC = $(wildcard *.cu)
CFLAGS =-arch=sm_35 -Xptxas -dlcm=cg -rdc=true -O3 -w -Xcompiler "/MTd /openmp" 

mnist: $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS)

clean:
	rm mnist
