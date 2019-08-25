NVCC = nvcc
SRC = $(wildcard *.cu)
CFLAGS = -arch=compute_50 -code=sm_50,compute_50 -rdc=true -O3 -w

mnist: $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS)

clean:
	rm mnist