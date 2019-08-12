NVCC = nvcc
SRC = $(wildcard *.cu)
CFLAGS =-rdc=true -O3 -w

mnist: $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS)

clean:
	rm mnist