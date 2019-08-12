NVCC = nvcc
SRC = $(wildcard *.cu)
CFLAGS = -O3 -w

mnist: $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS)

clean:
	rm mnist
