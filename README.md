# cuda_cnn

## Layer configuration
- Convolution

Input: 28 * 28 * 1
Output: 24 * 24 * 6
Kernel size: 5 * 5
Kernel amount: 6
Stride: 1

- Sigmoid

- Pooling

Input: 24 * 24 * 6
Output: 12 * 12 * 6
Window size: 2 * 2
Stride: 2

- FullyConnected1

Input: 12 * 12 * 6
Output: 45

- Sigmoid

- FullyConnected2

Input: 45
Output: 10

- Sigmoid

## Dataset
- MNIST: 60k train set, 10k test set
- 28 * 28 * 1

## Accuracy
- 20 epoch: 98.71%

## Experiment environment
#### Development environment: HP Pavilion Gaming Notebook
- CPU: Intel Core i7-6700HQ @ 2.60Ghz
- GPU: NVIDIA GeForce GTX 950M

#### Production environment: NVIDIA Jetson Nano
- CPU: Quad-core ARM A57 @ 1.43 GHz
- GPU: 128-core Maxwell

## Usage
- Generating model

Do the following in a development environment.
```
cd ./v8\(gpu_final\)
make
./mnist CPU
./mnist GPU
``` 
- Deployment model

The following operations are performed in the production environment.
```
sudo cp ./params.txt /usr/src/tensorrt/data/mnist
sudo mkdir /usr/src/tensorrt/samples/cuda_cnn
cd Jetson_Nano
sudo cp ./cuda_cnn.cpp /usr/src/tensorrt/samples/cuda_cnn
sudo cp ./Makefile /usr/src/tensorrt/samples/cuda_cnn
cd /usr/src/tensorrt/samples/cuda_cnn
sudo make
cd /usr/src/tensorrt/bin
./cuda_cnn
``` 
Focus on the frame, press the P key for prediction analysis, press the Q key to exit the program.



