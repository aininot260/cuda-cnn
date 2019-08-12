#include<iostream>
#include "common.h"
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int cnt;
int img[28][28];
static Logger gLogger;

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<nvinfer1::DataType>(type);

        // Load blob
        if (wt.type == nvinfer1::DataType::kFLOAT)
        {
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == nvinfer1::DataType::kHALF)
        {
            uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/"};
    return locateFile(input, dirs);
}

ICudaEngine* createMNISTEngine(unsigned int maxBatchSize, IBuilder* builder, nvinfer1::DataType dt)
{
	INetworkDefinition* network = builder->createNetwork();
	ITensor* data = network->addInput("data", dt, Dims3{ 1, 28, 28 });
	map<string, Weights> weightMap = loadWeights(locateFile("params.txt"));
	IConvolutionLayer* conv1 = network->addConvolution(*data, 6, DimsHW{ 5, 5 }, weightMap["conv1filter"], weightMap["conv1bias"]);
	conv1->setStride(DimsHW{ 1, 1 });
	IActivationLayer* sigmoid1 = network->addActivation(*conv1->getOutput(0), ActivationType::kSIGMOID);
	IPoolingLayer* pool1 = network->addPooling(*sigmoid1->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	pool1->setStride(DimsHW{ 2, 2 });
	IFullyConnectedLayer* ip1 = network->addFullyConnected(*pool1->getOutput(0), 45, weightMap["ip1filter"], weightMap["ip1bias"]);
	IActivationLayer* sigmoid2 = network->addActivation(*ip1->getOutput(0), ActivationType::kSIGMOID);
	IFullyConnectedLayer* ip2 = network->addFullyConnected(*sigmoid2->getOutput(0), 10, weightMap["ip2filter"], weightMap["ip2bias"]);
	IActivationLayer* sigmoid3 = network->addActivation(*ip2->getOutput(0), ActivationType::kSIGMOID);
	sigmoid3->getOutput(0)->setName("prob");
	network->markOutput(*sigmoid3->getOutput(0));
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	samplesCommon::enableDLA(builder, -1);
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	network->destroy();
	for (auto& mem : weightMap)
		free((void*)(mem.second.values));
	return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
	IBuilder* builder = createInferBuilder(gLogger);
	ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder, nvinfer1::DataType::kFLOAT);
	(*modelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	void* buffers[2];
	const int inputIndex = engine.getBindingIndex("data");
	const int outputIndex = engine.getBindingIndex("prob");
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 28 * 28 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 10 * sizeof(float)));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 28 * 28 * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

void run_model()
{
	IHostMemory* modelStream{ nullptr };
	APIToModel(1, &modelStream);
	float data[28 * 28];
	for (int i = 0; i < 28 * 28; i++)
	{
		if (img[i / 28][i % 28] != 0)
			img[i / 28][i % 28] = 255;
		data[i] = float(img[i / 28][i % 28]);
	}
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			if (data[28 * i + j] > 0)
				cout << " ";
			else
				cout << "#";
		}
		cout << endl;
	}
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
	modelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
	float prob[10];
	doInference(*context, data, prob, 1);
	context->destroy();
	engine->destroy();
	runtime->destroy();
	for (unsigned int i = 0; i < 10; i++)
		cout << i << ": " << string(int(floor(prob[i] * 28 + 0.5f)), '*') << "\n";
	cout << endl;
}

int main()
{
	VideoCapture cap;
	cap.open("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=800, height=450, framerate=60/1, format=NV12 ! nvvidconv ! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true", CAP_GSTREAMER); 
	Mat frame;
	while (1)
	{
		char key = waitKey(100);
		cap >> frame;
		resize(frame, frame, Size(256, 256), (0, 0), (0, 0), INTER_LINEAR);
		imshow("frame", frame);
		if (key == 'p')
		{
			cout<<"Testing"<<endl;
			cnt++;
			imshow("photo", frame);
			cvtColor(frame, frame, CV_BGR2GRAY);
			resize(frame, frame, Size(28, 28), (0, 0), (0, 0), INTER_LINEAR);
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					img[i][j] = frame.at<uchar>(i, j);
					if (255 - img[i][j] > 150)
						img[i][j] = 255 - img[i][j];
					else
						img[i][j] = 0;
				}
			}
			waitKey(500);
			destroyWindow("photo");
			run_model();
		}
		if (key == 'q')
			break;
	}
	return 0;
}
