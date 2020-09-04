#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
#include "ffCudaNn.h"

class MeasureTime
{
public:
	MeasureTime(const char* msg) : _msg(msg)
	{
		_s = std::chrono::high_resolution_clock::now();
	}
	~MeasureTime()
	{
		std::chrono::duration<float> delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - _s);
		printf("%s [%fs]\n", _msg, delta.count());
	}
	const char* _msg;
	std::chrono::high_resolution_clock::time_point _s;
};

void CheckAccuracy(const ff::CudaTensor* pSoftmax, const ff::CudaTensor& yLabel, int& top1, int& top3, int& top5);

void LoadCifar10(int trainingBatchSize, std::vector<ff::CudaTensor>& trainingImages, std::vector<ff::CudaTensor>& trainingLabels,
	ff::CudaTensor& testImages, ff::CudaTensor& testLabels)
{
	const int kTrainingDataSize = 50000;
	const int kTestDataSize = 10000;
	const int kNumBytesPerChannel = 1024; // 32 * 32
	int numBatches = (kTrainingDataSize + trainingBatchSize - 1) / trainingBatchSize;
	trainingImages.resize(numBatches);
	trainingLabels.resize(numBatches);
	int nLeft = kTrainingDataSize;
	for (int i = 0; i < numBatches; ++i)
	{
		int currBatchSize = (trainingBatchSize < nLeft ? trainingBatchSize : nLeft);
		trainingImages[i].ResetTensor(32, 32, 3, currBatchSize);
		trainingLabels[i].ResetTensor(currBatchSize);
		nLeft -= trainingBatchSize;
	}
	testImages.ResetTensor(32, 32, 3, kTestDataSize);
	testLabels.ResetTensor(kTestDataSize);

	int counter = 0;
	const char* filenames[5] = {
		"cifar-10/data_batch_1.bin",
		"cifar-10/data_batch_2.bin", 
		"cifar-10/data_batch_3.bin", 
		"cifar-10/data_batch_4.bin",
		"cifar-10/data_batch_5.bin" };

	// Data normalization
	float std[3] = { 63.0f, 62.1f, 66.7f };
	float mean[3] = { 125.3f, 123.0f, 113.9f };

	std::vector<unsigned char> raw(30730000);
	for (int i = 0; i < 5; ++i)
	{
		unsigned char* pCurr = &raw[0];
		FILE* fp = fopen(filenames[i], "rb");
		assert(nullptr != fp);
		fread(pCurr, 30730000, 1, fp);
		fclose(fp);
		for (int j = 0; j < 10000; ++j)
		{
			int batchIndex = counter / trainingBatchSize;
			int elementIndex = counter % trainingBatchSize;
			++counter;
			trainingLabels[batchIndex]._data[elementIndex] = static_cast<float>(*pCurr++);
			for (int c = 0; c < 3; ++c)
			{
				for (int k = 0; k < kNumBytesPerChannel; ++k)
				{
					float val = *pCurr++;
					trainingImages[batchIndex]._data[elementIndex * kNumBytesPerChannel * 3 + c * kNumBytesPerChannel + k] = (val - mean[c]) / std[c];
				}
			}
		}
	}
	for (size_t i = 0; i < trainingImages.size(); ++i)
	{
		trainingImages[i].PushToGpu();
		trainingLabels[i].PushToGpu();
	}

	FILE* fp = fopen("cifar-10/test_batch.bin", "rb");
	assert(nullptr != fp);
	unsigned char* pCurr = &raw[0];
	fread(pCurr, 30730000, 1, fp);
	fclose(fp);
	for (int i = 0; i < 10000; ++i)
	{
		testLabels._data[i] = static_cast<float>(*pCurr++);
		for (int c = 0; c < 3; ++c)
		{
			for (int j = 0; j < kNumBytesPerChannel; ++j)
			{
				float val = *pCurr++;
				testImages._data[i * kNumBytesPerChannel * 3 + c * kNumBytesPerChannel + j] = (val - mean[c]) / std[c];
			}
		}
	}
	testImages.PushToGpu();
	testLabels.PushToGpu();
}

int cifar10()
{
	const int kBatchSize = 50;

	std::vector<ff::CudaTensor> trainingImages;
	std::vector<ff::CudaTensor> trainingLabels;
	ff::CudaTensor testImages;
	ff::CudaTensor testLabels;
	LoadCifar10(kBatchSize, trainingImages, trainingLabels, testImages, testLabels);

	ff::CudaNn nn;
	nn.InitializeCudaNn("");
	nn.AddConv2d(3, 3, 4, 1, 1); // 32 * 32 * 4 
	nn.AddRelu();
	nn.AddConv2d(3, 4, 4, 1, 1); // 32 * 32 * 4 
	nn.AddRelu();
	nn.AddConv2d(3, 4, 4, 1, 1); // 32 * 32 * 4 
	nn.AddRelu();
	nn.AddConv2d(3, 4, 4, 1, 1); // 32 * 32 * 4 
	nn.AddRelu();
	nn.AddFc(4096, 2048);
	nn.AddRelu();
	nn.AddFc(2048, 1024);
	nn.AddRelu();
	nn.AddFc(1024, 10);
	nn.AddSoftmax();

	float loss = 0.0f;
	float last_validation_loss = 1e8f;
	float lowest_validation_loss = 1e8f;
	float last_test_loss = 1e8f;
	float lowest_test_loss = 1e8f;
	const size_t numBatch = trainingImages.size();
	int currValidationDataIndex = 0;
	int numValidationData = static_cast<int>(numBatch) / 5;
	int top1 = 0, top3 = 0, top5 = 0;
	ff::CudaTensor* pSoftmax = nullptr;

	char buffer[2048];
	const int numEpoch = 10000;
	float learningRate = 0.00001f;
	printf("* Initial learning rate(%f)\n", learningRate);
	for (int i = 0; i < numEpoch; ++i)
	{
		sprintf(buffer, "-- Epoch %03d", i + 1);
		MeasureTime __m(buffer);

		// Training
		{
			for (size_t j = 0; j < numBatch; ++j)
			{
				if (currValidationDataIndex <= j && j < currValidationDataIndex + numValidationData)
				{
					continue; // Exclude validation data from training set
				}

				nn.Forward(&trainingImages[j], true);
				nn.Backward(&trainingLabels[j]);
				nn.UpdateWs(learningRate);
			}
		}

		// Validation loss
		{
			loss = 0.0f;
			top1 = top3 = top5 = 0;
			int cntVal = 0;
			for (int j = currValidationDataIndex; j < currValidationDataIndex + numValidationData; ++j)
			{
				cntVal += trainingImages[j]._d1;
				pSoftmax = const_cast<ff::CudaTensor*>(nn.Forward(&trainingImages[j]));
				pSoftmax->PullFromGpu();
				for (int k = 0; k < trainingImages[j]._d1; ++k)
				{
					loss += -logf(pSoftmax->_data[static_cast<int>(trainingLabels[j]._data[k]) + pSoftmax->_d0 * k]);
				}
				int t1, t3, t5;
				CheckAccuracy(pSoftmax, trainingLabels[j], t1, t3, t5);
				top1 += t1; top3 += t3; top5 += t5;
			}
			loss /= cntVal;
			if (0 == i) last_validation_loss = loss;
			if (loss < lowest_validation_loss)
			{
				lowest_validation_loss = loss;
			}
			if (loss > last_validation_loss)
			{
				// Learning rate decay
				learningRate *= 0.5f;
				printf("- Learning rate decreased(%f)\n", learningRate);
			}
			learningRate *= 0.6f;
			printf("Val_[%05d](Loss: %f(%+f)/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n",
				cntVal,
				loss, loss - last_validation_loss, lowest_validation_loss,
				top1,
				top3,
				top5);
			last_validation_loss = loss;

			// Rotate validation set
			currValidationDataIndex += numValidationData;
			if (currValidationDataIndex + numValidationData > numBatch)
			{
				currValidationDataIndex = 0;
			}
		}

		// Test loss
		{
			pSoftmax = const_cast<ff::CudaTensor*>(nn.Forward(&testImages));
			pSoftmax->PullFromGpu();

			loss = 0.0f;
			for (int j = 0; j < testImages._d1; ++j)
			{
				loss += -logf(pSoftmax->_data[static_cast<int>(testLabels._data[j]) + pSoftmax->_d0 * j]);
			}
			loss /= testImages._d1;
			if (0 == i) last_test_loss = loss;
			if (loss < lowest_test_loss)
			{
				lowest_test_loss = loss;
			}
			CheckAccuracy(pSoftmax, testLabels, top1, top3, top5);
			printf("Test[%05d](Loss: %f(%+f)/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n",
				pSoftmax->_d1,
				loss, loss - last_test_loss, lowest_test_loss,
				top1,
				top3,
				top5);
			last_test_loss = loss;
		}
	}
	return 0;
}