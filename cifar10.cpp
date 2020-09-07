#include <stdio.h>
#include <assert.h>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <chrono>
#include "ffCudaNn.h"

class ProfileScope
{
public:
	ProfileScope(const char* msg) : _msg(msg)
	{
		_s = std::chrono::high_resolution_clock::now();
	}
	~ProfileScope()
	{
		std::chrono::duration<float> delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - _s);
		printf("%s [%fs]\n", _msg, delta.count());
	}
	const char* _msg;
	std::chrono::high_resolution_clock::time_point _s;
};

void CheckAccuracy(const ff::CudaTensor* pSoftmax, const ff::CudaTensor& yLabel, int& top1, int& top3, int& top5);

void LoadCifar10(int batchSize, int maxImages, const std::vector<std::string>& filenames, std::vector<ff::CudaTensor>& images, std::vector<ff::CudaTensor>& labels)
{
	const int kFileBinarySize = 30730000;
	const int kNumImagePerFile = 10000;
	const int kNumBytesPerChannel = 1024; // 32 * 32
	const int kNumChannel = 3;
	int numFiles = (int)filenames.size();
	int numTotalImages = numFiles * kNumImagePerFile;
	if (numTotalImages > maxImages) numTotalImages = maxImages;
	int numBatches = (numTotalImages + batchSize - 1) / batchSize;
	images.resize(numBatches);
	labels.resize(numBatches);
	int nLeft = numTotalImages;
	for (int i = 0; i < numBatches; ++i)
	{
		int currBatchSize = (batchSize < nLeft ? batchSize : nLeft);
		images[i].ResetTensor(32, 32, 3, currBatchSize);
		labels[i].ResetTensor(currBatchSize);
		nLeft -= batchSize;
	}

	// Data normalization
	float mean[3] = { 0.4914f, 0.4822f, 0.4465f };
	float std[3] = { 0.2023f, 0.1994f, 0.2010f };

	int imageCounter = 0;
	std::vector<unsigned char> raw(kFileBinarySize);
	for (int i = 0; i < numFiles; ++i)
	{
		unsigned char* pCurr = &raw[0];
		FILE* fp = fopen(filenames[i].c_str(), "rb");
		assert(nullptr != fp);
		fread(pCurr, kFileBinarySize, 1, fp);
		fclose(fp);
		for (int j = 0; j < kNumImagePerFile; ++j)
		{
			int batchIndex = imageCounter / batchSize;
			int elementIndex = imageCounter % batchSize;
			labels[batchIndex]._data[elementIndex] = static_cast<float>(*pCurr++);
			for (int c = 0; c < kNumChannel; ++c)
			{
				for (int k = 0; k < kNumBytesPerChannel; ++k)
				{
					float val = *pCurr++;
					images[batchIndex]._data[elementIndex * kNumBytesPerChannel * kNumChannel + c * kNumBytesPerChannel + k] =
						((val/255.0f) - mean[c]) / std[c];
				}
			}
			++imageCounter;
			if (imageCounter >= numTotalImages)
				break;
		}
		if (imageCounter >= numTotalImages)
			break;
	}

	for (size_t i = 0; i < images.size(); ++i)
	{
		images[i].PushToGpu();
		labels[i].PushToGpu();
	}
}

int ComputeLoss(ff::CudaNn& nn, std::vector<ff::CudaTensor>& images, std::vector<ff::CudaTensor>& labels,
	int startIndex, int endIndex, float& loss, int& top1, int& top3, int& top5)
{
	loss = 0.0f;
	int imageCounter = 0;
	ff::CudaTensor* pSoftmax = nullptr;
	for (int i = startIndex; i < endIndex; ++i)
	{
		pSoftmax = const_cast<ff::CudaTensor*>(nn.Forward(&images[i]));
		pSoftmax->PullFromGpu();
		assert(labels[i]._d0 == pSoftmax->_d1);
		for (int j = 0; j < pSoftmax->_d1; ++j)
		{
			float val = pSoftmax->_data[static_cast<int>(labels[i]._data[j]) + pSoftmax->_d0 * j];
			assert(val > 0.0f);
			if (val > 0.0f)
			{
				++imageCounter;
				loss += -logf(val);
			}
		}
		int t1, t3, t5;
		CheckAccuracy(pSoftmax, labels[i], t1, t3, t5);
		top1 += t1; top3 += t3; top5 += t5;
	}
	loss /= imageCounter;
	return imageCounter;
}

int cifar10()
{
	const int kBatchSize = 50;

	std::vector<std::string> trainingDataFilenames;
	trainingDataFilenames.push_back("cifar-10/data_batch_1.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_2.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_3.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_4.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_5.bin");
	std::vector<ff::CudaTensor> trainingImages;
	std::vector<ff::CudaTensor> trainingLabels;
	LoadCifar10(kBatchSize, 50000, trainingDataFilenames, trainingImages, trainingLabels);
	std::vector<std::string> testDataFilenames;
	testDataFilenames.push_back("cifar-10/test_batch.bin");
	std::vector<ff::CudaTensor> testImages;
	std::vector<ff::CudaTensor> testLabels;
	LoadCifar10(kBatchSize, 10000, testDataFilenames, testImages, testLabels);

	ff::CudaNn nn;
	nn.AddConv2d(3, 3, 32, 1, 1);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 32, 64, 1, 1);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 64, 64, 1, 1);
	nn.AddRelu();
	nn.AddConv2d(3, 64, 128, 1, 1);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 128, 256, 1, 1);
	nn.AddRelu();
	nn.AddConv2d(3, 256, 256, 1, 1);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddFc(4 * 256, 2048);
	nn.AddRelu();
	nn.AddFc(2048, 10);
	nn.AddSoftmax();

	float learningRate = 0.00001f;
	float last_validation_loss = 1e8f;
	float lowest_validation_loss = 1e8f;
	float last_test_loss = 1e8f;
	float lowest_test_loss = 1e8f;
	int stagnancy = 0;

	const size_t numBatch = trainingImages.size();
	int currValidationDataIndex = 0;
	int numValidationData = static_cast<int>(numBatch) / 5;

	char buffer[2048];
	const int numEpoch = 200;
	for (int i = 0; i < numEpoch; ++i)
	{
		float currLearningRate = learningRate;

		// gradual decay
		const float kDecay = 0.1f;
		const int kCooldown = 10;
		//if (i >= kCooldown)
		//{
		//	currLearningRate *= expf(-1.0f * kDecay * (i - kCooldown));
		//}

		sprintf(buffer, "-- Epoch %03d(lr: %f)", i + 1, currLearningRate);
		ProfileScope __m(buffer);

		// Training
		for (size_t j = 0; j < numBatch; ++j)
		{
			if (currValidationDataIndex <= j && j < currValidationDataIndex + numValidationData)
			{
				continue; // Exclude validation data from training set
			}

			nn.Forward(&trainingImages[j], true);
			nn.Backward(&trainingLabels[j]);
			nn.UpdateWs(currLearningRate);
		}

		// Validation loss
		{
			int top1 = 0, top3 = 0, top5 = 0;
			float loss = 0.0f;
			int testCounter = ComputeLoss(nn, trainingImages, trainingLabels, currValidationDataIndex, currValidationDataIndex + numValidationData,
				loss, top1, top3, top5);
			if (0 == i) last_validation_loss = loss;
			++stagnancy;
			if (loss < lowest_validation_loss)
			{
				lowest_validation_loss = loss;
				stagnancy = 0;
			}
			const int kPatient = 2;
			const float kDecayRate = 0.5f;
			if (stagnancy >= kPatient)
			{
				// Learning rate decay
				//learningRate *= kDecayRate;
			}
			printf("Val_[%05d](Loss: %f(%+f)/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n",
				testCounter,
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
			int top1 = 0, top3 = 0, top5 = 0;
			float loss = 0.0f;
			int testCounter = ComputeLoss(nn, testImages, testLabels, 0, (int)testImages.size(), loss, top1, top3, top5);
			if (0 == i) last_test_loss = loss;
			if (loss < lowest_test_loss)
			{
				lowest_test_loss = loss;
			}
			printf("Test[%05d](Loss: %f(%+f)/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n",
				testCounter,
				loss, loss - last_test_loss, lowest_test_loss,
				top1,
				top3,
				top5);
			last_test_loss = loss;
		}
	}
	return 0;
}