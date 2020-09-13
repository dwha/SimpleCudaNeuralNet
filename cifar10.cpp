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

float Bilinear(int nRow, int nCol, const float* srcImage, float u, float v)
{
	assert(u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f);
	float c = u * nCol;
	int c0 = (int)c;
	float alpha = c - c0;
	int c1 = c0 >= nCol ? c0 : c0 + 1;

	float r = v * nRow;
	int r0 = (int)r;
	float beta = r - r0;
	int r1 = r0 >= nRow ? r0 : r0 + 1;

	float v0 = (1.0f - alpha) * srcImage[r0 * nCol + c0] + alpha * srcImage[r0 * nCol + c1];
	float v1 = (1.0f - alpha) * srcImage[r1 * nCol + c0] + alpha * srcImage[r1 * nCol + c1];
	return (1.0f - beta) * v0 + beta * v1;
}

void LoadCifar10(int batchSize, int maxImages, bool augment, const std::vector<std::string>& filenames, std::vector<ff::CudaTensor>& images, std::vector<ff::CudaTensor>& labels)
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
	std::vector<float> buffer;
	for (int i = 0; i < numFiles; ++i)
	{
		unsigned char* pCurr = &raw[0];
		FILE* fp = fopen(filenames[i].c_str(), "rb");
		assert(nullptr != fp);
		fread(pCurr, kFileBinarySize, 1, fp);
		fclose(fp);
		for (int j = 0; j < kNumImagePerFile; ++j)
		{
			bool bFlip = false;
			if (true == augment && 1 == rand() % 2) bFlip = true;
			int batchIndex = imageCounter / batchSize;
			int elementIndex = imageCounter % batchSize;
			labels[batchIndex]._data[elementIndex] = static_cast<float>(*pCurr++);
			for (int ch = 0; ch < kNumChannel; ++ch)
			{
				for (int row = 0; row < 32; ++row)
				{
					for (int col = 0; col < 32; ++col)
					{
						float val = *pCurr++;
						int index = elementIndex * kNumBytesPerChannel * kNumChannel + ch * kNumBytesPerChannel;
						if (true == bFlip)
						{
							index += (row * 32 + (31 - col));
						}
						else
						{
							index += (row * 32 + col);
						}
						images[batchIndex]._data[index] = ((val / 255.0f) - mean[ch]) / std[ch];
					}
				}
			}
			if (true == augment)
			{
				int shift = 8;
				int newSize = 32 + shift;
				int baseIndex = elementIndex * kNumBytesPerChannel * kNumChannel;
				buffer.resize(newSize * newSize * kNumChannel);
				for (int ch = 0; ch < kNumChannel; ++ch)
				{
					for (int row = 0; row < newSize; ++row)
					{
						for (int col = 0; col < newSize; ++col)
						{
							buffer[ch * newSize * newSize + row * newSize + col] =
								Bilinear(32, 32, &images[batchIndex]._data[baseIndex + ch * kNumBytesPerChannel],
									static_cast<float>(col) / newSize, static_cast<float>(row) / newSize);
						}
					}
				}
				int rowShift = static_cast<int>(rand() % (shift+1));
				int colShift = static_cast<int>(rand() % (shift+1));
				for (int ch = 0; ch < kNumChannel; ++ch)
				{
					for (int row = 0; row < 32; ++row)
					{
						for (int col = 0; col < 32; ++col)
						{
							images[batchIndex]._data[baseIndex + ch * kNumBytesPerChannel + row * 32 + col] =
								buffer[ch * newSize * newSize + (row + rowShift) * newSize + (col + colShift)];
						}
					}
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
	if(imageCounter > 0) loss /= imageCounter;
	return imageCounter;
}

int cifar10()
{
	const int kBatchSize = 100;

	std::vector<std::string> trainingDataFilenames;
	trainingDataFilenames.push_back("cifar-10/data_batch_1.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_2.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_3.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_4.bin");
	trainingDataFilenames.push_back("cifar-10/data_batch_5.bin");
	std::vector<ff::CudaTensor> trainingImages;
	std::vector<ff::CudaTensor> trainingLabels;
	LoadCifar10(kBatchSize, 5000, false, trainingDataFilenames, trainingImages, trainingLabels);
	std::vector<std::string> testDataFilenames;
	testDataFilenames.push_back("cifar-10/test_batch.bin");
	std::vector<ff::CudaTensor> testImages;
	std::vector<ff::CudaTensor> testLabels;
	LoadCifar10(kBatchSize, 1000, false, testDataFilenames, testImages, testLabels);

#if 1
	ff::CudaNn nn;
	nn.AddConv2d(3, 3, 32, 1, 1);
	nn.AddBatchNorm2d(32);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 32, 64, 1, 1);
	nn.AddBatchNorm2d(64);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 64, 128, 1, 1);
	nn.AddBatchNorm2d(128);
	nn.AddRelu();
	nn.AddConv2d(3, 128, 128, 1, 1);
	nn.AddBatchNorm2d(128);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 128, 256, 1, 1);
	nn.AddBatchNorm2d(256);
	nn.AddRelu();
	nn.AddConv2d(3, 256, 256, 1, 1);
	nn.AddBatchNorm2d(256);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddFc(4 * 256, 1024);
	nn.AddRelu();
	nn.AddFc(1024, 10);
	nn.AddSoftmax();
#else
	ff::CudaNn nn;
	nn.AddConv2d(3, 3, 64, 1, 1);
	nn.AddBatchNorm2d(64);
	nn.AddRelu();
	nn.AddConv2d(3, 64, 64, 1, 1);
	nn.AddBatchNorm2d(64);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 64, 128, 1, 1);
	nn.AddBatchNorm2d(128);
	nn.AddRelu();
	nn.AddConv2d(3, 128, 128, 1, 1);
	nn.AddBatchNorm2d(128);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 128, 256, 1, 1);
	nn.AddBatchNorm2d(256);
	nn.AddRelu();
	nn.AddConv2d(3, 256, 256, 1, 1);
	nn.AddBatchNorm2d(256);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 256, 512, 1, 1);
	nn.AddBatchNorm2d(512);
	nn.AddRelu();
	nn.AddConv2d(3, 512, 512, 1, 1);
	nn.AddBatchNorm2d(512);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddConv2d(3, 512, 512, 1, 1);
	nn.AddBatchNorm2d(512);
	nn.AddRelu();
	nn.AddConv2d(3, 512, 512, 1, 1);
	nn.AddBatchNorm2d(512);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddFc(1 * 512, 1024);
	nn.AddRelu();
	nn.AddFc(1024, 10);
	nn.AddSoftmax();
#endif

	float learningRate = 0.001f;

	float last_train_loss = 0.0f;
	float lowest_train_loss = 1e8f;
	float last_test_loss = 0.0f;
	float lowest_test_loss = 1e8f;
	const int numBatch = (int)trainingImages.size();
	const int kNumEpoch = 200;
	for (int i = 0; i < kNumEpoch; ++i)
	{
		float currLearningRate = learningRate;

		// gradual decay
		//const float kDecay = 0.2f;
		//const int kCooldown = 3;
		//if (i >= kCooldown)
		//{
		//	currLearningRate *= expf(-1.0f * kDecay * (i - kCooldown));
		//}

		char buffer[2048];
		sprintf(buffer, "-- Epoch %03d(lr: %f)", i + 1, currLearningRate);
		ProfileScope __m(buffer);

		// Training
		int trainingImageCounter = 0;
		float train_loss = 0.0f;
		int top1 = 0, top3 = 0, top5 = 0;
		for (int j = 0; j < numBatch; ++j)
		{
			const ff::CudaTensor* pSoftmax = nullptr;
			pSoftmax = nn.Forward(&trainingImages[j], true);
			nn.Backward(&trainingLabels[j]);
			nn.UpdateWs(currLearningRate);

			// train loss
			const_cast<ff::CudaTensor*>(pSoftmax)->PullFromGpu();
			for (int k = 0; k < pSoftmax->_d1; ++k)
			{
				float val = pSoftmax->_data[static_cast<int>(trainingLabels[j]._data[k]) + pSoftmax->_d0 * k];
				assert(val > 0.0f);
				if (val > 0.0f)
				{
					++trainingImageCounter;
					train_loss += -logf(val);
				}
			}
			int t1, t3, t5;
			CheckAccuracy(pSoftmax, trainingLabels[j], t1, t3, t5);
			top1 += t1; top3 += t3; top5 += t5;
		}
		if (trainingImageCounter <= 0) trainingImageCounter = 1;
		train_loss /= trainingImageCounter;
		if (0 == i) last_train_loss = train_loss;
		if (train_loss < lowest_train_loss)
		{
			lowest_train_loss = train_loss;
		}
		printf("Train[%05d](Loss: %f(%+f)/%f, Top1: %05d(%5.2f%%), Top3: %05d, Top5: %05d)\n",
			trainingImageCounter,
			train_loss, train_loss - last_train_loss, lowest_train_loss,
			top1, top1 * 100.0f / trainingImageCounter,
			top3,
			top5);
		last_train_loss = train_loss;

		// Test loss
		{
			int top1 = 0, top3 = 0, top5 = 0;
			float test_loss = 0.0f;
			int testCounter = ComputeLoss(nn, testImages, testLabels, 0, (int)testImages.size(), test_loss, top1, top3, top5);
			if (testCounter <= 0)  testCounter = 1;
			if (0 == i) last_test_loss = test_loss;
			if (test_loss < lowest_test_loss)
			{
				lowest_test_loss = test_loss;
			}
			printf("Test_[%05d](Loss: %f(%+f)/%f, Top1: %05d(%5.2f%%), Top3: %05d, Top5: %05d)\n",
				testCounter,
				test_loss, test_loss - last_test_loss, lowest_test_loss,
				top1, top1 * 100.0f / testCounter,
				top3,
				top5);
			last_test_loss = test_loss;
		}
	}
	return 0;
}
