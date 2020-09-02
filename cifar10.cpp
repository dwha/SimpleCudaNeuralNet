#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include "ffCudaNn.h"

void CheckAccuracy(const ff::CudaTensor* pSoftmax, const ff::CudaTensor& yLabel, int& top1, int& top3, int& top5);

void LoadCifar10(int trainingBatchSize, std::vector<ff::CudaTensor>& trainingImages, std::vector<ff::CudaTensor>& trainingLabels,
	ff::CudaTensor& testImages, ff::CudaTensor& testLabels)
{
	const int kTrainingDataSize = 50000;
	const int kTestDataSize = 10000;
	const int kNumBytesPerImage = 1024 * 3;
	int numBatches = (kTrainingDataSize + trainingBatchSize - 1) / trainingBatchSize;
	trainingImages.resize(numBatches);
	trainingLabels.resize(numBatches);
	int nLeft = kTrainingDataSize;
	for (int i = 0; i < numBatches; ++i)
	{
		int currBatchSize = (trainingBatchSize < nLeft ? trainingBatchSize : nLeft);
		//trainingImages[i].ResetTensor(32, 32, 3, currBatchSize);
		trainingImages[i].ResetTensor(kNumBytesPerImage, currBatchSize);
		trainingLabels[i].ResetTensor(currBatchSize);
		nLeft -= trainingBatchSize;
	}
	//testImages.ResetTensor(32, 32, 3, kTestDataSize);
	testImages.ResetTensor(kNumBytesPerImage, kTestDataSize);
	testLabels.ResetTensor(kTestDataSize);

	int counter = 0;
	const char* filenames[5] = {
		"cifar-10/data_batch_1.bin",
		"cifar-10/data_batch_2.bin", 
		"cifar-10/data_batch_3.bin", 
		"cifar-10/data_batch_4.bin",
		"cifar-10/data_batch_5.bin" };
	std::vector<unsigned char> raw(30730000);
	for (int i = 0; i < 5; ++i)
	{
		unsigned char* pCurr = &raw[0];
		FILE* fp = fopen(filenames[i], "rb");
		assert(nullptr != fp);
		fread(pCurr, 30730000, 1, fp);
		fclose(fp);
		for (int i = 0; i < 10000; ++i)
		{
			int batchIndex = counter / trainingBatchSize;
			int elementIndex = counter % trainingBatchSize;
			++counter;
			trainingLabels[batchIndex]._data[elementIndex] = static_cast<double>(*pCurr++);
			for (int j = 0; j < kNumBytesPerImage; ++j)
			{
				trainingImages[batchIndex]._data[elementIndex * kNumBytesPerImage + j] = *pCurr++ / 255.0;
			}
		}
	}
	for (size_t i = 0; i < trainingImages.size(); ++i)
	{
		trainingImages[i].Push();
		trainingLabels[i].Push();
	}

	FILE* fp = fopen("cifar-10/test_batch.bin", "rb");
	assert(nullptr != fp);
	unsigned char* pCurr = &raw[0];
	fread(pCurr, 30730000, 1, fp);
	fclose(fp);
	for (int i = 0; i < 10000; ++i)
	{
		testLabels._data[i] = static_cast<double>(*pCurr++);
		for (int j = 0; j < kNumBytesPerImage; ++j)
		{
			testImages._data[i * kNumBytesPerImage + j] = *pCurr++ / 255.0;
		}
	}
	testImages.Push();
	testLabels.Push();
}

int cifar10()
{
	const int kBatchSize = 200;

	std::vector<ff::CudaTensor> trainingImages;
	std::vector<ff::CudaTensor> trainingLabels;
	ff::CudaTensor testImages;
	ff::CudaTensor testLabels;
	LoadCifar10(kBatchSize, trainingImages, trainingLabels, testImages, testLabels);

	ff::CudaNn nn;
	nn.InitializeCudaNn("");
	nn.AddFc(1024 * 3, 4096);
	nn.AddReluFc(4096, 4096);
	nn.AddDropout(0.5);
	nn.AddReluFc(4096, 1000);
	nn.AddReluFc(1000, 10);
	nn.AddSoftmax();

	const int numEpoch = 10000;
	const size_t numBatch = trainingImages.size();

	double learningRate = 0.0005;
	double loss = 0.0;
	double last_validation_loss = 1e12;
	double lowest_validation_loss = 1e12;
	double lowest_test_loss = 1e12;
	int top1 = 0, top3 = 0, top5 = 0;
	int currValidationDataIndex = 0;
	int numValidationData = static_cast<int>(numBatch) / 5;
	ff::CudaTensor* pSoftmax = nullptr;

	printf("* Initial learning rate(%f)\n", learningRate);
	for (int i = 0; i < numEpoch; ++i)
	{
		printf("-- Epoch[%03d] ...\n", i + 1);

		// Training
		for (size_t j = 0; j < numBatch; ++j)
		{
			// Note(dongwook): Exclude validation set
			if (currValidationDataIndex <= j && j < currValidationDataIndex + numValidationData)
			{
				continue;
			}

			nn.Forward(&trainingImages[j], true);
			nn.Backward(&trainingLabels[j]);
			nn.UpdateWs(learningRate);
		}

		// Validation loss
		loss = 0.0f;
		top1 = top3 = top5 = 0;
		int cntVal = 0;
		for (int j = currValidationDataIndex; j < currValidationDataIndex + numValidationData; ++j)
		{
			cntVal += trainingImages[j]._d1;
			pSoftmax = const_cast<ff::CudaTensor*>(nn.Forward(&trainingImages[j]));
			pSoftmax->Pull();
			for (int k = 0; k < trainingImages[j]._d1; ++k)
			{
				loss += -log(pSoftmax->_data[static_cast<int>(trainingLabels[j]._data[k]) + pSoftmax->_d0 * k]);
			}
			int t1, t3, t5;
			CheckAccuracy(pSoftmax, trainingLabels[j], t1, t3, t5);
			top1 += t1; top3 += t3; top5 += t5;
		}
		loss /= cntVal;
		if (loss < lowest_validation_loss)
		{
			lowest_validation_loss = loss;
		}
		if (loss > last_validation_loss)
		{
			// Learning rate decay
			learningRate *= 0.33333f;
			printf("- Learning rate decreased(%f)\n", learningRate);
		}
		last_validation_loss = loss;
		printf("Val_[%05d](Loss: %f/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n", cntVal, loss, lowest_validation_loss,
			top1,
			top3,
			top5);

		// Rotate validation set
		currValidationDataIndex += numValidationData;
		if (currValidationDataIndex + numValidationData > numBatch)
		{
			currValidationDataIndex = 0;
		}

		// Test loss
		pSoftmax = const_cast<ff::CudaTensor*>(nn.Forward(&testImages));
		pSoftmax->Pull();

		loss = 0.0f;
		for (int j = 0; j < testImages._d1; ++j)
		{
			loss += -log(pSoftmax->_data[static_cast<int>(testLabels._data[j]) + pSoftmax->_d0 * j]);
		}
		loss /= testImages._d1;
		if (loss < lowest_test_loss)
		{
			lowest_test_loss = loss;
		}

		CheckAccuracy(pSoftmax, testLabels, top1, top3, top5);
		printf("Test[%05d](Loss: %f/%f, Top1: %05d, Top3: %05d, Top5: %05d)\n", pSoftmax->_d1, loss, lowest_test_loss,
			top1,
			top3,
			top5);
	}
	return 0;
}