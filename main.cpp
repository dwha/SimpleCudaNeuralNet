#include <stdio.h>
#include <math.h>
#include <random>
#include "ffCudaNn.h"

int mnist();
int cifar10();

namespace ff
{
	extern std::default_random_engine g_generator;
	extern std::uniform_real_distribution<float> g_uniformDistribution;
}

void EulerToQuat(float* q, float yaw, float pitch, float roll)
{
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);

    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

void NormalizeQuat(float* q)
{
	float a = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	if(a > 0.0f)
	{
		q[0] /= a;
		q[1] /= a;
		q[2] /= a;
		q[3] /= a;
	}
}

int TestQuatNorm()
{
	ff::CudaTensor x(3 * 64, 32);
	ff::CudaTensor y(4 * 64, 32);
	ff::CudaTensor xTest(3 * 64, 32);
	ff::CudaTensor yTest(4 * 64, 32);
	for (int batch = 0; batch < xTest._d1; ++batch)
	{
		for (int elem = 0; elem < 64; ++elem)
		{
			int baseIndexX = batch * 64 * 3 + elem * 3;
			int baseIndexY = batch * 64 * 4 + elem * 4;
			float yaw = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
			float pitch = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
			float roll = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
			xTest._data[baseIndexX] = yaw;
			xTest._data[baseIndexX + 1] = pitch;
			xTest._data[baseIndexX + 2] = roll;
			EulerToQuat(&yTest._data[baseIndexY], yaw, pitch, roll);
		}
	}
	xTest.PushToGpu();
	yTest.PushToGpu();

	float learningRate = 0.001f;
	ff::CudaNn nn;
	nn.AddFc(3*64, 1000);
	nn.AddRelu();
	nn.AddFc(1000, 4 * 64);
	nn.AddQuatNorm();
	nn.AddSumOfSquares();

	float lastLoss[1000];
	for (int i = 0; i < 120000; ++i)
	{
		if (i == 49999)
		{
			learningRate *= 0.1f;
		}
		if (i == 99999)
		{
			learningRate *= 0.1f;
		}
		for (int batch = 0; batch < x._d1; ++batch)
		{
			for (int elem = 0; elem < 64; ++elem)
			{
				int baseIndexX = batch * 64 * 3 + elem * 3;
				int baseIndexY = batch * 64 * 4 + elem * 4;
				float yaw = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
				float pitch = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
				float roll = (ff::g_uniformDistribution(ff::g_generator) * 2.0f - 1.0f) * 3.141592f;
				x._data[baseIndexX] = yaw;
				x._data[baseIndexX + 1] = pitch;
				x._data[baseIndexX + 2] = roll;
				EulerToQuat(&y._data[baseIndexY], yaw, pitch, roll);
			}
		}
		x.PushToGpu();
		y.PushToGpu();

		for (int j = 0; j < 1; ++j)
		{
			nn.Forward(&x, true);
			nn.Backward(&y);
			nn.UpdateWs(learningRate);
		}

		ff::CudaTensor* yPred = const_cast<ff::CudaTensor*>(nn.Forward(&xTest));
		yPred->PullFromGpu();

		float loss = 0.0;
		for (int r = 0; r < yPred->_d1; ++r)
		{
			for (int c = 0; c < yPred->_d0; c+=4)
			{
				int index = c + r * yPred->_d0;
				NormalizeQuat(&yPred->_data[index]);
				float aa = yPred->_data[index + 0] - yTest._data[index + 0];
				float bb = yPred->_data[index + 1] - yTest._data[index + 1];
				float cc = yPred->_data[index + 2] - yTest._data[index + 2];
				float dd = yPred->_data[index + 3] - yTest._data[index + 3];
				loss += sqrtf(aa * aa + bb * bb + cc * cc + dd * dd);
			}
		}
		loss /= (yPred->_d1 * yPred->_d0 / 4);
		lastLoss[i % 1000] = loss;
		if (0 == i % 1000)
			printf("[%05d]loss: %f\n", i, loss);
	}

	float loss = 0.0f;
	for (int i = 0; i < 1000; ++i)
	{
		loss += lastLoss[i];
	}
	printf("Last 1000's loss: %f\n", loss / 1000.0f);
	return 0;

}

int simple()
{
#if 1
	float learningRate = 0.0001f;
	ff::CudaNn nn;
	nn.AddFc(1000, 2000);
	nn.AddRelu();
	nn.AddFc(2000, 500);
	nn.AddRelu();
	nn.AddFc(500, 500);
	nn.AddRelu();
	nn.AddFc(500, 10);
	nn.AddSumOfSquares();

	ff::CudaTensor x(1000, 256);
	ff::CudaTensor y(10, 256);
	x.SetRandom();
	y.SetRandom();
#else
	float learningRate = 0.001f;
	ff::CudaNn nn;
	nn.AddConv2d(3, 1, 16, 1, 1);
	nn.AddRelu();
	nn.AddConv2d(3, 16, 32, 1, 1);
	nn.AddRelu();
	nn.AddConv2d(3, 32, 64, 1, 1);
	nn.AddRelu();
	nn.AddMaxPool();
	nn.AddFc(4*4*64, 10);
	nn.AddSumOfSquares();

	ff::CudaTensor x(8, 8, 1, 256);
	ff::CudaTensor y(10, 256);
	x.SetRandom();
	y.SetRandom();
#endif

	const ff::CudaTensor* yPred = nullptr;
	for (int i = 0; i < 10000; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			nn.Forward(&x, true);
			nn.Backward(&y);
			nn.UpdateWs(learningRate);
		}

		yPred = nn.Forward(&x);
		const_cast<ff::CudaTensor*>(yPred)->PullFromGpu();

		float loss = 0.0;
		for (int r = 0; r < yPred->_d1; ++r)
		{
			for (int c = 0; c < yPred->_d0; ++c)
			{
				int index = c + r * yPred->_d0;
				float diff = yPred->_data[index] - y._data[index];
				loss += (diff * diff);
			}
		}
		printf("[%05d]loss: %f\n", i, loss / yPred->_d1);

	}
	return 0;
}

int main()
{
	return TestQuatNorm();
	//return cifar10();
	//return mnist();
	//return simple();
}
