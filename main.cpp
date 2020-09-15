#include <stdio.h>
#include "ffCudaNn.h"

int mnist();
int cifar10();

int simple()
{
#if 1
	float learningRate = 0.0001f;
	ff::CudaNn nn;
	nn.AddFc(1000, 1000);
	nn.AddRelu();
	nn.AddFc(1000, 500);
	nn.AddDropout(0.5f);
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
	//return cifar10();
	//return mnist();
	return simple();
}
