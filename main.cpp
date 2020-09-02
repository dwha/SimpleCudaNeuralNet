#include <stdio.h>
#include "ffCudaNn.h"

int mnist();
int cifar10();

int simple()
{
	ff::CudaNn nn;
	nn.AddFc(1000, 500);
	nn.AddDropout(0.5);
	nn.AddReluFc(500, 10);
	nn.AddSumOfSquares();

	ff::CudaTensor x(1000, 200);
	ff::CudaTensor y(10, 200);
	x.Random();
	y.Random();

	float learningRate = 0.0001f;
	const ff::CudaTensor* yPred = nullptr;
	for (int i = 0; i < 500; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			nn.Forward(&x, true);
			nn.Backward(&y);
			nn.UpdateWs(learningRate);
		}

		yPred = nn.Forward(&x);
		const_cast<ff::CudaTensor*>(yPred)->Pull();

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
		printf("loss: %f\n", loss);
	}
	return 0;
}

int main()
{
	return cifar10();
	//return mnist();
	//return simple();
}