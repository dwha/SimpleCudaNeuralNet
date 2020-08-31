#include "ffCudaNn.h"

int main()
{
	ff::CudaNn nn;
	nn.AddFc(1000, 500);
	nn.AddReluFc(500, 100);
	nn.AddReluFc(100, 10);
	nn.AddSumOfSquares();

	ff::CudaTensor x(1000, 200);
	ff::CudaTensor y(10, 200);
	x.Random();
	y.Random();

	double learningRate = 0.0001;
	const ff::CudaTensor* yPred = nullptr;
	for (int i = 0; i < 10000; ++i)
	{
		yPred = nn.Forward(&x);
		if (nullptr == yPred)
		{
			printf("Error: at Forward()\n");
			return -1;
		}

		nn.Backward(&y);
		nn.UpdateWs(learningRate);

		ff::CudaTensor* __yPred = const_cast<ff::CudaTensor*>(yPred);
		__yPred->Pull();

		double loss = 0.0;
		for (int r = 0; r < yPred->_d1; ++r)
		{
			for (int c = 0; c < yPred->_d0; ++c)
			{
				int index = c + r * yPred->_d0;
				double diff = yPred->_data[index] - y._data[index];
				loss += (diff * diff);
			}
		}
		printf("loss: %f\n", loss);
	}
	return 0;
}