#pragma once

#include <vector>

namespace ff
{
	class CudaTensor
	{
	public:
		CudaTensor();
		CudaTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);
		~CudaTensor();

		void ResetTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);

		void Random(const double multiplier = 1.0f);

		void Zero();

		void Push();

		void Pull();

	public:
		int _d0, _d1, _d2, _d3, _dataSize;
		std::vector<double> _data;

		double* _dataGpu;
		int _dataGpuSize;
	};

	class CudaLayer
	{
	public:
		virtual ~CudaLayer() {}

		virtual const CudaTensor* Forward(const CudaTensor*) = 0;

		virtual const CudaTensor* Backward(const CudaTensor*, const int layerIndex) = 0;

		virtual void UpdateWs(double learningRate) {}
	};

	class FcLayer : public CudaLayer
	{
	public:
		FcLayer(int inDim, int outDit);

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

		void UpdateWs(double learningRate) override;

	public:
		const CudaTensor* _pX;
		CudaTensor _xG;
		CudaTensor _w;
		CudaTensor _wG;
		CudaTensor _b;
		CudaTensor _bG;
		CudaTensor _y;
	};

	class ReluFcLayer : public CudaLayer
	{
	public:
		ReluFcLayer(int inDim, int outDit);

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

		void UpdateWs(double learningRate) override;

	public:
		const CudaTensor* _pX;
		CudaTensor _xRelu;
		CudaTensor _xG;
		CudaTensor _w;
		CudaTensor _wG;
		CudaTensor _b;
		CudaTensor _bG;
		CudaTensor _y;
	};

	class SoftmaxLayer : public CudaLayer
	{
	public:
		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		CudaTensor _softmax;
		CudaTensor _lossG;
	};

	class SumOfSquaresLayer : public CudaLayer
	{
	public:
		SumOfSquaresLayer();

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		const CudaTensor* _pY;
		CudaTensor _yG;
	};

	class CudaNn
	{
	public:
		~CudaNn();

		bool InitializeCudaNn(const char* desc);

		bool AddFc(int inDim, int outDim);

		bool AddReluFc(int inDim, int outDim);

		bool AddSoftmax();

		bool AddSumOfSquares();

		const CudaTensor* Forward(const CudaTensor* x);

		void Backward(const CudaTensor* yLabel);

		void UpdateWs(double learningRate);

	public:
		std::vector<CudaLayer*> _layers;
	};

	void SetLinearTransform(CudaTensor* y, const CudaTensor* x, const CudaTensor* w, const CudaTensor* b);

} // namespace ff
