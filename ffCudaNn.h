#pragma once

#include <vector>

namespace ff
{
	class CudaNn;

	class CudaTensor
	{
	public:
		CudaTensor();
		CudaTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);
		~CudaTensor();

		void ResetTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);

		void Random(const double multiplier = 1.0f);

		void Zero();

		void Dropout(double ratio);

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
		CudaLayer(CudaNn* nn) : _nn(nn) {}

		virtual ~CudaLayer() {}

		virtual const CudaTensor* Forward(const CudaTensor*) = 0;

		virtual const CudaTensor* Backward(const CudaTensor*, const int layerIndex) = 0;

		virtual void UpdateWs(double learningRate, double beta1, double beta2, double beta1t, double beta2t) {}

	public:
		CudaNn* _nn;
	};

	class FcLayer : public CudaLayer
	{
	public:
		FcLayer(CudaNn* nn, int inDim, int outDit);

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

		void UpdateWs(double learningRate, double beta1, double beta2, double beta1t, double beta2t) override;

	public:
		const CudaTensor* _pX;
		CudaTensor _xG;
		CudaTensor _w;
		CudaTensor _wG;
		CudaTensor _wG_m;
		CudaTensor _wG_v;
		CudaTensor _b;
		CudaTensor _bG;
		CudaTensor _bG_m;
		CudaTensor _bG_v;
		CudaTensor _y;
	};

	class ReluFcLayer : public FcLayer
	{
	public:
		ReluFcLayer(CudaNn* nn, int inDim, int outDit);

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		CudaTensor _xRelu;
	};

	class DropoutLayer : public CudaLayer
	{
	public:
		DropoutLayer(CudaNn* nn, double dropoutRate) : CudaLayer(nn), _crossCheck(0), _dropoutRate(dropoutRate) {}

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		int	_crossCheck;
		double _dropoutRate;
		CudaTensor _dropoutMask;
	};

	class SoftmaxLayer : public CudaLayer
	{
	public:
		SoftmaxLayer(CudaNn* nn) : CudaLayer(nn) {}

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		CudaTensor _softmax;
		CudaTensor _lossG;
	};

	class SumOfSquaresLayer : public CudaLayer
	{
	public:
		SumOfSquaresLayer(CudaNn* nn) : CudaLayer(nn), _pY(nullptr) {}

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		const CudaTensor* _pY;
		CudaTensor _yG;
	};

	class CudaNn
	{
	public:
		CudaNn();

		~CudaNn();

		bool InitializeCudaNn(const char* desc);

		bool AddFc(int inDim, int outDim);

		bool AddReluFc(int inDim, int outDim);

		bool AddDropout(double dropoutRatio);

		bool AddSoftmax();

		bool AddSumOfSquares();

		const CudaTensor* Forward(const CudaTensor* x, bool dropout = false);

		void Backward(const CudaTensor* yLabel);

		void UpdateWs(double learningRate);

		bool IsDropoutEnabled() { return _dropoutEnabled; }

	public:
		std::vector<CudaLayer*> _layers;

		const double kBeta1 = 0.9;

		const double kBeta2 = 0.999;

		double _beta1t;

		double _beta2t;

		bool _dropoutEnabled;
	};

} // namespace ff
