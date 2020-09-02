#pragma once

#include <vector>

namespace ff
{
	class CudaNn;

#define DISALLOW_COPY_AND_ASSIGN(TypeName)	\
	TypeName(TypeName&) = delete;			\
	void operator=(TypeName) = delete;

	class CudaTensor
	{
	public:
		CudaTensor();

		CudaTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);

		CudaTensor(const CudaTensor& rhs);

		~CudaTensor();

		CudaTensor& operator=(const CudaTensor& rhs);

		void ResetTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1);

		void Random(const float multiplier = 1.0f);

		void Zero();

		void Dropout(float ratio);

		void Push();

		void Pull();

	public:
		int _d0, _d1, _d2, _d3, _dataSize;
		std::vector<float> _data;

		int _dataGpuSize;
		float* _dataGpu;
	};

	class CudaLayer
	{
	public:
		CudaLayer(CudaNn* nn) : _nn(nn) {}

		virtual ~CudaLayer() {}

		virtual const CudaTensor* Forward(const CudaTensor*) = 0;

		virtual const CudaTensor* Backward(const CudaTensor*, const int layerIndex) = 0;

		virtual void UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t) {}

	public:
		CudaNn* _nn;
	};

	class FcLayer : public CudaLayer
	{
	public:
		FcLayer(CudaNn* nn, int inDim, int outDit);

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

		void UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t) override;

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
		DropoutLayer(CudaNn* nn, float dropoutRate) : CudaLayer(nn), _crossCheck(0), _dropoutRate(dropoutRate) {}

		const CudaTensor* Forward(const CudaTensor*) override;

		const CudaTensor* Backward(const CudaTensor*, const int layerIndex) override;

	public:
		int	_crossCheck;
		float _dropoutRate;
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

		bool AddDropout(float dropoutRatio);

		bool AddSoftmax();

		bool AddSumOfSquares();

		const CudaTensor* Forward(const CudaTensor* x, bool dropout = false);

		void Backward(const CudaTensor* yLabel);

		void UpdateWs(float learningRate);

		bool IsDropoutEnabled() { return _dropoutEnabled; }

	public:
		std::vector<CudaLayer*> _layers;

		const float kBeta1 = 0.9f;

		const float kBeta2 = 0.999f;

		float _beta1t;

		float _beta2t;

		bool _dropoutEnabled;
	};

} // namespace ff
