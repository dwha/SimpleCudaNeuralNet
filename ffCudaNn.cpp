#include "ffCudaNn.h"

#include <cuda_runtime.h>
#include <random>
#include <assert.h>

namespace ff
{
	///////////////////////////////////////////////////////////////////////
	static std::default_random_engine g_generator;
	static std::normal_distribution<double> g_distribution;

	CudaTensor::CudaTensor() : _d0(0), _d1(0), _d2(0), _d3(0), _dataSize(0), _dataGpu(nullptr), _dataGpuSize(0)
	{
	}

	CudaTensor::CudaTensor(int d0, int d1, int d2, int d3) : _dataGpu(nullptr), _dataGpuSize(0)
	{
		ResetTensor(d0, d1, d2, d3);
	}

	CudaTensor::~CudaTensor()
	{
		if (nullptr != _dataGpu) cudaFree(_dataGpu);
	}

	void CudaTensor::ResetTensor(int d0, int d1, int d2, int d3)
	{
		_d0 = d0; _d1 = d1; _d2 = d2; _d3 = d3;
		_dataSize = _d0 * _d1 * _d2 * _d3;
		_data.clear();
		_data.resize(_dataSize);

		if (_dataGpuSize < _dataSize)
		{
			_dataGpuSize = _dataSize;
			if (_dataGpu) cudaFree(_dataGpu);
			cudaError_t err = cudaMalloc(&_dataGpu, _dataGpuSize * sizeof(double));
			assert(err == cudaSuccess);
		}
	}

	void CudaTensor::Random(const double multiplier)
	{
		for (int i = 0; i < _dataSize; ++i)
		{
			_data[i] = g_distribution(g_generator) * multiplier;
		}
		cudaMemcpy(_dataGpu, &_data[0], _dataSize * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	void CudaTensor::Zero()
	{
		memset(&_data[0], 0, _data.size() * sizeof(double));
		cudaMemcpy(_dataGpu, &_data[0], _dataSize * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	void CudaTensor::Push()
	{
		cudaMemcpy(_dataGpu, &_data[0], _dataSize * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	void CudaTensor::Pull()
	{
		cudaMemcpy(&_data[0], _dataGpu, _dataSize * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	///////////////////////////////////////////////////////////////////////
	__global__ void LinearTransform_Cuda(double* y, const double* x, const double* w, const double* b, int xw, int ww)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		double v = 0.0;
		for (int i = 0; i < xw; ++i)
		{
			v += x[i + r * xw] * w[c + i * ww];
		}
		y[c + r * ww] = v;
	}

	void LinearTransform(CudaTensor* y, const CudaTensor* x, const CudaTensor* w, const CudaTensor* b)
	{
		//y = xw+b
		dim3 block(x->_d1), threads(w->_d0);
		LinearTransform_Cuda <<< block, threads >>> (y->_dataGpu, x->_dataGpu, w->_dataGpu, b->_dataGpu, x->_d0, w->_d0);
		assert(cudaGetLastError() == cudaSuccess);
	}

	__global__ void ComputeWg_Cuda(double* wG, const double* x, const double* yG, int xTh, int xTw, int yGw)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		double v = 0.0;
		for (int i = 0; i < xTw; ++i)
		{
			v += x[r + i * xTh] * yG[c + i * yGw];
		}
		wG[c + r * yGw] = v;
	}

	void ComputeWg(CudaTensor* wG, const CudaTensor* x, const CudaTensor* yG)
	{
		// wG = x.T * yG
		dim3 block(x->_d0), threads(yG->_d0);
		ComputeWg_Cuda << < block, threads >> > (wG->_dataGpu, x->_dataGpu, yG->_dataGpu, x->_d0, x->_d1, yG->_d0);
		assert(cudaGetLastError() == cudaSuccess);
	}

	__global__ void ComputeXg_Cuda(double* xG, const double* yG, const double* w, int yGw, int yGh, int wTh)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		double v = 0.0;
		for (int i = 0; i < yGw; ++i)
		{
			v += yG[i + r * yGw] * w[i + c * wTh];
		}
		xG[c + r * yGh] = v;
	}

	void ComputeXg(CudaTensor* xG, const CudaTensor* yG, const CudaTensor* w)
	{
		// xG = yG * w.T
		dim3 block(yG->_d1), threads(w->_d1);
		ComputeWg_Cuda << < block, threads >> > (xG->_dataGpu, yG->_dataGpu, w->_dataGpu, yG->_d0, yG->_d1, w->_d0);
		assert(cudaGetLastError() == cudaSuccess);
	}

	__global__ void ComputeSumOfSquresGradient(double* yG, const double* y, const double* yLabel, int nCol)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		int index = c + r * nCol;
		double diff = y[index] - yLabel[index];
		yG[index] = 2.0f * diff;
	}

	__global__ void UpdateWs_Cuda(int nCol, double learningRate, double* w, const double* wG)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		int index = c + r * nCol;
		w[index] -= wG[index] * learningRate;
	}

	__global__ void Relu_Cuda(double* relu_x, const double* x, int nCol)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		int index = c + r * nCol;
		relu_x[index] = fmax(x[index], 0.0);
	}

	__global__ void ReluG_Cuda(double* xG, const double* x, int nCol)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		int index = c + r * nCol;
		xG[index] = xG[index] * (fmax(x[index], 1e-32) / x[index]);
		//if (x[index] < 0.0) xG[index] = 0.0;
	}

	__global__ void SoftmaxBackward_Cuda(double* lossG, const double* softmax, const double* yLabel, int nCol)
	{
		int r = blockIdx.x;
		int c = threadIdx.x;
		int index = c + r * nCol;
		lossG[index] = softmax[index];
		if ((int)yLabel[r] == c) lossG[index] -= 1.0;
	}

	///////////////////////////////////////////////////////////////////////
	FcLayer::FcLayer(int inDim, int outDim) : _pX(nullptr)
	{
		_w.ResetTensor(outDim, inDim);
		_w.Random(1.0 / sqrt(inDim));
		_wG.ResetTensor(outDim, inDim);
		_b.ResetTensor(outDim);
		_b.Zero();
		_bG.ResetTensor(outDim);
	}

	const CudaTensor* FcLayer::Forward(const CudaTensor* x)
	{
		if (x->_d0 != _w._d1)
			return nullptr;

		_pX = x;
		_y.ResetTensor(_w._d0, _pX->_d1);
		LinearTransform(&_y, _pX, &_w, &_b);
		return &_y;
	}

	const CudaTensor* FcLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		ComputeWg(&_wG, _pX, yG);
		if (layerIndex > 0)
		{
			_xG.ResetTensor(_pX->_d0, _pX->_d1);
			ComputeXg(&_xG, yG, &_w);
		}
		return &_xG;
	}

	void FcLayer::UpdateWs(double learningRate)
	{
		dim3 block(_w._d1), threads(_w._d0);
		UpdateWs_Cuda <<< block, threads >>> (_w._d0, learningRate, _w._dataGpu, _wG._dataGpu);
		assert(cudaGetLastError() == cudaSuccess);
	}

	ReluFcLayer::ReluFcLayer(int inDim, int outDim) : _pX(nullptr)
	{
		_w.ResetTensor(outDim, inDim);
		_w.Random(1.0 / sqrt(inDim));
		_wG.ResetTensor(outDim, inDim);
		_b.ResetTensor(outDim);
		_b.Zero();
		_bG.ResetTensor(outDim);
	}

	const CudaTensor* ReluFcLayer::Forward(const CudaTensor* x)
	{
		if (x->_d0 != _w._d1)
			return nullptr;

		_pX = x;
		_xRelu.ResetTensor(_pX->_d0, _pX->_d1);
		dim3 block(_xRelu._d1), threads(_xRelu._d0);
		Relu_Cuda<<< block, threads >>>(_xRelu._dataGpu, _pX->_dataGpu, _xRelu._d0);
		assert(cudaGetLastError() == cudaSuccess);

		_y.ResetTensor(_w._d0, _pX->_d1);
		LinearTransform(&_y, &_xRelu, &_w, &_b);
		return &_y;
	}

	const CudaTensor* ReluFcLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		ComputeWg(&_wG, &_xRelu, yG);
		if (layerIndex > 0)
		{
			_xG.ResetTensor(_pX->_d0, _pX->_d1);
			ComputeXg(&_xG, yG, &_w);
			dim3 block(_xG._d1), threads(_xG._d0);
			ReluG_Cuda<<< block, threads >>> (_xG._dataGpu, _pX->_dataGpu, _xG._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	void ReluFcLayer::UpdateWs(double learningRate)
	{
		dim3 block(_w._d1), threads(_w._d0);
		UpdateWs_Cuda <<< block, threads >>> (_w._d0, learningRate, _w._dataGpu, _wG._dataGpu);
		assert(cudaGetLastError() == cudaSuccess);
	}

	const CudaTensor* SoftmaxLayer::Forward(const CudaTensor* x)
	{
		_softmax.ResetTensor(x->_d0, x->_d1);
		_lossG.ResetTensor(x->_d0, x->_d1);

		const_cast<CudaTensor*>(x)->Pull();
		for (int r = 0; r < x->_d1; ++r)
		{
			double maxValue = x->_data[0 + x->_d0 * r];
			for (int i = 1; i < x->_d0; ++i)
			{
				double currValue = x->_data[i + x->_d0 * r];
				if (maxValue < currValue)
				{
					maxValue = currValue;
				}
			}

			double sum = 0.0;
			for (int i = 0; i < x->_d0; ++i)
			{
				sum += exp(x->_data[i + x->_d0 * r] - maxValue); // stable softmax
			}
			for (int i = 0; i < x->_d0; ++i)
			{
				_softmax._data[i + _softmax._d0 * r] = exp(x->_data[i + x->_d0 * r] - maxValue) / sum;
			}
		}
		_softmax.Push();
		return &_softmax;
	}

	const CudaTensor* SoftmaxLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		dim3 block(yG->_d1), threads(_softmax._d0);
		SoftmaxBackward_Cuda <<< block, threads >>> (_lossG._dataGpu, _softmax._dataGpu, yG->_dataGpu, _lossG._d0);
		assert(cudaGetLastError() == cudaSuccess);
		return &_lossG;
	}

	SumOfSquaresLayer::SumOfSquaresLayer() : _pY(nullptr)
	{
	}

	const CudaTensor* SumOfSquaresLayer::Forward(const CudaTensor* x)
	{
		_pY = x;
		return _pY;
	}

	const CudaTensor* SumOfSquaresLayer::Backward(const CudaTensor* yLabel, const int layerIndex)
	{
		_yG.ResetTensor(yLabel->_d0, yLabel->_d1);

		dim3 block(_yG._d1), threads(_yG._d0);
		ComputeSumOfSquresGradient <<< block, threads >>> (_yG._dataGpu, _pY->_dataGpu, yLabel->_dataGpu, _yG._d0);
		assert(cudaGetLastError() == cudaSuccess);
		return &_yG;
	}

	///////////////////////////////////////////////////////////////////////
	CudaNn::~CudaNn()
	{
		InitializeCudaNn("");
	}

	bool CudaNn::InitializeCudaNn(const char* desc)
	{
		size_t numLayers = _layers.size();
		for (size_t i = 0; i < numLayers; ++i)
		{
			delete _layers[i];
		}
		_layers.clear();

		return true;
	}

	bool CudaNn::AddFc(int inDim, int outDim)
	{
		_layers.push_back(new FcLayer(inDim, outDim));
		return true;
	}

	bool CudaNn::AddReluFc(int inDim, int outDim)
	{
		_layers.push_back(new ReluFcLayer(inDim, outDim));
		return true;
	}

	bool CudaNn::AddSoftmax()
	{
		_layers.push_back(new SoftmaxLayer);
		return true;
	}

	bool CudaNn::AddSumOfSquares()
	{
		_layers.push_back(new SumOfSquaresLayer);
		return true;
	}

	const CudaTensor* CudaNn::Forward(const CudaTensor* x)
	{
		const CudaTensor* y = nullptr;
		size_t numLayer = _layers.size();
		for (size_t i = 0; i < numLayer; ++i)
		{
			if (nullptr == x)
				return nullptr;

			y = _layers[i]->Forward(x);
			x = y;
		}

		return y;
	}

	void CudaNn::Backward(const CudaTensor* yLabel)
	{
		const CudaTensor* y = yLabel;
		const CudaTensor* yGradient = nullptr;
		int numLayer = (int)_layers.size();
		for (int i = 0; i < numLayer; ++i)
		{
			int layerIndex = numLayer - i - 1;
			yGradient = _layers[layerIndex]->Backward(y, layerIndex);
			y = yGradient;
		}
	}

	void CudaNn::UpdateWs(double learningRate)
	{
		int numLayer = (int)_layers.size();
		for (int i = 0; i < numLayer; ++i)
		{
			_layers[i]->UpdateWs(learningRate);
		}
	}
} // namespace ff