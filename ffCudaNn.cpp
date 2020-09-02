#include "ffCudaNn.h"

#include <cuda_runtime.h>
#include <random>
#include <assert.h>

#define K_THREAD_PER_BLOCK 1024

namespace ff
{
	///////////////////////////////////////////////////////////////////////
	static std::default_random_engine g_generator;
	static std::uniform_real_distribution<float> g_uniformDistribution;
	static std::normal_distribution<float> g_normalDistribution;


	CudaTensor::CudaTensor() : _d0(0), _d1(0), _d2(0), _d3(0), _dataSize(0), _dataGpu(nullptr), _dataGpuSize(0)
	{
	}

	CudaTensor::CudaTensor(int d0, int d1, int d2, int d3) : _dataGpu(nullptr), _dataGpuSize(0)
	{
		ResetTensor(d0, d1, d2, d3);
	}

	CudaTensor::CudaTensor(const CudaTensor& rhs)
	{
		ResetTensor(rhs._d0, rhs._d1, rhs._d2, rhs._d3);
		_data = rhs._data;
	}

	CudaTensor::~CudaTensor()
	{
		if (nullptr != _dataGpu) cudaFree(_dataGpu);
	}

	CudaTensor& CudaTensor::operator=(const CudaTensor& rhs)
	{
		ResetTensor(rhs._d0, rhs._d1, rhs._d2, rhs._d3);
		_data = rhs._data;
		return *this;
	}

	void CudaTensor::ResetTensor(int d0, int d1, int d2, int d3)
	{
		_d0 = d0; _d1 = d1; _d2 = d2; _d3 = d3;
		_dataSize = _d0 * _d1 * _d2 * _d3;
		_data.resize(_dataSize);

		if (_dataGpuSize < _dataSize)
		{
			_dataGpuSize = _dataSize;
			if (_dataGpu) cudaFree(_dataGpu);
			cudaError_t err = cudaMalloc(&_dataGpu, _dataGpuSize * sizeof(float));
			assert(err == cudaSuccess);
		}
	}

	void CudaTensor::Random(const float multiplier)
	{
		for (int i = 0; i < _dataSize; ++i)
		{
			_data[i] = g_normalDistribution(g_generator) * multiplier;
		}
		Push();
	}

	void CudaTensor::Zero()
	{
		memset(&_data[0], 0, _data.size() * sizeof(float));
		Push();
	}

	void CudaTensor::Dropout(float ratio)
	{
		assert(ratio > 0.0f && ratio < 1.0f);
		float s = 1.0f / (1.0f - ratio);
		for (int i = 0; i < _dataSize; ++i)
		{
			_data[i] = 0.0;
			if (g_uniformDistribution(g_generator) > ratio)
				_data[i] = s;
		}
		Push();
	}

	void CudaTensor::Push()
	{
		cudaError_t err = cudaMemcpy(_dataGpu, &_data[0], _dataSize * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		assert(err == cudaSuccess);
	}

	void CudaTensor::Pull()
	{
		cudaError_t err = cudaMemcpy(&_data[0], _dataGpu, _dataSize * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		assert(err == cudaSuccess);
	}

	///////////////////////////////////////////////////////////////////////
	__global__ void LinearTransform_Cuda(float* y, const float* x, const float* w, const float* b, int xw, int ww, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;
		int r = jobIndex / ww;
		int c = jobIndex % ww;

		float v = 0.0;
		for (int i = 0; i < xw; ++i)
		{
			v += x[i + r * xw] * w[c + i * ww];
		}
		y[c + r * ww] = v + b[c];
	}

	__global__ void ComputeWg_Cuda(float* wG, const float* x, const float* yG, int nXcol, int nXrow, int nWcol, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;
		int r = jobIndex / nWcol;
		int c = jobIndex % nWcol;

		// wG = x.T * yG
		float v = 0.0;
		for (int i = 0; i < nXrow; ++i)
		{
			v += x[r + i * nXcol] * yG[c + i * nWcol];
		}
		wG[c + r * nWcol] = v;
	}

	__global__ void ComputeXg_Cuda(float* xG, const float* yG, const float* w, int yGw, int wTh, int xGw, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;
		int r = jobIndex / xGw;
		int c = jobIndex % xGw;

		// xG = yG * w.T
		float v = 0.0;
		for (int i = 0; i < yGw; ++i)
		{
			v += yG[i + r * yGw] * w[i + c * wTh];
		}
		xG[c + r * xGw] = v;
	}

	__global__ void ComputeBg_Cuda(float* bG, const float* yG, int nYgCol, int nYgRow)
	{
		int c = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (c >= nYgCol) return;

		bG[c] = 0.0;
		for (int i = 0; i < nYgRow; ++i)
		{
			bG[c] += yG[c + i * nYgRow];
		}
	}

	__global__ void ComputeSumOfSquresGradient(float* yG, const float* y, const float* yLabel, int nCol, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		float diff = y[index] - yLabel[index];
		yG[index] = 2.0f * diff;
	}

	__global__ void UpdateWs_Cuda(int nCol, float learningRate, float beta1, float beta2, float beta1t, float beta2t,
		float* w, const float* wG, float* wG_m, float* wG_v, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		// Vanilla
		//w[index] -= wG[index] * learningRate;

		// Adam
		wG_m[index] = beta1 * wG_m[index] + (1.0 - beta1) * wG[index];
		wG_v[index] = beta2 * wG_v[index] + (1.0 - beta2) * wG[index] * wG[index];
		float unbiased_m = wG_m[index] / (1.0 - beta1t);
		float unbiased_v = wG_v[index] / (1.0 - beta2t);
		w[index] -= (learningRate * unbiased_m / (sqrtf(unbiased_v) + 1e-8f));
	}

	__global__ void UpdateBs_Cuda(float learningRate, float beta1, float beta2, float beta1t, float beta2t,
		float* b, const float* bG, float* bG_m, float* bG_v, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		// Vanilla
		//b[index] -= bG[index] * learningRate;

		// Adam
		bG_m[index] = beta1 * bG_m[index] + (1.0 - beta1) * bG[index];
		bG_v[index] = beta2 * bG_v[index] + (1.0 - beta2) * bG[index] * bG[index];
		float unbiased_m = bG_m[index] / (1.0 - beta1t);
		float unbiased_v = bG_v[index] / (1.0 - beta2t);
		b[index] -= (learningRate * unbiased_m / (sqrtf(unbiased_v) + 1e-8f));
	}

	__global__ void Relu_Cuda(float* relu_x, const float* x, int nCol, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;

		relu_x[jobIndex] = fmaxf(x[jobIndex], 0.0);
	}

	__global__ void ReluG_Cuda(float* xG, const float* x, int nCol, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;

		if (x[jobIndex] < 0.0) xG[jobIndex] = 0.0;
	}

	__global__ void ForwardSoftmax_Step1_Cuda(float* sum, const float* x, int nRow, int nCol)
	{
		int r = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (nRow <= r) return;

		sum[0 + r * nCol] = 1e-8;
		for (int i = 0; i < nCol; ++i)
		{
			sum[0 + r * nCol] += exp(x[i + r * nCol]);
		}
	}

	__global__ void ForwardSoftmax_Step2_Cuda(float* softmax, const float* sum, const float* x, int nRow, int nCol)
	{
		int r = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (nRow <= r) return;

		for (int i = 0; i < nCol; ++i)
		{
			softmax[i + r * nCol] = exp(x[i + r * nCol]) / sum[0 + r * nCol];
		}
	}

	__global__ void SoftmaxBackward_Cuda(float* lossG, const float* softmax, const float* yLabel, int nCol, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;
		int r = index / nCol;
		int c = index % nCol;

		lossG[index] = softmax[index];
		if (yLabel[r] == c) lossG[index] -= 1.0;
	}

	__global__ void Dropout_Cuda(float* x, const float* dropoutMask, int nCol, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		x[index] *= dropoutMask[index];
	}

	///////////////////////////////////////////////////////////////////////
	FcLayer::FcLayer(CudaNn* nn, int inDim, int outDim) : CudaLayer(nn), _pX(nullptr)
	{
		_w.ResetTensor(outDim, inDim);
		//_w.Random(1.0 / sqrtf(inDim)); // Xavier initialization
		_w.Random(1.0f / sqrtf(inDim * 0.5f)); // He initialization
		_wG.ResetTensor(outDim, inDim);
		_wG_m.ResetTensor(outDim, inDim);
		_wG_m.Zero();
		_wG_v.ResetTensor(outDim, inDim);
		_wG_v.Zero();
		_b.ResetTensor(outDim);
		_b.Zero();
		_bG.ResetTensor(outDim);
		_bG_m.ResetTensor(outDim);
		_bG_m.Zero();
		_bG_v.ResetTensor(outDim);
		_bG_v.Zero();
	}

	const CudaTensor* FcLayer::Forward(const CudaTensor* x)
	{
		assert(x->_d0 == _w._d1);

		_pX = x;
		_y.ResetTensor(_w._d0, _pX->_d1);

		// y = xw+b
		int nJobs = x->_d1 * _w._d0;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		LinearTransform_Cuda << < block, threads >> > (_y._dataGpu, _pX->_dataGpu, _w._dataGpu, _b._dataGpu, _pX->_d0, _w._d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_y;
	}

	const CudaTensor* FcLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_d0 == _wG._d0);
		{
			// wG = x.T * yG
			int nJobs = _wG._d1 * _wG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			ComputeWg_Cuda <<< block, threads >>> (_wG._dataGpu, _pX->_dataGpu, yG->_dataGpu, _pX->_d0, _pX->_d1, _wG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			ComputeBg_Cuda <<< block, threads >>> (_bG._dataGpu, yG->_dataGpu, yG->_d0, yG->_d1);
			assert(cudaGetLastError() == cudaSuccess);
		}

		if (layerIndex > 0)
		{
			assert(yG->_d1 == _pX->_d1);
			_xG.ResetTensor(_pX->_d0, _pX->_d1);
			// xG = yG * w.T
			int nJobs = _xG._d1 * _xG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			ComputeXg_Cuda << < block, threads >> > (_xG._dataGpu, yG->_dataGpu, _w._dataGpu, yG->_d0, _w._d0, _xG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	void FcLayer::UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t)
	{
		{
			int nJobs = _w._d1 * _w._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda <<<block, threads >>> (_w._d0, learningRate, beta1, beta2, beta1t, beta2t, _w._dataGpu, _wG._dataGpu, _wG_m._dataGpu, _wG_v._dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateBs_Cuda <<< block, threads >>> (learningRate, beta1, beta2, beta1t, beta2t, _b._dataGpu, _bG._dataGpu, _bG_m._dataGpu, _bG_v._dataGpu, _b._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
	}

	ReluFcLayer::ReluFcLayer(CudaNn* nn, int inDim, int outDim) : FcLayer(nn, inDim, outDim)
	{
	}

	const CudaTensor* ReluFcLayer::Forward(const CudaTensor* x)
	{
		assert(x->_d0 == _w._d1);

		_pX = x;
		_xRelu.ResetTensor(_pX->_d0, _pX->_d1);
		{
			int nJobs = _xRelu._d1 * _xRelu._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			Relu_Cuda <<< block, threads >>> (_xRelu._dataGpu, _pX->_dataGpu, _xRelu._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}

		_y.ResetTensor(_w._d0, _xRelu._d1);
		{
			// y = xw+b
			int nJobs = _xRelu._d1 * _w._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			LinearTransform_Cuda <<< block, threads >>> (_y._dataGpu, _xRelu._dataGpu, _w._dataGpu, _b._dataGpu, _xRelu._d0, _w._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}

		return &_y;
	}

	const CudaTensor* ReluFcLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_d0 == _wG._d0);
		{
			int nJobs = _wG._d1 * _wG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			ComputeWg_Cuda <<< block, threads >>> (_wG._dataGpu, _xRelu._dataGpu, yG->_dataGpu, _xRelu._d0, _xRelu._d1, _wG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}

		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			ComputeBg_Cuda << < block, threads >> > (_bG._dataGpu, yG->_dataGpu, yG->_d0, yG->_d1);
			assert(cudaGetLastError() == cudaSuccess);
		}

		if (layerIndex > 0)
		{
			{
				assert(yG->_d1 == _pX->_d1);
				_xG.ResetTensor(_pX->_d0, _pX->_d1);
				int nJobs = _xG._d1 * _xG._d0;
				int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
				dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
				ComputeXg_Cuda <<< block, threads >>> (_xG._dataGpu, yG->_dataGpu, _w._dataGpu, yG->_d0, _w._d0, _xG._d0, nJobs);
				assert(cudaGetLastError() == cudaSuccess);
			}
			{
				int nJobs = _xG._d1 * _xG._d0;
				int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
				dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
				ReluG_Cuda <<< block, threads >>> (_xG._dataGpu, _pX->_dataGpu, _xG._d0, nJobs);
				assert(cudaGetLastError() == cudaSuccess);
			}
		}

		return &_xG;
	}

	const CudaTensor* DropoutLayer::Forward(const CudaTensor* x)
	{
		if (false == _nn->IsDropoutEnabled())
		{
			return x;
		}

		_crossCheck = 1;
		_dropoutMask.ResetTensor(x->_d0, x->_d1);
		_dropoutMask.Dropout(_dropoutRate);

		int nJobs = x->_d1 * x->_d0;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		Dropout_Cuda<<< block, threads >>>(x->_dataGpu, _dropoutMask._dataGpu, x->_d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return x;
	}

	const CudaTensor* DropoutLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		if (1 != _crossCheck)
		{
			return yG;
		}
		_crossCheck = 0;

		assert(yG->_d0 == _dropoutMask._d0 && yG->_d1 == _dropoutMask._d1);
		int nJobs = yG->_d1 * yG->_d0;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		Dropout_Cuda<<< block, threads >>>(yG->_dataGpu, _dropoutMask._dataGpu, yG->_d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return yG;
	}

	const CudaTensor* SoftmaxLayer::Forward(const CudaTensor* x)
	{
		_softmax.ResetTensor(x->_d0, x->_d1);
		_lossG.ResetTensor(x->_d0, x->_d1);

#if 1
		int nBlocks = (x->_d1 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
		ForwardSoftmax_Step1_Cuda <<< block, threads >>> (_lossG._dataGpu, x->_dataGpu, x->_d1, x->_d0);
		assert(cudaGetLastError() == cudaSuccess);
		ForwardSoftmax_Step2_Cuda <<< block, threads >>> (_softmax._dataGpu, _lossG._dataGpu, x->_dataGpu, x->_d1, x->_d0);
		assert(cudaGetLastError() == cudaSuccess);
#else
		const_cast<CudaTensor*>(x)->Pull();
		for (int r = 0; r < x->_d1; ++r)
		{
			float maxValue = x->_data[0 + x->_d0 * r];
			for (int i = 1; i < x->_d0; ++i)
			{
				float currValue = x->_data[i + x->_d0 * r];
				if (maxValue < currValue)
				{
					maxValue = currValue;
				}
			}

			float sum = 0.0;
			for (int i = 0; i < x->_d0; ++i)
			{
				sum += exp(x->_data[i + x->_d0 * r] - maxValue); // stable softmax
			}
			for (int i = 0; i < x->_d0; ++i)
			{
				_softmax._data[i + _softmax._d0 * r] = expf(x->_data[i + x->_d0 * r] - maxValue) / sum;
			}
		}
		_softmax.Push();
#endif
		return &_softmax;
	}

	const CudaTensor* SoftmaxLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_d0 == _lossG._d1);
		int nJobs = _lossG._d1 * _lossG._d0;
		int nBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
		SoftmaxBackward_Cuda <<< block, threads >>> (_lossG._dataGpu, _softmax._dataGpu, yG->_dataGpu, _lossG._d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_lossG;
	}

	const CudaTensor* SumOfSquaresLayer::Forward(const CudaTensor* x)
	{
		_pY = x;
		return _pY;
	}

	const CudaTensor* SumOfSquaresLayer::Backward(const CudaTensor* yLabel, const int layerIndex)
	{
		_yG.ResetTensor(yLabel->_d0, yLabel->_d1);

		int nJobs = _yG._d1 * _yG._d0;
		int nBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
		ComputeSumOfSquresGradient << < block, threads >> > (_yG._dataGpu, _pY->_dataGpu, yLabel->_dataGpu, _yG._d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_yG;
	}

	///////////////////////////////////////////////////////////////////////
	CudaNn::CudaNn() : _beta1t(kBeta1), _beta2t(kBeta2), _dropoutEnabled(false)
	{
	}

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
		_layers.push_back(new FcLayer(this, inDim, outDim));
		return true;
	}

	bool CudaNn::AddReluFc(int inDim, int outDim)
	{
		_layers.push_back(new ReluFcLayer(this, inDim, outDim));
		return true;
	}

	bool CudaNn::AddDropout(float dropoutRatio)
	{
		assert(dropoutRatio > 0.0 && dropoutRatio < 1.0);
		_layers.push_back(new DropoutLayer(this, dropoutRatio));
		return true;
	}

	bool CudaNn::AddSoftmax()
	{
		_layers.push_back(new SoftmaxLayer(this));
		return true;
	}

	bool CudaNn::AddSumOfSquares()
	{
		_layers.push_back(new SumOfSquaresLayer(this));
		return true;
	}

	const CudaTensor* CudaNn::Forward(const CudaTensor* x, bool dropout)
	{
		_dropoutEnabled = dropout;
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

	void CudaNn::UpdateWs(float learningRate)
	{
		int numLayer = (int)_layers.size();
		for (int i = 0; i < numLayer; ++i)
		{
			_layers[i]->UpdateWs(learningRate, kBeta1, kBeta2, _beta1t, _beta2t);
		}
		_beta1t *= kBeta1;
		_beta2t *= kBeta2;
	}
} // namespace ff