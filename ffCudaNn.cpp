#include "ffCudaNn.h"

#include <cuda_runtime.h>
#include <assert.h>
#include <random>
#include <chrono>

#define K_THREAD_PER_BLOCK			512
#define K_SMALL_THREAD_PER_BLOCK	64 

namespace ff
{
	///////////////////////////////////////////////////////////////////////
	//std::default_random_engine g_generator;
	std::default_random_engine g_generator(static_cast<int>(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::uniform_real_distribution<float> g_uniformDistribution;
	static std::normal_distribution<float> g_normalDistribution(0.0f, 1.0f);

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

	CudaTensor::CudaTensor(const CudaTensor& rhs)
	{
		assert(0); // DISALLOW_COPY_AND_ASSIGN
	}

	CudaTensor& CudaTensor::operator=(const CudaTensor& rhs)
	{
		assert(0); // DISALLOW_COPY_AND_ASSIGN
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
			//printf("%s\n", cudaGetErrorName(cudaGetLastError()));
			//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		}
	}

	void CudaTensor::Reshape(int d0, int d1, int d2, int d3)
	{
		assert((d0 * d1 * d2 * d3) == _dataSize);
		_d0 = d0; _d1 = d1; _d2 = d2; _d3 = d3;
	}

	void CudaTensor::SetRandom(const float multiplier)
	{
		for (int i = 0; i < _dataSize; ++i)
		{
			_data[i] = (g_uniformDistribution(g_generator) * 2.0f - 1.0f) * multiplier;
		}
		PushToGpu();
	}

	void CudaTensor::SetZero()
	{
		memset(&_data[0], 0, _data.size() * sizeof(float));
		PushToGpu();
	}

	void CudaTensor::SetDropoutMask(float zeroRatio)
	{
		assert(zeroRatio > 0.0f && zeroRatio < 1.0f);
		float s = 1.0f / (1.0f - zeroRatio);
		for (int i = 0; i < _dataSize; ++i)
		{
			_data[i] = 0.0f;
			if (g_uniformDistribution(g_generator) > zeroRatio)
				_data[i] = s;
		}
		PushToGpu();
	}

	void CudaTensor::PushToGpu()
	{
		cudaError_t err = cudaMemcpy(_dataGpu, &_data[0], _dataSize * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
		assert(err == cudaSuccess);
	}

	void CudaTensor::PullFromGpu()
	{
		cudaError_t err = cudaMemcpy(&_data[0], _dataGpu, _dataSize * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		assert(err == cudaSuccess);
	}

	///////////////////////////////////////////////////////////////////////
	FcLayer::FcLayer(CudaNn* nn, int inDim, int outDim) : CudaLayer(nn), _pX(nullptr)
	{
		_w.ResetTensor(outDim, inDim);
		_w.SetRandom(1.0f / sqrtf((float)inDim));
		_wG.ResetTensor(outDim, inDim);
		_wG_m.ResetTensor(outDim, inDim);
		_wG_m.SetZero();
		_wG_v.ResetTensor(outDim, inDim);
		_wG_v.SetZero();
		_b.ResetTensor(outDim);
		_b.SetRandom(1.0f / sqrtf((float)inDim));
		_bG.ResetTensor(outDim);
		_bG_m.ResetTensor(outDim);
		_bG_m.SetZero();
		_bG_v.ResetTensor(outDim);
		_bG_v.SetZero();
	}

	__global__ void LinearTransform_Cuda(float* y, const float* x, const float* w, const float* b, int nColX, int nColW, int nJobs)
	{
		const int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (yIndex >= nJobs) return;
		int rowX = yIndex / nColW;
		int colW = yIndex % nColW;

		float v = 0.0f;
		int xBaseIndex = rowX * nColX;
		for (int i = 0; i < nColX; ++i)
		{
			v += x[i + xBaseIndex] * w[colW + i * nColW];
		}
		y[yIndex] = v + b[colW];
	}

	const CudaTensor* FcLayer::Forward(const CudaTensor* x)
	{
		int x_d0 = x->_d0;
		int x_d1 = x->_d1;
		if (x_d0 != _w._d1)
		{
			x_d0 = x->_d0 * x->_d1 * x->_d2;
			x_d1 = x->_d3;
		}
		assert(x_d0 == _w._d1);

		_pX = x;
		_y.ResetTensor(_w._d0, x_d1);

		// y = xw+b
		int nJobs = x_d1 * _w._d0;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		LinearTransform_Cuda <<< block, threads >>> (_y._dataGpu, _pX->_dataGpu, _w._dataGpu, _b._dataGpu, x_d0, _w._d0, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_y;
	}

	__global__ void BackwardFc_Wg_Cuda(float* wG, const float* x, const float* yG, int nXcol, int nXrow, int nWcol, int nJobs)
	{
		const int wGindex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (wGindex >= nJobs) return;
		int r = wGindex / nWcol;
		int c = wGindex % nWcol;

		// wG = x.T * yG
		float val = 0.0f;
		for (int i = 0; i < nXrow; ++i)
		{
			val += x[r + i * nXcol] * yG[c + i * nWcol];
		}
		//wG[wGindex] = val / nXrow;
		wG[wGindex] = val;
	}

	__global__ void BackwardFc_Bg_Cuda(float* bG, const float* yG, int nYgCol, int nYgRow)
	{
		const int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c >= nYgCol) return;

		float val = 0.0f;
		for (int i = 0; i < nYgRow; ++i)
		{
			val += yG[c + i * nYgCol];
		}
		//bG[c] = val / nYgRow;
		bG[c] = val;
	}

	__global__ void BackwardFc_Xg_Cuda(float* xG, const float* yG, const float* w, int yGw, int wTh, int xGw, int nJobs)
	{
		const int xGindex = blockIdx.x * blockDim.x + threadIdx.x;
		if (xGindex >= nJobs) return;
		int r = xGindex / xGw;
		int c = xGindex % xGw;

		// xG = yG * w.T
		float val = 0.0f;
		int yGbaseIndex = r * yGw;
		int wBaseIndex = c * wTh;
		for (int i = 0; i < yGw; ++i)
		{
			val += yG[i + yGbaseIndex] * w[i + wBaseIndex];
		}
		xG[xGindex] = val;
	}

	const CudaTensor* FcLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_d0 == _wG._d0);
		int x_d0 = _pX->_d0;
		int x_d1 = _pX->_d1;
		if (x_d0 != _w._d1)
		{
			x_d0 = _pX->_d0 * _pX->_d1 * _pX->_d2;
			x_d1 = _pX->_d3;
		}

		{
			// wG = x.T * yG
			int nJobs = _wG._d1 * _wG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardFc_Wg_Cuda << < block, threads >> > (_wG._dataGpu, _pX->_dataGpu, yG->_dataGpu, x_d0, x_d1, _wG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardFc_Bg_Cuda << < block, threads >> > (_bG._dataGpu, yG->_dataGpu, yG->_d0, yG->_d1);
			assert(cudaGetLastError() == cudaSuccess);
		}

		if (layerIndex > 0)
		{
			assert(yG->_d1 == x_d1);
			_xG.ResetTensor(x_d0, x_d1);
			// xG = yG * w.T
			int nJobs = _xG._d1 * _xG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardFc_Xg_Cuda << < block, threads >> > (_xG._dataGpu, yG->_dataGpu, _w._dataGpu, yG->_d0, _w._d0, _xG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	__global__ void UpdateWs_Cuda(float learningRate, float beta1, float beta2, float beta1t, float beta2t,
		float* w, const float* wG, float* wG_m, float* wG_v, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		// Vanilla
		//w[index] -= wG[index] * learningRate;

		// Adam
		float g = wG[index];
		float m = wG_m[index];
		float v = wG_v[index];
		m = beta1 * m + (1.0f - beta1) * g;
		v = beta2 * v + (1.0f - beta2) * g * g;
		float unbiased_m = m / (1.0f - beta1t);
		float unbiased_v = v / (1.0f - beta2t);
		wG_m[index] = m;
		wG_v[index] = v;
		float impv = learningRate * unbiased_m / (sqrtf(unbiased_v) + 1e-8f);
		w[index] -= impv;
	}

	void FcLayer::UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t)
	{
		{
			int nJobs = _w._d1 * _w._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda <<<block, threads >>> (learningRate, beta1, beta2, beta1t, beta2t, _w._dataGpu, _wG._dataGpu, _wG_m._dataGpu, _wG_v._dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda <<< block, threads >>> (learningRate, beta1, beta2, beta1t, beta2t, _b._dataGpu, _bG._dataGpu, _bG_m._dataGpu, _bG_v._dataGpu, _b._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
	}

	void FcLayer::Pull()
	{
		_xG.PullFromGpu();
		_w.PullFromGpu();
		_wG.PullFromGpu();
		_wG_m.PullFromGpu();
		_wG_v.PullFromGpu();
		_b.PullFromGpu();
		_bG.PullFromGpu();
		_bG_m.PullFromGpu();
		_bG_v.PullFromGpu();
		_y.PullFromGpu();
	}

	Conv2dLayer::Conv2dLayer(CudaNn* nn, int kernelSize, int nInChannel, int nOutChannel, int stride, int padding) :
		CudaLayer(nn), _kernelSize(kernelSize), _stride(stride), _padding(padding)
	{
		_w.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_w.SetRandom(1.0f / sqrtf(float(_kernelSize * _kernelSize * nInChannel)));
		_wG.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_m.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_m.SetZero();
		_wG_v.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_v.SetZero();
		_b.ResetTensor(nOutChannel);
		_b.SetRandom(1.0f / sqrtf(float(_kernelSize * _kernelSize * nInChannel)));
		_bG.ResetTensor(nOutChannel);
		_bG_m.ResetTensor(nOutChannel);
		_bG_m.SetZero();
		_bG_v.ResetTensor(nOutChannel);
		_bG_v.SetZero();
	}

	__global__ void ForwardConv2d_Cuda(
		float* y, const float* x, const float* w, const float* b,
		int nOutChannel, int nRowY, int nColY,
		int nInChannel, int nRowX, int nColX,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		const int yIndex = index;
		int image = index / (nOutChannel * nRowY * nColY);
		index -= image * (nOutChannel * nRowY * nColY);
		int outChannel = index / (nRowY * nColY);
		index -= outChannel * (nRowY * nColY);
		int rowY = index / nColY;
		int colY = index % nColY;

		float val = 0.0f;
		int startRowX = rowY * stride - padding;
		int startColX = colY * stride - padding;
		int startRowKernel= max(0, padding - (rowY * stride));
		int startColKernel = max(0, padding - (colY * stride));
		for (int rowX = startRowX + startRowKernel; rowX < startRowX + kernelSize; ++rowX)
		{
			if (rowX >= nRowX) break;
			for (int colX = startColX + startColKernel; colX < startColX + kernelSize; ++colX)
			{
				if (colX >= nColX) break;
				for (int inChannel = 0; inChannel < nInChannel; ++inChannel)
				{
					int xBaseIndex = image * (nInChannel * nRowX * nColX) + inChannel * (nRowX * nColX) + rowX * nColX;
					int wBaseIndex = outChannel * (nInChannel * kernelSize * kernelSize) + inChannel * (kernelSize * kernelSize) + (rowX - startRowX) * kernelSize;
					val += x[xBaseIndex + colX] * w[wBaseIndex + (colX - startColX)];
				}
			}
		}
		y[yIndex] = val + b[outChannel];
	}

	const CudaTensor* Conv2dLayer::Forward(const CudaTensor* x)
	{
		assert(x->_d2 == _w._d2); // Check # of input channels

		// (N - F) / stride + 1
		_pX = x;
		int nColY = (_pX->_d0 + _padding * 2 - _w._d0) / _stride + 1;
		int nRowY = (_pX->_d1 + _padding * 2 - _w._d1) / _stride + 1;

		// x x y x nOutChannel x nImages 
		_y.ResetTensor(nColY, nRowY, _w._d3, _pX->_d3);

		int nJobs = _y._dataSize;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 blocks(numBlocks), threads(K_THREAD_PER_BLOCK);
		ForwardConv2d_Cuda<<<blocks, threads>>>(
			_y._dataGpu, _pX->_dataGpu, _w._dataGpu, _b._dataGpu,
			_y._d2, _y._d1, _y._d0, 
			_pX->_d2, _pX->_d1, _pX->_d0, 
			_kernelSize, _stride, _padding, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_y;
	}

	__global__ void BackwardConv2d_Wg_Cuda(
		float* wG, const float* x, const float* yG,
		int nOutChannel, int nInChannel,
		int nImages, int nRowY, int nColY, int nRowX, int nColX,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		const int wIndex = index;
		int outChannel = index / (nInChannel * kernelSize * kernelSize);
		index -= outChannel * (nInChannel * kernelSize * kernelSize);
		int inChannel = index / (kernelSize * kernelSize);
		index -= inChannel * (kernelSize * kernelSize);
		int rowW = index / kernelSize;
		int colW = index % kernelSize;

		float val = 0.0f;
		int startRowY = max(int(ceil((padding - rowW) / float(stride))), 0);
		int startColY = max(int(ceil((padding - colW) / float(stride))), 0);
		for (int image = 0; image < nImages; ++image)
		{
			int yBaseIndex = image * (nOutChannel * nRowY * nColY) + outChannel * (nRowY * nColY);
			int xBaseIndex = image * (nInChannel * nRowX * nColX) + inChannel * (nRowX * nColX);
			for (int rowY = startRowY; rowY < nRowY; ++rowY)
			{
				int rowX = rowY * stride - padding + rowW;
				if (rowX >= nRowX) break;
				for (int colY = startColY; colY < nColY; ++colY)
				{
					int colX = colY * stride - padding + colW;
					if (colX >= nColX) break;
					int yIndex = yBaseIndex + rowY * nColY + colY;
					int xIndex = xBaseIndex + rowX * nColX + colX;
					val += x[xIndex] * yG[yIndex];
				}
			}
		}
		//wG[wIndex] = val / nImages;
		wG[wIndex] = val;
	}

	__global__ void BackwardConv2d_Bg_Cuda(float* bG, const float* yG, int nImages, int nOutChannel, int nRowY, int nColY)
	{
		const int outChannel = blockIdx.x * blockDim.x + threadIdx.x;
		if (outChannel >= nOutChannel) return;

		float val = 0.0f;
		for (int image = 0; image < nImages; ++image)
		{
			for (int rowY = 0; rowY < nRowY; ++rowY)
			{
				int yBaseIndex = image * (nOutChannel * nRowY * nColY) + outChannel * (nRowY * nColY) + rowY * nColY;
				for (int colY = 0; colY < nColY; ++colY)
				{
					val += yG[yBaseIndex + colY];
				}
			}
		}
		//bG[outChannel] = val / nImages;
		bG[outChannel] = val;
	}

	__global__ void BackwardConv2d_Xg_Cuda(
		float* xG, const float* w, const float* yG,
		int nImages, int nInChannel, int nRowX, int nColX,
		int nOutChannel, int nRowY, int nColY,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		int xIndex = index;
		int image = index / (nInChannel * nRowX * nColX);
		index -= image * (nInChannel * nRowX * nColX);
		int inChannel = index / (nRowX * nColX);
		index -= inChannel * (nRowX * nColX);
		int rowX = index / nColX;
		int colX = index % nColX;

		float val = 0.0f;
		// Note(dongwook): Iterate all y's which use the current x
		int startRowY = max(0, int(floor(float(rowX + padding - kernelSize) / stride)) + 1);
		int startColY = max(0, int(floor(float(colX + padding - kernelSize) / stride)) + 1);
		for (int rowY = startRowY; rowY < nRowY; ++rowY)
		{
			int upperX = rowY * stride - padding;
			if (rowX < upperX) break;
			int rowW = rowX - upperX;
			for (int colY = startColY; colY < nColY; ++colY)
			{
				int leftX = colY * stride - padding;
				if (colX < leftX) break;
				int colW = colX - leftX;
				for (int outChannel = 0; outChannel < nOutChannel; ++outChannel)
				{
					int yIndex = image * (nOutChannel * nRowY * nColY) + outChannel * (nRowY * nColY) + rowY * nColY + colY;
					int wIndex = outChannel * (nInChannel * kernelSize * kernelSize) + inChannel * (kernelSize * kernelSize) + rowW * kernelSize + colW;
					val += w[wIndex] * yG[yIndex];
				}
			}
		}
		xG[xIndex] = val;
	}

	const CudaTensor* Conv2dLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(_y._dataSize == yG->_dataSize);
		_xG.ResetTensor(_pX->_d0, _pX->_d1, _pX->_d2, _pX->_d3);

		{
			assert(_y._d3 <= 256);
			int nJobs = _wG._dataSize;
			int numBlocks = (nJobs + K_SMALL_THREAD_PER_BLOCK - 1) / K_SMALL_THREAD_PER_BLOCK;
			dim3 blocks(numBlocks), threads(K_SMALL_THREAD_PER_BLOCK);
			BackwardConv2d_Wg_Cuda <<<blocks, threads >>> (
				_wG._dataGpu, _pX->_dataGpu, yG->_dataGpu,
				_wG._d3, _wG._d2,
				_y._d3, _y._d1, _y._d0, _pX->_d1, _pX->_d0,
				_kernelSize, _stride, _padding, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			assert(_bG._d0 == _y._d2);
			int nJobs = _bG._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 blocks(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardConv2d_Bg_Cuda <<< blocks, threads >>> (_bG._dataGpu, yG->_dataGpu, _y._d3, _y._d2, _y._d1, _y._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
		if (layerIndex > 0)
		{
			int nJobs = _xG._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 blocks(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardConv2d_Xg_Cuda <<<blocks, threads >>> (
				_xG._dataGpu, _w._dataGpu, yG->_dataGpu,
				_xG._d3, _xG._d2, _xG._d1, _xG._d0,
				_y._d2, _y._d1, _y._d0,
				_kernelSize, _stride, _padding, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	void Conv2dLayer::UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t)
	{
		{
			int nJobs = _w._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda <<< block, threads >>> (learningRate, beta1, beta2, beta1t, beta2t, _w._dataGpu, _wG._dataGpu, _wG_m._dataGpu, _wG_v._dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda <<< block, threads >>> (learningRate, beta1, beta2, beta1t, beta2t, _b._dataGpu, _bG._dataGpu, _bG_m._dataGpu, _bG_v._dataGpu, _b._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
	}

	void Conv2dLayer::Pull()
	{
		_xG.PullFromGpu();
		_w.PullFromGpu();
		_wG.PullFromGpu();
		_wG_m.PullFromGpu();
		_wG_v.PullFromGpu();
		_b.PullFromGpu();
		_bG.PullFromGpu();
		_bG_m.PullFromGpu();
		_bG_v.PullFromGpu();
		_y.PullFromGpu();
	}

	__global__ void ForwardMaxPool_Cuda(float* y, float* maxIndex, const float* x, int nImages, int nInChannel, int nRow, int nCol)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int nJobs = nImages * nInChannel * nRow * nCol;
		if (nJobs <= index) return;

		const int yIndex = index;
		int image = index / (nInChannel * nRow * nCol);
		index -= image * (nInChannel * nRow * nCol);
		int inChannel = index / (nRow * nCol);
		index -= inChannel * (nRow * nCol);
		int row = index / nCol;
		int col = index % nCol;

		const int offset[4][4] = {
			0, 1, nCol * 2, nCol * 2 + 1,
			1, nCol * 2, nCol * 2 + 1, 0,
			nCol * 2, nCol * 2 + 1, 0, 1,
			nCol * 2 + 1, 0, 1, nCol * 2
		};
		int s = nJobs % 4;
		int xIndex = image * (nInChannel * nRow * nCol * 4) + inChannel * (nRow * nCol * 4) + row * nCol * 4 + col * 2;
		float maxVal = x[xIndex + offset[s][0]], maxIdx = (float)offset[s][0];
		if (maxVal < x[xIndex + offset[s][1]])
		{
			maxIdx = (float)offset[s][1];
			maxVal = x[xIndex + offset[s][1]];
		}
		if (maxVal < x[xIndex + offset[s][2]])
		{
			maxIdx = (float)offset[s][2];
			maxVal = x[xIndex + offset[s][2]];
		}
		if (maxVal < x[xIndex + offset[s][3]])
		{
			maxIdx = (float)offset[s][3];
			maxVal = x[xIndex + offset[s][3]];
		}
		maxIndex[yIndex] = maxIdx;
		y[yIndex] = maxVal;
	}

	const CudaTensor* MaxPoolLayer::Forward(const CudaTensor* x)
	{
		assert(0 == (x->_d0 % 2) && 0 == (x->_d1 % 2)); 

		_pX = x;
		_y.ResetTensor(x->_d0 / 2, x->_d1 / 2, x->_d2, x->_d3);
		_maxIndex.ResetTensor(_y._d0, _y._d1, _y._d2, _y._d3);

		int nJobs = _y._dataSize;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		ForwardMaxPool_Cuda <<< block, threads >>> (_y._dataGpu, _maxIndex._dataGpu, _pX->_dataGpu, _y._d3, _y._d2, _y._d1, _y._d0);
		assert(cudaGetLastError() == cudaSuccess);
		return &_y;
	}

	__global__ void BackwardMaxPool_Cuda(float* xG, const float* yG, const float* maxIndex, int nImages, int nOutChannel, int nRow, int nCol)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int nJobs = nImages * nOutChannel * nRow * nCol;
		if (nJobs <= index) return;

		int yGindex = index;
		int image = index / (nOutChannel * nRow * nCol);
		index -= image * (nOutChannel * nRow * nCol);
		int outChannel = index / (nRow * nCol);
		index -= outChannel * (nRow * nCol);
		int row = index / nCol;
		int col = index % nCol;

		int xGindex = image * (nOutChannel * nRow * nCol * 4) + outChannel * (nRow * nCol * 4) + row * nCol * 4 + col * 2;
		xG[xGindex] = 0.0f;
		xG[xGindex + 1] = 0.0f;
		xG[xGindex + nCol * 2] = 0.0f;
		xG[xGindex + nCol * 2 + 1] = 0.0f;
		xG[xGindex + (int)maxIndex[yGindex]] = yG[yGindex];
	}

	const CudaTensor* MaxPoolLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(_y._dataSize == yG->_dataSize);
		_xG.ResetTensor(2 * _y._d0, 2 * _y._d1, _y._d2, _y._d3);

		if (layerIndex > 0)
		{
			int nJobs = _y._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardMaxPool_Cuda <<< block, threads >>> (_xG._dataGpu, yG->_dataGpu, _maxIndex._dataGpu, _y._d3, _y._d2, _y._d1, _y._d0);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	void MaxPoolLayer::Pull()
	{
		_maxIndex.PullFromGpu();
		_xG.PullFromGpu();
		_y.PullFromGpu();
	}

	__global__ void ForwardRelu_Cuda(float* relu_x, const float* x, int nJobs)
	{
		int jobIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (jobIndex >= nJobs) return;

		relu_x[jobIndex] = fmaxf(x[jobIndex], 0.0f);
	}

	const CudaTensor* ReluLayer::Forward(const CudaTensor* x)
	{
		_pX = x;
		_xRelu.ResetTensor(_pX->_d0, _pX->_d1, _pX->_d2, _pX->_d3);
		int nJobs = _xRelu._dataSize;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		ForwardRelu_Cuda <<< block, threads >>> (_xRelu._dataGpu, _pX->_dataGpu, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_xRelu;
	}

	__global__ void BackwardRelu_Cuda(float* xG, const float* yG, const float* x, int nJobs)
	{
		int jobIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (jobIndex >= nJobs) return;

		xG[jobIndex] = x[jobIndex] < 0.0f ? 0.0f : yG[jobIndex];
	}

	const CudaTensor* ReluLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		if (layerIndex > 0)
		{
			_xG.ResetTensor(_xRelu._d0, _xRelu._d1, _xRelu._d2, _xRelu._d3);
			int nJobs = _xRelu._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardRelu_Cuda <<< block, threads >>> (_xG._dataGpu, yG->_dataGpu, _pX->_dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_xG;
	}

	void ReluLayer::Pull()
	{
		_xRelu.PullFromGpu();
		_xG.PullFromGpu();
	}

	BatchNorm2dLayer::BatchNorm2dLayer(CudaNn* nn, int inDim) : CudaLayer(nn), _pX(nullptr), _accCount(0)
	{
		_meanAndVariance.ResetTensor(2, inDim);
		_meanAndVarianceAcc.ResetTensor(2, inDim);
		_meanAndVarianceAcc.SetZero();
		_meanAndVarianceG.ResetTensor(2, inDim);
		_w.ResetTensor(2, inDim);
		for (int i = 0; i < inDim; ++i)
		{
			_w._data[i*2] = 1.0f; // alpha
			_w._data[i*2+1] = 0.0f; // beta
		}
		_w.PushToGpu();
		_wG.ResetTensor(2, inDim);
		_wG_m.ResetTensor(2, inDim);
		_wG_m.SetZero();
		_wG_v.ResetTensor(2, inDim);
		_wG_v.SetZero();
	}

	template<int BLOCK_SIZE> __global__ void ForwardBatchNorm2d_Train_Cuda(
		float* meanAndVariance, float* meanAndVarianceAcc, float* xHat, float* y,
		const float* w, const float* x, int nRow, int nCol)
	{
		int nChannel = gridDim.x;
		int nImages = blockDim.x;
		int ch = blockIdx.x;
		int image = threadIdx.x;

		int mDash = nImages * nRow * nCol;
		int imageStride = nChannel * nRow * nCol;
		int channelStride = nRow * nCol;
		int currChBaseIndex = ch * channelStride;
		int baseIndex = image * imageStride + currChBaseIndex;

		__shared__ float meanArr[BLOCK_SIZE];
		meanArr[image] = 0.0f;
		for (int i = 0; i < channelStride; ++i)
		{
			meanArr[image] += x[baseIndex + i];
		}
		__syncthreads();
		float mean = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			mean += meanArr[i];
		}
		mean /= mDash;
		
		__shared__ float varianceArr[BLOCK_SIZE];
		varianceArr[image] = 0.0f;
		for (int i = 0; i < channelStride; ++i)
		{
			float a = x[baseIndex + i] - mean;
			varianceArr[image] += (a * a);
		}
		__syncthreads();
		float variance = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			variance += varianceArr[i];
		}
		variance /= mDash;

		float alpha = w[ch * 2 + 0];
		float beta = w[ch * 2 + 1];
		float d = sqrtf(variance) + 1e-8f;
		for (int i = 0; i < channelStride; ++i)
		{
			float xDash = (x[baseIndex + i] - mean) / d;
			xHat[baseIndex + i] = xDash;
			y[baseIndex + i] = alpha * xDash + beta;
		}

		if (threadIdx.x == 0)
		{
			meanAndVariance[ch * 2 + 0] = mean;
			meanAndVariance[ch * 2 + 1] = variance;
			meanAndVarianceAcc[ch * 2 + 0] += mean;
			meanAndVarianceAcc[ch * 2 + 1] += variance;
		}
	}

	__global__ void ForwardBatchNorm2d_Cuda(
		float* y, const float* meanAndVarianceAcc, int accCount,
		const float* w, const float* x, int nRow, int nCol)
	{
		int nChannel = gridDim.x;
		int ch = blockIdx.x;
		int image = threadIdx.x;

		int imageStride = nChannel * nRow * nCol;
		int channelStride = nRow * nCol;
		int currChBaseIndex = ch * channelStride;
		int baseIndex = image * imageStride + currChBaseIndex;

		float alpha = w[ch * 2 + 0];
		float beta = w[ch * 2 + 1];

#if 1	// Note(dongwook): Deteministic network
		float mean = meanAndVarianceAcc[ch * 2 + 0] / accCount;
		float variance = meanAndVarianceAcc[ch * 2 + 1] / (accCount - 1);
		//float variance = (meanAndVarianceAcc[ch * 2 + 1] / (accCount));
#else	// Note(dongwook): Non-deteministic network for batch input.
		int nImages = blockDim.x;
		int mDash = nImages * nRow * nCol;
		__shared__ float meanArr[100];
		meanArr[image] = 0.0f;
		for (int i = 0; i < channelStride; ++i)
		{
			meanArr[image] += x[baseIndex + i];
		}
		__syncthreads();
		float mean = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			mean += meanArr[i];
		}
		mean /= mDash;

		__shared__ float varianceArr[100];
		varianceArr[image] = 0.0f;
		for (int i = 0; i < channelStride; ++i)
		{
			float a = x[baseIndex + i] - mean;
			varianceArr[image] += (a * a);
		}
		__syncthreads();
		float variance = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			variance += varianceArr[i];
		}
		variance /= mDash;
#endif
		float d = 1.0f / (sqrtf(variance) + 1e-8f);
		float a = alpha * d;
		float b = beta - (alpha * mean) * d;
		for (int i = 0; i < channelStride; ++i)
		{
			y[baseIndex + i] = a * x[baseIndex + i] + b;
		}
	}

	const CudaTensor* BatchNorm2dLayer::Forward(const CudaTensor* x)
	{
		assert(_w._d1 == x->_d2);
		_pX = x;
		_xHat.ResetTensor(x->_d0, x->_d1, x->_d2, x->_d3);
		_y.ResetTensor(x->_d0, x->_d1, x->_d2, x->_d3);

		if (_nn->IsTraining())
		{
			assert(x->_d3 <= 256);
			ForwardBatchNorm2d_Train_Cuda <256> <<< x->_d2, x->_d3 >>> (
				_meanAndVariance._dataGpu, _meanAndVarianceAcc._dataGpu, _xHat._dataGpu, _y._dataGpu,
				_w._dataGpu, x->_dataGpu, x->_d1, x->_d0);
			assert(cudaGetLastError() == cudaSuccess);
			++_accCount;
		}
		else
		{
			assert(x->_d3 <= 256);
			ForwardBatchNorm2d_Cuda <<< x->_d2, x->_d3 >>> (
				_y._dataGpu, _meanAndVarianceAcc._dataGpu, _accCount > 1 ? _accCount : 2,
				_w._dataGpu, x->_dataGpu, x->_d1, x->_d0);
			assert(cudaGetLastError() == cudaSuccess);
		}

		return &_y;
	}

	template<int BLOCK_SIZE> __global__ void BackwardBatchNorm2d_Cuda(
		float* wG, float* xG, float* meanAndVarianceG, float* meanAndVarianceAcc,
		const float* w, const float* x, const float* xHat, const float* meanAndVariance, const float* yG,
		int nRow, int nCol)
	{
		int nChannel = gridDim.x;
		int nImages = blockDim.x;
		int ch = blockIdx.x;
		int image = threadIdx.x;

		int mDash = nImages * nRow * nCol;
		int imageStride = nChannel * nRow * nCol;
		int channelStride = nRow * nCol;
		int currChBaseIndex = ch * channelStride;

		__shared__ float alphaArr[BLOCK_SIZE], betaArr[BLOCK_SIZE];
		alphaArr[image] = 0.0f;
		betaArr[image] = 0.0f;
		int baseIndex = image * imageStride + currChBaseIndex;
		for (int i = 0; i < channelStride; ++i)
		{
			float currYg = yG[baseIndex + i];
			alphaArr[image] += currYg * xHat[baseIndex + i];
			betaArr[image] += currYg;
			xG[baseIndex + i] = currYg * w[ch * 2 + 0];
		}
		__syncthreads();
		float alpha = 0.0f, beta = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			alpha += alphaArr[i];
			beta += betaArr[i];
		}
		wG[ch * 2 + 0] = alpha;
		wG[ch * 2 + 1] = beta;

		__shared__ float varianceGarr[BLOCK_SIZE];
		varianceGarr[image] = 0.0f;
		float mean = meanAndVariance[ch * 2 + 0];
		float variance = meanAndVariance[ch * 2 + 1];
		float b = -0.5f * __powf(variance + 1e-8f, -1.5f);
		for (int i = 0; i < channelStride; ++i)
		{
			varianceGarr[image] += (xG[baseIndex + i] * (x[baseIndex + i] - mean) * b);
		}
		__syncthreads();
		float varianceG = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			varianceG += varianceGarr[i];
		}
		meanAndVarianceG[ch * 2 + 1] = varianceG;

		__shared__ float meanGarr[BLOCK_SIZE];
		meanGarr[image] = 0.0f;
		float varianceSqrtInv = 1.0f / (sqrtf(variance) + 1e-8f);
		b = -varianceSqrtInv;
		for (int i = 0; i < channelStride; ++i)
		{
			meanGarr[image] += xG[baseIndex + i] * b + varianceG * ((-2.0f * (x[baseIndex + i] - mean)) / mDash);
		}
		__syncthreads();
		float meanG = 0.0f;
		for (int i = 0; i < nImages; ++i)
		{
			meanG += meanGarr[i];
		}
		meanAndVarianceG[ch * 2 + 0] = meanG;

		b = meanG / mDash;
		for (int i = 0; i < channelStride; ++i)
		{
			float currXg = xG[baseIndex + i];
			xG[baseIndex + i] = currXg * varianceSqrtInv + varianceG * ((2.0f * (x[baseIndex + i] - mean)) / mDash) + b;
		}

		// Reset
		meanAndVarianceAcc[ch * 2 + 0] = 0.0f;
		meanAndVarianceAcc[ch * 2 + 1] = 0.0f;
	}

	const CudaTensor* BatchNorm2dLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		// _pX.shape == _xG.shape == yG.shape == _y.shape
		_xG.ResetTensor(_pX->_d0, _pX->_d1, _pX->_d2, _pX->_d3);

		assert(_xG._d3 <= 256);
		BackwardBatchNorm2d_Cuda<256> <<< _xG._d2, _xG._d3 >>> (
			_wG._dataGpu, _xG._dataGpu, _meanAndVarianceG._dataGpu, _meanAndVarianceAcc._dataGpu,
			_w._dataGpu, _pX->_dataGpu, _xHat._dataGpu, _meanAndVariance._dataGpu, yG->_dataGpu,
			_xG._d1, _xG._d0);
		assert(cudaGetLastError() == cudaSuccess);
		_accCount = 0;
		return &_xG;
	}

	void BatchNorm2dLayer::UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t)
	{
		{
			int nJobs = _w._d1 * _w._d0;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			UpdateWs_Cuda << <block, threads >> > (learningRate, beta1, beta2, beta1t, beta2t, _w._dataGpu, _wG._dataGpu, _wG_m._dataGpu, _wG_v._dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
	}

	void BatchNorm2dLayer::Pull()
	{
		_meanAndVariance.PullFromGpu();
		_meanAndVarianceAcc.PullFromGpu();
		_meanAndVarianceG.PullFromGpu();
		_w.PullFromGpu();
		_wG.PullFromGpu();
		_wG_m.PullFromGpu();
		_wG_v.PullFromGpu();
		_xG.PullFromGpu();
		_xHat.PullFromGpu();
		_y.PullFromGpu();
	}

	__global__ void Dropout_Cuda(float* x, const float* inputX, const float* dropoutMask, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		x[index] = inputX[index] * dropoutMask[index];
	}

	const CudaTensor* DropoutLayer::Forward(const CudaTensor* x)
	{
		if (false == _nn->IsTraining())
		{
			return x;
		}

		_crossCheck = 1;
		_dropoutMask.ResetTensor(x->_dataSize);
		_dropoutMask.SetDropoutMask(_dropoutRate);
		_xDropped.ResetTensor(x->_d0, x->_d1, x->_d2, x->_d3);

		int nJobs = _xDropped._dataSize;
		int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
		Dropout_Cuda<<< block, threads >>>(_xDropped._dataGpu, x->_dataGpu, _dropoutMask._dataGpu, nJobs);
		assert(cudaGetLastError() == cudaSuccess);
		return &_xDropped;
	}

	const CudaTensor* DropoutLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(_dropoutMask._dataSize == yG->_dataSize);
		if (1 != _crossCheck) return yG;
		_crossCheck = 0;

		if (layerIndex > 0)
		{
			_yGdropped.ResetTensor(yG->_d0, yG->_d1, yG->_d2, yG->_d3);
			int nJobs = _yGdropped._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			Dropout_Cuda <<< block, threads >>> (_yGdropped._dataGpu, yG->_dataGpu, _dropoutMask._dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_yGdropped;
	}

	void DropoutLayer::Pull()
	{
		_dropoutMask.PullFromGpu();
		_xDropped.PullFromGpu();
		_yGdropped.PullFromGpu();
	}

	__global__ void ForwardQuatNorm_Cuda(float* y, const float* x, int nQuats, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		int batch = index / nQuats;
		int elem = index % nQuats;
		int baseIndex = batch * nQuats * 4 + elem * 4;
		float q[4] = { x[baseIndex], x[baseIndex + 1], x[baseIndex + 2], x[baseIndex + 3] };
		float l = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) + 1e-8f;
		y[baseIndex+0] = q[0] / l;
		y[baseIndex+1] = q[1] / l;
		y[baseIndex+2] = q[2] / l;
		y[baseIndex+3] = q[3] / l;
	}

	const CudaTensor* QuatNormLayer::Forward(const CudaTensor* x)
	{
		assert(x->_d0 % 4 == 0);
		_pX = x;
		_y.ResetTensor(x->_d0, x->_d1, x->_d2, x->_d3);

		int nJobs = _pX->_d0 * _pX->_d1 / 4;
		int nBlocks = (nJobs + K_SMALL_THREAD_PER_BLOCK - 1) / K_SMALL_THREAD_PER_BLOCK;
		dim3 block(nBlocks), threads(K_SMALL_THREAD_PER_BLOCK);
		ForwardQuatNorm_Cuda <<< block, threads >>> (_y._dataGpu, _pX->_dataGpu, _pX->_d1 / 4, nJobs);
		assert(cudaGetLastError() == cudaSuccess);

		return &_y;
	}

	__global__ void BackwardQuatNorm_Cuda(float* xG, const float* x, const float* yG, int nQuats, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		int batch = index / nQuats;
		int elem = index % nQuats;
		int baseIndex = batch * nQuats * 4 + elem * 4;
		float q[4] = { x[baseIndex], x[baseIndex + 1], x[baseIndex + 2], x[baseIndex + 3] };
		float tYg[4] = { yG[baseIndex], yG[baseIndex + 1], yG[baseIndex + 2], yG[baseIndex + 3] };
		float squaredSum = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
		float a = powf(squaredSum + 1e-8f, -1.5f);
		for (int i = 0; i < 4; ++i)
		{
			float b = squaredSum * tYg[i];
			float c = 0.0f;
			for (int j = 0; j < 4; ++j)
			{
				c += (q[i] * q[j] * tYg[j]);
			}
			xG[baseIndex + i] = a * (b - c);
		}
	}

	const CudaTensor* QuatNormLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_dataSize == _pX->_dataSize);
		_xG.ResetTensor(_pX->_d0, _pX->_d1, _pX->_d2, _pX->_d3);

		if (layerIndex > 0)
		{
			int nJobs = _xG._d0 * _xG._d1 / 4;
			int nBlocks = (nJobs + K_SMALL_THREAD_PER_BLOCK - 1) / K_SMALL_THREAD_PER_BLOCK;
			dim3 block(nBlocks), threads(K_SMALL_THREAD_PER_BLOCK);
			BackwardQuatNorm_Cuda <<< block, threads >>> (_xG._dataGpu, _pX->_dataGpu, yG->_dataGpu, _xG._d0 / 4, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}

		return &_xG;
	}

	void QuatNormLayer::Pull()
	{
		_y.PullFromGpu();
		_xG.PullFromGpu();
	}

	__global__ void ForwardSoftmax_Cuda(float* softmax , const float* x, int nRow, int nCol)
	{
		int r = blockIdx.x * blockDim.x + threadIdx.x;
		if (nRow <= r) return;

		int baseIndex = r * nCol;
		float maxValue = x[baseIndex];
		for (int i = 1; i < nCol; ++i)
		{
			maxValue = max(maxValue, x[baseIndex + i]);
		}
		float sum = 0.0f;
		for (int i = 0; i < nCol; ++i)
		{
			sum += expf(x[baseIndex + i] - maxValue);
		}
		for (int i = 0; i < nCol; ++i)
		{
			softmax[baseIndex + i] = expf(x[baseIndex + i] - maxValue) / sum;
		}
	}

	const CudaTensor* SoftmaxLayer::Forward(const CudaTensor* x)
	{
		assert(1 == x->_d2 && 1 == x->_d3);
		_softmax.ResetTensor(x->_d0, x->_d1);

		int nBlocks = (x->_d1 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
		dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
		ForwardSoftmax_Cuda <<< block, threads >>> (_softmax._dataGpu, x->_dataGpu, x->_d1, x->_d0);
		assert(cudaGetLastError() == cudaSuccess);
		return &_softmax;
	}

	__global__ void BackwardSoftmax_Cuda(float* lossG, const float* softmax, const float* yLabel, int nCol, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;
		int r = index / nCol;
		int c = index % nCol;

		lossG[index] = softmax[index];
		if (yLabel[r] == c) lossG[index] -= 1.0f;
	}

	const CudaTensor* SoftmaxLayer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(yG->_d0 == yG->_dataSize && yG->_d0 == _softmax._d1);
		_lossG.ResetTensor(_softmax._d0, _softmax._d1);
		if (layerIndex > 0)
		{
			int nJobs = _lossG._d1 * _lossG._d0;
			int nBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardSoftmax_Cuda <<< block, threads >>> (_lossG._dataGpu, _softmax._dataGpu, yG->_dataGpu, _lossG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_lossG;
	}

	void SoftmaxLayer::Pull()
	{
		_softmax.PullFromGpu();
		_lossG.PullFromGpu();
	}

	__global__ void BackwardSumOfSqures(float* yG, const float* y, const float* yLabel, int nJobs)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= nJobs) return;

		float diff = y[index] - yLabel[index];
		yG[index] = 2.0f * diff;
	}

	const CudaTensor* SumOfSquaresLayer::Forward(const CudaTensor* x)
	{
		_pY = x;
		return _pY;
	}

	const CudaTensor* SumOfSquaresLayer::Backward(const CudaTensor* yLabel, const int layerIndex)
	{
		assert(yLabel->_dataSize == _pY->_dataSize);
		_yG.ResetTensor(yLabel->_d0, yLabel->_d1, yLabel->_d2, yLabel->_d3);

		if (layerIndex > 0)
		{
			int nJobs = _yG._dataSize;
			int nBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(nBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardSumOfSqures <<< block, threads >>> (_yG._dataGpu, _pY->_dataGpu, yLabel->_dataGpu, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		return &_yG;
	}

	void SumOfSquaresLayer::Pull()
	{
		_yG.PullFromGpu();
	}

	///////////////////////////////////////////////////////////////////////
	CudaNn::CudaNn() : _beta1t(kBeta1), _beta2t(kBeta2), _train(false)
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

	bool CudaNn::AddConv2d(int kernelSize, int nInChannel, int nOutChannel, int stride, int padding)
	{
		_layers.push_back(new Conv2dLayer(this, kernelSize, nInChannel, nOutChannel, stride, padding));
		return true;
	}

	bool CudaNn::AddMaxPool()
	{
		_layers.push_back(new MaxPoolLayer(this));
		return true;
	}

	bool CudaNn::AddFc(int inDim, int outDim)
	{
		_layers.push_back(new FcLayer(this, inDim, outDim));
		return true;
	}

	bool CudaNn::AddRelu()
	{
		_layers.push_back(new ReluLayer(this));
		return true;
	}

	bool CudaNn::AddBatchNorm2d(int inDim)
	{
		_layers.push_back(new BatchNorm2dLayer(this, inDim));
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

	bool CudaNn::AddQuatNorm()
	{
		_layers.push_back(new QuatNormLayer(this));
		return true;
	}

	const CudaTensor* CudaNn::Forward(const CudaTensor* x, bool train)
	{
		_train = train;
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
		size_t numLayer = _layers.size();
		for (size_t i = 0; i < numLayer; ++i)
		{
			_layers[i]->UpdateWs(learningRate, kBeta1, kBeta2, _beta1t, _beta2t);
		}
		_beta1t *= kBeta1;
		_beta2t *= kBeta2;
	}

	void CudaNn::Pull()
	{
		size_t numLayer = _layers.size();
		for (size_t i = 0; i < numLayer; ++i)
		{
			_layers[i]->Pull();
		}
	}

} // namespace ff
