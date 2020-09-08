#include "ffCudaNn.h"

#include <cuda_runtime.h>
#include <assert.h>
#include <random>
#include <chrono>

#define K_THREAD_PER_BLOCK 1024

namespace ff
{
	///////////////////////////////////////////////////////////////////////
	//static std::default_random_engine g_generator;
	static std::default_random_engine g_generator(static_cast<int>(std::chrono::steady_clock::now().time_since_epoch().count()));
	static std::uniform_real_distribution<float> g_uniformDistribution;
	static std::normal_distribution<float> g_normalDistribution(0.0f, 0.1f);

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
			_data[i] = g_normalDistribution(g_generator) * multiplier;
			//_data[i] = (g_uniformDistribution(g_generator) * 2.0f - 1.0f) * multiplier;
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

	void CudaTensor::WriteOut(int nCol, int nRow, const char* filename)
	{
		PullFromGpu();
		assert(nCol * nRow * 3 <= _dataSize);
		FILE* fp = fopen(filename, "wt");
		assert(fp != nullptr);
		fprintf(fp, "P3\n%d %d\n255\n", nCol, nRow);
		for (int i = 0; i < nRow; ++i)
		{
			for (int j = 0; j < nCol; ++j)
			{
				for (int c = 0; c < 3; ++c)
				{
					float v = _data[c * nCol * nRow + i * nRow + j];
					fprintf(fp, "%d ", (int)(v * 255.0f));
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	///////////////////////////////////////////////////////////////////////
	__global__ void LinearTransform_Cuda(float* y, const float* x, const float* w, const float* b, int nColX, int nColW, int nJobs)
	{
		const int yIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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
		wG[wGindex] = val / nXrow;
		//wG[wGindex] = val;
	}

	__global__ void BackwardFc_Bg_Cuda(float* bG, const float* yG, int nYgCol, int nYgRow)
	{
		const int c = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (c >= nYgCol) return;

		float val = 0.0f;
		for (int i = 0; i < nYgRow; ++i)
		{
			val += yG[c + i * nYgCol];
		}
		bG[c] = val / nYgRow;
		//bG[c] = val;
	}

	__global__ void BackwardFc_Xg_Cuda(float* xG, const float* yG, const float* w, int yGw, int wTh, int xGw, int nJobs)
	{
		const int xGindex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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

	__global__ void ForwardConv2d_Cuda(
		float* y, const float* x, const float* w, const float* b,
		int nOutChannel, int nRowY, int nColY,
		int nInChannel, int nRowX, int nColX,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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
		for (int inChannel = 0; inChannel < nInChannel; ++inChannel)
		{
			for (int rowX = startRowX; rowX < startRowX + kernelSize; ++rowX)
			{
				if (rowX < 0 || rowX >= nRowX) continue;
				int xBaseIndex = image * (nInChannel * nRowX * nColX) + inChannel * (nRowX * nColX) + rowX * nColX;
				int wBaseIndex = outChannel * (nInChannel * kernelSize * kernelSize) + inChannel * (kernelSize * kernelSize) + (rowX - startRowX) * kernelSize;
				for (int colX = startColX; colX < startColX + kernelSize; ++colX)
				{
					if (colX < 0 || colX >= nColX) continue;
					val += x[xBaseIndex + colX] * w[wBaseIndex + (colX - startColX)];
				}
			}
		}
		y[yIndex] = val + b[outChannel];
	}

	__global__ void BackwardConv2d_Wg_Cuda(
		float* wG, const float* x, const float* yG,
		int nOutChannel, int nInChannel,
		int nImages, int nRowY, int nColY, int nRowX, int nColX,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		const int wIndex = index;
		int outChannel = index / (nInChannel * kernelSize * kernelSize);
		index -= outChannel * (nInChannel * kernelSize * kernelSize);
		int inChannel = index / (kernelSize * kernelSize);
		index -= inChannel * (kernelSize * kernelSize);
		int rowW = index / kernelSize;
		int colW = index % kernelSize;

		float val = 0.0f;
		for (int image = 0; image < nImages; ++image)
		{
			for (int rowY = 0; rowY < nRowY; ++rowY)
			{
				int rowX = rowY * stride - padding + rowW;
				if (rowX < 0 || rowX >= nRowX) continue;
				for (int colY = 0; colY < nColY; ++colY)
				{
					int colX = colY * stride - padding + colW;
					if (colX < 0 || colX >= nColX) continue;
					int yIndex = image * (nOutChannel * nRowY * nColY) + outChannel * (nRowY * nColY) + rowY * nColY + colY;
					int xIndex = image * (nInChannel * nRowX * nColX) + inChannel * (nRowX * nColX) + rowX * nColX + colX;
					val += x[xIndex] * yG[yIndex];
				}
			}
		}
		wG[wIndex] = val / nImages;
		//wG[wIndex] = val;
	}

	__global__ void BackwardConv2d_Bg_Cuda(float* bG, const float* yG, int nImages, int nOutChannel, int nRowY, int nColY)
	{
		const int outChannel = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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
		bG[outChannel] = val / nImages;
		//bG[outChannel] = val;
	}

	__global__ void BackwardConv2d_Xg_Cuda(
		float* xG, const float* w, const float* yG,
		int nImages, int nInChannel, int nRowX, int nColX, 
		int nOutChannel, int nRowY, int nColY,
		int kernelSize, int stride, int padding, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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
		for (int rowY = 0; rowY < nRowY; ++rowY)
		{
			int upperX = rowY * stride - padding;
			if (rowX < upperX) break;
			if (upperX + kernelSize <= rowX) continue;
			int rowW = rowX - upperX;
			for (int colY = 0; colY < nColY; ++colY)
			{
				int leftX = colY * stride - padding;
				if (colX < leftX) break;
				if (leftX + kernelSize <= colX) continue;
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

	__global__ void BackwardSumOfSqures(float* yG, const float* y, const float* yLabel, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		float diff = y[index] - yLabel[index];
		yG[index] = 2.0f * diff;
	}

	__global__ void UpdateWs_Cuda(float learningRate, float beta1, float beta2, float beta1t, float beta2t,
		float* w, const float* wG, float* wG_m, float* wG_v, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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

	__global__ void ForwardRelu_Cuda(float* relu_x, const float* x, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;

		relu_x[jobIndex] = fmaxf(x[jobIndex], 0.0f);
	}

	__global__ void BackwardRelu_Cuda(float* xG, const float* yG, const float* x, int nJobs)
	{
		int jobIndex = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (jobIndex >= nJobs) return;

		xG[jobIndex] = x[jobIndex] < 0.0f ? 0.0f : yG[jobIndex];
	}

	__global__ void ForwardSoftmax_Cuda(float* softmax , const float* x, int nRow, int nCol)
	{
		int r = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (nRow <= r) return;

		int baseIndex = r * nCol;
#if 0
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
#else
		float sum = 0.0f;
		for (int i = 0; i < nCol; ++i)
		{
			sum += expf(x[baseIndex + i]);
		}
		for (int i = 0; i < nCol; ++i)
		{
			softmax[baseIndex + i] = expf(x[baseIndex + i]) / sum;
		}
#endif
	}

	__global__ void BackwardSoftmax_Cuda(float* lossG, const float* softmax, const float* yLabel, int nCol, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;
		int r = index / nCol;
		int c = index % nCol;

		lossG[index] = softmax[index];
		if (yLabel[r] == c) lossG[index] -= 1.0f;
	}

	__global__ void Dropout_Cuda(float* x, const float* inputX, const float* dropoutMask, int nJobs)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		if (index >= nJobs) return;

		x[index] = inputX[index] * dropoutMask[index];
	}

	__global__ void ForwardMaxPool_Cuda(float* y, float* maxIndex, const float* x, int nImages, int nInChannel, int nRow, int nCol)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
		int nJobs = nImages * nInChannel * nRow * nCol;
		if (nJobs <= index) return;

		const int yIndex = index;
		int image = index / (nInChannel * nRow * nCol);
		index -= image * (nInChannel * nRow * nCol);
		int inChannel = index / (nRow * nCol);
		index -= inChannel * (nRow * nCol);
		int row = index / nCol;
		int col = index % nCol;

		int xIndex = image * (nInChannel * nRow * nCol * 4) + inChannel * (nRow * nCol * 4) + row * nCol * 4 + col * 2;
		float maxVal = x[xIndex], maxIdx = 0.0f;
		if (maxVal < x[xIndex + 1])
		{
			maxIdx = (float)1;
			maxVal = x[xIndex + 1];
		}
		if (maxVal < x[xIndex + nCol * 2])
		{
			maxIdx = (float)(nCol * 2);
			maxVal = x[xIndex + nCol * 2];
		}
		if (maxVal < x[xIndex + nCol * 2 + 1])
		{
			maxIdx = (float)(nCol * 2 + 1);
			maxVal = x[xIndex + nCol * 2 + 1];
		}
		maxIndex[yIndex] = maxIdx;
		y[yIndex] = maxVal;
	}

	__global__ void BackwardMaxPool_Cuda(float* xG, const float* yG, const float* maxIndex, int nImages, int nOutChannel, int nRow, int nCol)
	{
		int index = blockIdx.x * K_THREAD_PER_BLOCK + threadIdx.x;
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

	///////////////////////////////////////////////////////////////////////
	Conv2Layer::Conv2Layer(CudaNn* nn, int kernelSize, int nInChannel, int nOutChannel, int stride, int padding) :
		CudaLayer(nn), _kernelSize(kernelSize), _stride(stride), _padding(padding)
	{
		_w.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_w.SetRandom(1.0f / sqrtf(float(_kernelSize * _kernelSize * nInChannel * 0.5f))); // He initialization
		_wG.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_m.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_m.SetZero();
		_wG_v.ResetTensor(_kernelSize, _kernelSize, nInChannel, nOutChannel);
		_wG_v.SetZero();
		_b.ResetTensor(nOutChannel);
		_b.SetZero();
		_bG.ResetTensor(nOutChannel);
		_bG_m.ResetTensor(nOutChannel);
		_bG_m.SetZero();
		_bG_v.ResetTensor(nOutChannel);
		_bG_v.SetZero();
	}

	const CudaTensor* Conv2Layer::Forward(const CudaTensor* x)
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

	const CudaTensor* Conv2Layer::Backward(const CudaTensor* yG, const int layerIndex)
	{
		assert(_y._dataSize == yG->_dataSize);
		_xG.ResetTensor(_pX->_d0, _pX->_d1, _pX->_d2, _pX->_d3);

		{
			int nJobs = _wG._dataSize;
			int numBlocks = (nJobs + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 blocks(numBlocks), threads(K_THREAD_PER_BLOCK);
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

	void Conv2Layer::UpdateWs(float learningRate, float beta1, float beta2, float beta1t, float beta2t)
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

	void Conv2Layer::Pull()
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

	FcLayer::FcLayer(CudaNn* nn, int inDim, int outDim) : CudaLayer(nn), _pX(nullptr)
	{
		_w.ResetTensor(outDim, inDim);
		//_w.Random(1.0 / sqrtf(inDim)); // Xavier initialization
		_w.SetRandom(1.0f / sqrtf(inDim * 0.5f)); // He initialization
		_wG.ResetTensor(outDim, inDim);
		_wG_m.ResetTensor(outDim, inDim);
		_wG_m.SetZero();
		_wG_v.ResetTensor(outDim, inDim);
		_wG_v.SetZero();
		_b.ResetTensor(outDim);
		_b.SetZero();
		_bG.ResetTensor(outDim);
		_bG_m.ResetTensor(outDim);
		_bG_m.SetZero();
		_bG_v.ResetTensor(outDim);
		_bG_v.SetZero();
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
			BackwardFc_Wg_Cuda <<< block, threads >>> (_wG._dataGpu, _pX->_dataGpu, yG->_dataGpu, x_d0, x_d1, _wG._d0, nJobs);
			assert(cudaGetLastError() == cudaSuccess);
		}
		{
			int numBlocks = (_b._d0 + K_THREAD_PER_BLOCK - 1) / K_THREAD_PER_BLOCK;
			dim3 block(numBlocks), threads(K_THREAD_PER_BLOCK);
			BackwardFc_Bg_Cuda <<< block, threads >>> (_bG._dataGpu, yG->_dataGpu, yG->_d0, yG->_d1);
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
			BackwardFc_Xg_Cuda <<< block, threads >>> (_xG._dataGpu, yG->_dataGpu, _w._dataGpu, yG->_d0, _w._d0, _xG._d0, nJobs);
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

	const CudaTensor* DropoutLayer::Forward(const CudaTensor* x)
	{
		if (false == _nn->IsDropoutEnabled())
		{
			return x;
		}

		_crossCheck = 1;
		_dropoutMask.ResetTensor(x->_dataSize);
		_dropoutMask.SetDropoutMask(_dropoutRate);
		_xDropped.ResetTensor(x->_dataSize);

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

	bool CudaNn::AddConv2d(int kernelSize, int nInChannel, int nOutChannel, int stride, int padding)
	{
		_layers.push_back(new Conv2Layer(this, kernelSize, nInChannel, nOutChannel, stride, padding));
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