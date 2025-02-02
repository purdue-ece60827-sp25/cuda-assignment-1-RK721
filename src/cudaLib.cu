
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int lThreadID = blockIdx.x*blockDim.x + threadIdx.x;
	if (lThreadID < size)
	{
    	y[lThreadID] = scale*x[lThreadID] + y[lThreadID];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	srand(time(NULL));
	float* hX = new float[vectorSize];
	float* hY = new float[vectorSize];
	float* hYCopy = new float[vectorSize];
	float a = (float)(rand() % 100);
	float* dX;
	float* dY;

	vectorInit(hX, vectorSize);
	vectorInit(hY, vectorSize);
	std::memcpy(hYCopy, hY, vectorSize * sizeof(float));

    cudaMalloc(&dX, vectorSize*sizeof(float));
    cudaMalloc(&dY, vectorSize*sizeof(float));

    cudaMemcpy(dX, hX, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", a);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", hX[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", hY[i]);
		}
		printf(" ... }\n");
	#endif

	saxpy_gpu<<<(vectorSize+255)/256, 256>>>(dX, dY, a, vectorSize);

	cudaMemcpy(hY, dY, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", hY[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(hX, hYCopy, hY, a, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(dX);
    cudaFree(dY);

	delete[] hX;
	delete[] hY;

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int lThreadID = blockIdx.x*blockDim.x + threadIdx.x;
	if (lThreadID < pSumSize)
	{
		curandState_t rng;
		curand_init(clock64(), lThreadID, 0, &rng);

		uint64_t lHitCount = 0;

		for (uint64_t i = 0; i < sampleSize; i++)
		{
			float x = curand_uniform(&rng);
			float y = curand_uniform(&rng);

			if ((x*x + y*y) <= 1.0f)
			{
				lHitCount++;
			}
		}

    	pSums[lThreadID] = lHitCount;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int lThreadID = blockIdx.x*blockDim.x + threadIdx.x;

	uint64_t lSum = 0;

	if (lThreadID < (pSumSize/reduceSize))
	{
		for (int i = lThreadID*reduceSize; (i < lThreadID*reduceSize + reduceSize) && (i < pSumSize); i++)
		{
			lSum += pSums[i];
		}

		totals[lThreadID] = lSum;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//uint64_t* h_pHitCount = new uint64_t[generateThreadCount];
	uint64_t* h_pReducedTotals = new uint64_t[reduceThreadCount];

	uint64_t* d_pHitCount;
	uint64_t* d_pReducedTotals;

	cudaMalloc(&d_pHitCount, generateThreadCount*sizeof(uint64_t));
    cudaMalloc(&d_pReducedTotals, reduceThreadCount*sizeof(uint64_t));

    //cudaMemcpy(d_pHitCount, h_pHitCount, generateThreadCount*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pReducedTotals, h_pReducedTotals, reduceThreadCount*sizeof(uint64_t), cudaMemcpyHostToDevice);

	generatePoints<<<(generateThreadCount+255)/256, 256>>>(d_pHitCount, generateThreadCount, sampleSize);

	if(reduceThreadCount < 256)
	{
		reduceCounts<<<1, reduceThreadCount>>>(d_pHitCount, d_pReducedTotals, generateThreadCount, reduceSize);
	}
	else
	{
		reduceCounts<<<(reduceThreadCount+255)/256, 256>>>(d_pHitCount, d_pReducedTotals, generateThreadCount, reduceSize);
	}

	cudaMemcpy(h_pReducedTotals, d_pReducedTotals, reduceThreadCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t lTotalHits = 0;
	for (uint64_t i = 0; i < reduceThreadCount; i++)
	{
		lTotalHits += h_pReducedTotals[i];
	}

	uint64_t lTotalSamples = generateThreadCount * sampleSize;
	approxPi = 4.0 * double(lTotalHits) / (double)lTotalSamples;

	cudaFree(d_pHitCount);
    cudaFree(d_pReducedTotals);

	//delete[] h_pHitCount;
	delete[] h_pReducedTotals;

	return approxPi;
}
