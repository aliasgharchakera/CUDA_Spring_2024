#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

__global__ void cudaMatrixMulKernel(float *M, float *N, float *P, int rows,
                                    int cols, int common)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols))
    {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < common; ++k)
        {
            Pvalue += M[k * rows + row] * N[col * common + k];
        }
        P[col * rows + row] = Pvalue;
    }
}

__global__ void cudaMatrixScalarMulKernel(float *M, float N, float *P, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols))
    {
        P[col * rows + row] = M[col * rows + row] * N;
    }
}

Eigen::MatrixXf cudaMatrixMul(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = N.cols();
    if (M.cols() != N.rows())
    {
        std::cout << M.rows() << "," << M.cols() << std::endl;
        std::cout << N.rows() << "," << N.cols() << std::endl;
        std::cout << "Matrix dimensions are not compatible for multiplication" << std::endl;
        return Eigen::MatrixXf::Zero(1, 1);
    }
    int common = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * common * sizeof(float);
    int size_N = common * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);

    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_N);
    cudaMalloc((void **)&d_P, size_P);
    checkCudaErr(cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice), "Memcpy M");
    checkCudaErr(cudaMemcpy(d_N, N.data(), size_N, cudaMemcpyHostToDevice), "Memcpy N");

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    cudaMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols, common);
    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
    checkCudaErr(cudaGetLastError(), "GPU Error");

    Eigen::MatrixXf P(rows, cols);
    checkCudaErr(cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost), "Memcpy P");
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}

Eigen::MatrixXf cudaMatrixScalarMul(const Eigen::MatrixXf &M, float N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_P, size_P);
    checkCudaErr(cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice), "Memcpy M");

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    cudaMatrixScalarMulKernel<<<dimGrid, dimBlock>>>(d_M, N, d_P, rows, cols);
    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
    checkCudaErr(cudaGetLastError(), "GPU Error");

    Eigen::MatrixXf P(rows, cols);
    checkCudaErr(cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost), "Memcpy P");
    cudaFree(d_M);
    cudaFree(d_P);

    return P;
}

__global__ void cudaMatrixAddKernel(float *M, float *N, float *P, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols))
    {
        P[col * rows + row] = M[col * rows + row] + N[col * rows + row];
    }
}

Eigen::MatrixXf cudaMatrixAdd(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_M);
    cudaMalloc((void **)&d_P, size_P);
    checkCudaErr(cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice), "Memcpy M");
    checkCudaErr(cudaMemcpy(d_N, N.data(), size_M, cudaMemcpyHostToDevice), "Memcpy N");

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    cudaMatrixAddKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols);
    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
    checkCudaErr(cudaGetLastError(), "GPU Error");

    Eigen::MatrixXf P(rows, cols);
    checkCudaErr(cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost), "Memcpy P");
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}

__global__ void cudaMatrixSubKernel(float *M, float *N, float *P, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < rows) && (col < cols))
    {
        P[col * rows + row] = M[col * rows + row] - N[col * rows + row];
    }
}

Eigen::MatrixXf cudaMatrixSub(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_M);
    cudaMalloc((void **)&d_P, size_P);
    checkCudaErr(cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice), "Memcpy M");
    checkCudaErr(cudaMemcpy(d_N, N.data(), size_M, cudaMemcpyHostToDevice), "Memcpy N");

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    cudaMatrixSubKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols);
    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
    checkCudaErr(cudaGetLastError(), "GPU Error");

    Eigen::MatrixXf P(rows, cols);
    checkCudaErr(cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost), "Memcpy P");
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}