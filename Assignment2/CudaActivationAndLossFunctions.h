#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <Eigen/Dense>

extern Eigen::MatrixXf cudaMatrixMul(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf cudaMatrixScalarMul(const Eigen::MatrixXf &M, float scalar);
extern Eigen::MatrixXf cudaMatrixAdd(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf cudaMatrixSub(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);

//activation functions
float sigmoid(float x)
{
	return 1.0f / 1.0f + exp(-x);
}

float sigmoid_prime(float x)
{
	float s = sigmoid(x);
	return s * (1 - s);
}
float tanh2(float x)
{
	return tanh(x);
}

float tanh_prime(float x)
{
	return 1.0f - powf(tanh(x), 2.0f);
}

float relu(float x)
{
	return std::max(x, 0.0f);
}
float relu_prime(float x)
{
	return (float)((int)(x >= 0));
}

float one_minus(float x)
{
	return 1 - x;
}
//loss function and their derivative
float mse(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
	return cudaMatrixMul(cudaMatrixSub(y_true, y_pred), cudaMatrixSub(y_true, y_pred).transpose()).mean();
}

Eigen::MatrixXf mse_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
  return cudaMatrixScalarMul(cudaMatrixScalarMul(cudaMatrixSub(y_pred, y_true), 2), 1 / (y_true.rows() * y_true.cols()));
	// return  2 * (y_pred - y_true) / (y_true.rows()*y_true.cols());
}

float binary_cross_entropy(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
  return cudaMatrixMul(cudaMatrixSub(cudaMatrixScalarMul(y_true, -1), cudaMatrixMul(y_pred, cudaMatrixScalarMul(y_true, -1)).log()), cudaMatrixSub(cudaMatrixScalarMul(cudaMatrixScalarMul(y_true, -1), cudaMatrixMul(y_pred, cudaMatrixScalarMul(y_true, -1)).log()), cudaMatrixScalarMul(cudaMatrixScalarMul(y_true, -1), cudaMatrixMul(y_pred, cudaMatrixScalarMul(y_true, -1)).log())).transpose()).mean();
}

#endif
