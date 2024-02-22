#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <Eigen/Dense>

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
	auto diff = (y_true - y_pred ).array() ;
	return  ( diff * diff).mean();
	//return ((y_true - y_pred) * (y_true - y_pred)).mean();
}

Eigen::MatrixXf mse_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  2 * (y_pred - y_true) / (y_true.rows()*y_true.cols());
}

/*
float binary_cross_entropy(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return  (-y_true * y_pred.log()).mean() - (y_true.unaryExpr(one_minus)) * (y_pred.unaryExpr(one_minus)).log());
}

Eigen::MatrixXf binary_cross_entropy_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
	return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size();
}
*/
#endif
