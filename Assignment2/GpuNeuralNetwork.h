#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <numeric> //std::iota

extern Eigen::MatrixXf cudaMatrixMul(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf cudaMatrixScalarMul(const Eigen::MatrixXf &M, float scalar);
extern Eigen::MatrixXf cudaMatrixAdd(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf cudaMatrixSub(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);

void printMatrixSize(const std::string msg, const Eigen::MatrixXf &m)
{
  std::cout << msg.c_str() << "[" << m.rows() << "," << m.cols() << "]" << std::endl;
}

class GPULayer
{
public:
  GPULayer() : input(), output() {}
  virtual ~GPULayer() {}

  virtual Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf &input) = 0;
  virtual Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf &output, float learningRate) = 0;

protected:
  Eigen::MatrixXf input;
  Eigen::MatrixXf output;
};

class GPUDenseLayer : public GPULayer
{
public:
  GPUDenseLayer(int inputSize, int outputSize)
  {
    // Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
    // Do the matrix multiplication on the GPU
    weights = cudaMatrixScalarMul(Eigen::MatrixXf::Random(inputSize, outputSize), 0.5f);
    bias = cudaMatrixScalarMul(Eigen::MatrixXf::Random(1, outputSize), 0.5f);
  }

  Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf &input)
  {
    this->input = input;
    // Do the matrix multiplication on the GPU
    this->output = cudaMatrixAdd(cudaMatrixMul(input, weights), bias);
    return this->output;
  }

  // computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.
  Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf &outputError, float learningRate)
  {
    // Do the matrix multiplication on the GPU
    Eigen::MatrixXf inputError = cudaMatrixMul(outputError, weights.transpose());
    Eigen::MatrixXf weightsError = cudaMatrixMul(input.transpose(), outputError);

    // update parameters on the GPU
    weights = cudaMatrixSub(weights, cudaMatrixScalarMul(weightsError, learningRate));
    bias = cudaMatrixSub(bias, cudaMatrixScalarMul(outputError, learningRate));

    return inputError;
  }

private:
  Eigen::MatrixXf weights;
  Eigen::MatrixXf bias;
};

class GPUActivationLayer : public GPULayer
{
public:
  GPUActivationLayer(std::function<float(float)> activation,
                     std::function<float(float)> activationPrime)
  {
    this->activation = activation;
    this->activationPrime = activationPrime;
  }

  // returns the activated input
  Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf &input)
  {
    this->input = input;
    this->output = input.unaryExpr(activation);
    return this->output;
  }

  // Returns inputRrror = dE / dX for a given output_error = dE / dY.
  // learningRate is not used because there is no "learnable" parameters.
  Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf &outputError, float learningRate)
  {
    return (input.unaryExpr(activationPrime).array() * outputError.array()).matrix();
  }

private:
  std::function<float(float)> activation;
  std::function<float(float)> activationPrime;
};

class GPUFlattenLayer : public GPULayer
{
public:
  Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf &input)
  {
    this->input = input;
    this->output = input;
    this->output.resize(1, input.rows() * input.cols()); // flatten
    return this->output;
  }
  Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf &outputError, float learningRate)
  {
    outputError.resize(input.rows(), input.cols());
    return outputError;
  }
};

class GPUNetwork
{
public:
  GPUNetwork() {}
  virtual ~GPUNetwork() {}

  void add(GPULayer *layer)
  {
    layers.push_back(layer);
  }

  void use(std::function<float(Eigen::MatrixXf &, Eigen::MatrixXf &)> lossF, std::function<Eigen::MatrixXf(Eigen::MatrixXf &, Eigen::MatrixXf &)> lossDer)
  {
    loss = lossF;
    lossPrime = lossDer;
  }

  std::vector<Eigen::MatrixXf> predict(Eigen::MatrixXf input)
  {
    int samples = input.rows();

    std::vector<Eigen::MatrixXf> result;

    // forward propagation
    for (int j = 0; j < samples; ++j)
    {
      Eigen::MatrixXf output = input.row(j);
      for (GPULayer *layer : layers)
        output = layer->forwardPropagation(output);

      result.push_back(output);
    }

    return result;
  }

  // train the network
  virtual void fit(Eigen::MatrixXf x_train, Eigen::MatrixXf y_train, int epochs, float learningRate)
  {
    int samples = x_train.rows();
    std::cout << "Samples: " << samples << std::endl;
    printMatrixSize("x_train", x_train);
    printMatrixSize("y_train", y_train);

    std::vector<int> order(samples);
    std::iota(order.begin(), order.end(), 0);

    // training loop
    for (int i = 0; i < epochs; ++i)
    {
      float err = 0.0f;

      // feed forward
      std::random_shuffle(order.begin(), order.end());

      // forward propagation
      for (int j = 0; j < samples; ++j)
      {
        int index = order[j];
        Eigen::MatrixXf output = x_train.row(index);

        for (GPULayer *layer : layers)
          output = layer->forwardPropagation(output);

        // compute loss(for display purpose only)
        Eigen::MatrixXf y = y_train.row(index);

        err += loss(y, output);

        // backward propagation
        Eigen::MatrixXf error = lossPrime(y, output);

        for (std::vector<GPULayer *>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer)
          error = (*layer)->backwardPropagation(error, learningRate);
      }
      err /= (float)samples;
      std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
    }
  }

protected:
  std::vector<GPULayer *> layers;
  std::function<float(Eigen::MatrixXf &, Eigen::MatrixXf &)> loss;
  std::function<Eigen::MatrixXf(Eigen::MatrixXf &, Eigen::MatrixXf &)> lossPrime;
};
#endif