#include <iostream>
#include <vector>
#include "GpuNeuralNetwork.h"
#include "ActivationAndLossFunctions.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include "GpuNeuralNetwork.h"
#include "ActivationAndLossFunctions.h"

void timeAndStoreOperations(const std::string& filePath)
{
	std::ofstream outputFile(filePath, std::ios::app);

	std::cout << "Using Eigen ver: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

	// test the XOR solver
	Eigen::MatrixXf x_train{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	Eigen::MatrixXf y_train{{0}, {1}, {1}, {0}};

	GPUNetwork nn;
	nn.add(new GPUDenseLayer(2, 3));
	nn.add(new GPUActivationLayer(tanh2, tanh_prime));
	nn.add(new GPUDenseLayer(3, 1));
	nn.add(new GPUActivationLayer(tanh2, tanh_prime));

	nn.use(mse, mse_prime);

	// Measure execution time
	auto startTime = std::chrono::high_resolution_clock::now();

	// train
	nn.fit(x_train, y_train, 1000, 0.1f);

	// test
	std::vector<Eigen::MatrixXf> output = nn.predict(x_train);
	for (Eigen::MatrixXf out : output)
		std::cout << out << std::endl;

	// End timing
	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

	outputFile << duration << std::endl;

	outputFile.close();
}

int main()
{
	for (int i = 0; i < 3; i++)
		timeAndStoreOperations("gpu_xor.txt");
	return 0;
}