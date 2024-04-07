#include "imageio.hpp"
#include "sampling.cuh"
#include "specular.cuh"
#include "sphere.hpp"
#include "cuda_tools.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

#define REFRACTIVE_INDEX_OUT 1.0
#define REFRACTIVE_INDEX_IN  1.5
#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

namespace smallpt {

	const Sphere g_spheres[] = {
		Sphere(1e5,  Vector3(1e5 + 1, 40.8, 81.6),   Vector3(),   Vector3(0.75,0.25,0.25), Reflection_t::Diffuse),	 //Left
		Sphere(1e5,  Vector3(-1e5 + 99, 40.8, 81.6), Vector3(),   Vector3(0.25,0.25,0.75), Reflection_t::Diffuse),	 //Right
		Sphere(1e5,  Vector3(50, 40.8, 1e5),         Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),	 //Back
		Sphere(1e5,  Vector3(50, 40.8, -1e5 + 170),  Vector3(),   Vector3(),               Reflection_t::Diffuse),	 //Front
		Sphere(1e5,  Vector3(50, 1e5, 81.6),         Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),	 //Bottom
		Sphere(1e5,  Vector3(50, -1e5 + 81.6, 81.6), Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),	 //Top
		Sphere(16.5, Vector3(27, 16.5, 47),          Vector3(),   Vector3(0.999),          Reflection_t::Specular),	 //Mirror
		Sphere(16.5, Vector3(73, 16.5, 78),          Vector3(),   Vector3(0.999),          Reflection_t::Refractive),//Glass
		Sphere(600,	 Vector3(50, 681.6 - .27, 81.6), Vector3(12), Vector3(),               Reflection_t::Diffuse)	 //Light
	};

//   const Sphere g_spheres[] = {
//     // Room definition (large spheres for walls, floor, ceiling)
//     // Sphere(1e5,  Vector3(1e5 + 1, 0, 0),    Vector3(), Vector3(0.75), Reflection_t::Diffuse), // Left wall
//     // Sphere(1e5,  Vector3(-1e5 + 99, 0, 0),  Vector3(), Vector3(0.75), Reflection_t::Diffuse), // Right wall
//     Sphere(1e5,  Vector3(50, 0, 1e5),       Vector3(), Vector3(0.85), Reflection_t::Diffuse), // Back wall
//     // Sphere(1e5,  Vector3(50, 0, -1e5),      Vector3(), Vector3(0.75), Reflection_t::Diffuse), // Front
//     // Sphere(1e5,  Vector3(50, 1e5, 0),       Vector3(), Vector3(0.75), Reflection_t::Diffuse), // Floor
//     // Sphere(1e5,  Vector3(50, -1e5, 0),      Vector3(), Vector3(0.75), Reflection_t::Diffuse), // Ceiling

//     // Spheres in the pyramid
//     Sphere(10, Vector3(33, 53, 70),         Vector3(), Vector3(1,0.6,0.4), Reflection_t::Diffuse), // Bottom-left (orange)
//     Sphere(10, Vector3(33, 53, 40),         Vector3(), Vector3(0.999), Reflection_t::Specular), // Bottom-middle (mirror)
//     Sphere(10, Vector3(67, 53, 70),         Vector3(), Vector3(0.3,0.5,0.8), Reflection_t::Diffuse), // Bottom-middle (blue)
//     Sphere(10, Vector3(67, 53, 40),         Vector3(), Vector3(0.35,0.7,0.41), Reflection_t::Diffuse), // Bottom-right (green)

//     Sphere(10, Vector3(33, 27, 70),         Vector3(), Vector3(0.35,0.7,0.41), Reflection_t::Diffuse), // Bottom-middle (green)
//     Sphere(10, Vector3(33, 27, 40),         Vector3(), Vector3(0.3,0.5,0.8), Reflection_t::Diffuse), // Bottom-right (blue)
//     Sphere(10, Vector3(67, 27, 70),         Vector3(), Vector3(0.999), Reflection_t::Specular), // Bottom-right (mirror)
//     Sphere(10, Vector3(67, 27, 40),         Vector3(), Vector3(1,0.6,0.4), Reflection_t::Diffuse), // Bottom-left (orange)

//     Sphere(10, Vector3(50, 40, 55),         Vector3(12), Vector3(), Reflection_t::Diffuse), // center (light)

//     // Ambient light (a large, very dimly emitting sphere overhead)
//     Sphere(1e5, Vector3(50, -1e5 + 150, 0), Vector3(0.1), Vector3(), Reflection_t::Diffuse) // Light
//   };


	__device__ inline bool Intersect(const Sphere* dev_spheres, std::size_t nb_spheres, const Ray& ray, size_t& id) {

		bool hit = false;
		for (std::size_t i = 0u; i < nb_spheres; ++i) {
			if (dev_spheres[i].Intersect(ray)) {
				hit = true;
				id  = i;
			}
		}

		return hit;
	}

	__device__ static Vector3 Radiance(const Sphere* dev_spheres, std::size_t nb_spheres, const Ray& ray, curandState* state) {

		Ray r = ray;
		Vector3 L;
		Vector3 F(1.0);

		while (true) {

			std::size_t id;

			if (!Intersect(dev_spheres, nb_spheres, r, id)) {
				return L;
			}

			const Sphere& shape = dev_spheres[id];
			const Vector3 p = r(r.m_tmax);
			const Vector3 n = Normalize(p - shape.m_p);

			L += F * shape.m_e;
			F *= shape.m_f;

			// Russian roulette
			if (4 < r.m_depth) {
				const double continue_probability = shape.m_f.Max();
				if (curand_uniform_double(state) >= continue_probability) {
					return L;
				}
				F /= continue_probability;
			}

			// Next path segment
			switch (shape.m_reflection_t) {
			
			case Reflection_t::Specular: {
				const Vector3 d = IdealSpecularReflect(r.m_d, n);
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
				break;
			}
			
			case Reflection_t::Refractive: {
				double pr;
				const Vector3 d = IdealSpecularTransmit(r.m_d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, pr, state);
				F *= pr;
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
				break;
			}
			
			default: {
				const Vector3 w = (0.0 > n.Dot(r.m_d)) ? n : -n;
				const Vector3 u = Normalize((abs(w.m_x) > 0.1 ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0)).Cross(w));
				const Vector3 v = w.Cross(u);

				const Vector3 sample_d = CosineWeightedSampleOnHemisphere(curand_uniform_double(state), curand_uniform_double(state));
				const Vector3 d = Normalize(sample_d.m_x * u + sample_d.m_y * v + sample_d.m_z * w);
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
			}
			}
		}
	}

	__global__ static void kernel(const Sphere* dev_spheres, std::size_t nb_spheres, std::uint32_t w, std::uint32_t h, Vector3* Ls, std::uint32_t nb_samples) {
		
		const std::uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		const std::uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		const std::uint32_t offset = x + y * blockDim.x * gridDim.x;

		if (x >= w || y >= h) {
			return;
		}

		// RNG
		curandState state;
		curand_init(offset, 0u, 0u, &state);

		const Vector3 eye = { 50.0, 52.0, 295.6 };
    	// const Vector3 eye = { 50.0, 72.0, 295.6 };
		const Vector3 gaze = Normalize(Vector3(0.0, -0.042612, -1.0));
		const double fov = 0.5135;
		const Vector3 cx = { w * fov / h, 0.0, 0.0 };
		const Vector3 cy = Normalize(cx.Cross(gaze)) * fov;

		for (std::size_t sy = 0u, i = (h - 1u - y) * w + x; sy < 2u; ++sy) { // 2 subpixel row

			for (std::size_t sx = 0u; sx < 2u; ++sx) { // 2 subpixel column

				Vector3 L;

				for (std::size_t s = 0u; s < nb_samples; ++s) { // samples per subpixel
					const double u1 = 2.0 * curand_uniform_double(&state);
					const double u2 = 2.0 * curand_uniform_double(&state);
					const double dx = (u1 < 1.0) ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
					const double dy = (u2 < 1.0) ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
					const Vector3 d = cx * (((sx + 0.5 + dx) * 0.5 + x) / w - 0.5) +
						              cy * (((sy + 0.5 + dy) * 0.5 + y) / h - 0.5) + gaze;
					
					L += Radiance(dev_spheres, nb_spheres, 
						Ray(eye + d * 130, Normalize(d), EPSILON_SPHERE), &state) 
						* (1.0 / nb_samples);
				}
				
				Ls[i] += 0.25 * Clamp(L);
			}
		}
	}

	static void Render(std::uint32_t nb_samples) noexcept {

		const std::uint32_t w = 1024u;
		const std::uint32_t h = 768u;
		const std::uint32_t nb_pixels = w * h;

		// Set up device memory
		//cudaCheckErr( cudaMemcpyToSymbol(dev_spheres, spheres, sizeof(spheres)) );
		Sphere* dev_spheres;
		cudaCheckErr(cudaMalloc((void**)&dev_spheres, sizeof(g_spheres)));
		cudaCheckErr(cudaMemcpy(dev_spheres, g_spheres, sizeof(g_spheres), cudaMemcpyHostToDevice));
		Vector3* dev_Ls;
		cudaCheckErr(cudaMalloc((void**)&dev_Ls, nb_pixels * sizeof(Vector3)));
		cudaCheckErr(cudaMemset(dev_Ls, 0, nb_pixels * sizeof(Vector3)));

		// Kernel execution
		const dim3 nblocks(w / 16u, h / 16u);
		const dim3 nthreads(16u, 16u);
		kernel<<< nblocks, nthreads >>>(dev_spheres, ARRAYSIZE(g_spheres), w, h, dev_Ls, nb_samples);

		// Set up host memory
		Vector3* Ls = (Vector3*)malloc(nb_pixels * sizeof(Vector3));
		// Transfer device -> host
		cudaCheckErr(cudaMemcpy(Ls, dev_Ls, nb_pixels * sizeof(Vector3), cudaMemcpyDeviceToHost));

		// Clean up device memory
		cudaCheckErr(cudaFree(dev_Ls));
		cudaCheckErr(cudaFree(dev_spheres));

		WritePPM(w, h, Ls);

		// Clean up host memory
		free(Ls);
	}
}

int main(int argc, char* argv[]) {
	
  std::uint32_t nb_samples = 1;

  if (argc >= 2) {
      nb_samples = atoi(argv[1]) / 4;
  }


	bool all_flag = false;
    if (argc >= 3 && strcmp(argv[2], "--all") == 0) {
        all_flag = true;
    }

	// print the number of samples
	std::cout << "Number of samples: " << nb_samples << std::endl;
	
	// Store the results in a text file
	std::ofstream outputFile("gpu_timing.txt");
	if (!outputFile.is_open()) {
		std::cerr << "Failed to open the output file." << std::endl;
		return 1;
	}

	// Run the code for different nb_samples and time it
	if (all_flag) {
		// std::cout << "Running for all samples" << std::endl;
		for (std::uint32_t samples = 1; samples <= nb_samples; samples *= 2) {
      int end = (samples==1) ? 4 : 3;
			for (int i = 0; i < end; i++) {
				auto start = std::chrono::high_resolution_clock::now();
				smallpt::Render(samples);
				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

				// Store the timing results in the text file
				outputFile << "nb_samples: " << samples << ", time: " << duration << " ms" << std::endl;
			}
		}
	}
	else {
		// std::cout << "Running for one sample" << std::endl;
		for (int i = 0; i < 5; i++) {
			auto start = std::chrono::high_resolution_clock::now();
			smallpt::Render(nb_samples);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			// Store the timing results in the text file
			outputFile << "nb_samples: " << nb_samples << ", time: " << duration << " ms" << std::endl;
		}
	}

	outputFile.close();

	return 0;
}