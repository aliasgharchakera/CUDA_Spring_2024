#pragma once

#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>

namespace smallpt {

	inline void cudaCheckError(cudaError_t err, const char* file, int line) {
		if (cudaSuccess != err) {
			std::printf("%s in %s at line %d\n", 
						cudaGetErrorString(err), file, line);
			std::exit(EXIT_FAILURE);
		}
	}
}

#define cudaCheckErr(err) (cudaCheckError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL(a) {if (a == NULL) { \
	std::printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); \
    std::exit( EXIT_FAILURE );}}
