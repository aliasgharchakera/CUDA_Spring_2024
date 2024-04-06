#pragma once

#include "device_launch_parameters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace smallpt {

	constexpr double g_pi = 3.14159265358979323846;

	__host__ __device__ inline double Clamp(double v, 
											double low = 0.0, 
											double high = 1.0) noexcept {

		return fmin(fmax(v, low), high);
	}

	inline std::uint8_t ToByte(double color, double gamma = 2.2) noexcept {
		const double gcolor = std::pow(color, 1.0 / gamma);
		return static_cast< std::uint8_t >(Clamp(255.0 * gcolor, 0.0, 255.0));
	}
}