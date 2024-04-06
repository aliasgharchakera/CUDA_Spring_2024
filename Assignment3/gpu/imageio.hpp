#pragma once

#include "vector.hpp"

#include <fstream>

namespace smallpt {

	inline void WritePPM(std::uint32_t w,
						 std::uint32_t h,
						 const Vector3* Ls,
						 const char* fname = "gpu-image.ppm") noexcept {

		std::ofstream file(fname);
		if (!file.is_open()) {
			// Handle file open error
			return;
		}

		file << "P3\n" << w << " " << h << "\n255\n";
		for (std::size_t i = 0; i < w * h; ++i) {
			file << static_cast<int>(ToByte(Ls[i].m_x)) << " "
				 << static_cast<int>(ToByte(Ls[i].m_y)) << " "
				 << static_cast<int>(ToByte(Ls[i].m_z)) << " ";
		}

		file.close();
	}
}
