#pragma once

#include "vector.hpp"

namespace smallpt {

	struct Ray {

	public:
		__device__ explicit Ray(Vector3 o, 
								Vector3 d, 
								double tmin = 0.0, 
								double tmax = std::numeric_limits< double >::infinity(), 
								std::uint32_t depth = 0u) noexcept
			: m_o(std::move(o)),
			m_d(std::move(d)),
			m_tmin(tmin),
			m_tmax(tmax),
			m_depth(depth) {};
		Ray(const Ray& ray) noexcept = default;
		Ray(Ray&& ray) noexcept = default;
		~Ray() = default;

		Ray& operator=(const Ray& ray) = default;
		Ray& operator=(Ray&& ray) = default;

		__device__ const Vector3 operator()(double t) const {
			return m_o + m_d * t;
		}

		Vector3 m_o, m_d;
		mutable double m_tmin, m_tmax;
		std::uint32_t m_depth;
	};
}