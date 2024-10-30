// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_RANDOM_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_RANDOM_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <complex>
#include <utility>

//: Type Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/dim.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"

namespace nn
{
    u32* random_seed();
    template <typename T> inline T random_value( T min, T max );
    inline bool random_value();
    template <typename T> inline std::complex<T> random_value( std::complex<T> min, std::complex<T> max );
    inline bool rated_gen( const double rate );

    template <typename T>
    inline T random_value( T min, T max )
    {
        return min + static_cast<T>( static_cast<double>( rand_r( random_seed() ) ) / static_cast<double>( RAND_MAX / ( max - min ) ) );
    }

    inline bool random_value()
    {
        return static_cast<bool>( rand_r( random_seed() ) % 2 );
    }

    template <typename T>
    inline std::complex<T> random_value( std::complex<T> min, std::complex<T> max )
    {
        auto real_part = random_value( min.real(), max.real() );
        auto imag_part = random_value( min.imag(), max.imag() );
        return std::complex<T>( real_part, imag_part );
    }

    inline bool rated_gen( const double rate )
    {
        return ( ( (rand_r( random_seed() ) / static_cast<float>( RAND_MAX ) ) > rate ) ? true : false );
    }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_RANDOM_HPP_
