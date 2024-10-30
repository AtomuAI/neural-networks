// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_MATH_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_MATH_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

#include "bewusstsein_neural_networks/c++/include/core/constants.hpp"

namespace nn
{
    template <typename T, typename U>
    constexpr T pow( const T base, const U exponent ) requires ( ( is_int<T>::value || is_float<T>::value ) && is_int<U>::value )
    {
        T x = base;
        U n = exponent;
        if ( exponent < 0 )
        {
            x = 1 / x;
            n = -n;
        }
        else if ( exponent == 0 )
        {
            return 1;
        }
        if ( base != 0 )
        {
            T y = 1;
            while ( n > 1 )
            {
                if ( n % 2 != 0 )
                {
                    y = x * y;
                    --n;
                }
                x = x * x;
                n = n / 2;
            }
            return x * y;
        }
        return 0;
    }

    template <typename T, typename U>
    constexpr T pow( const T base, const U exponent ) requires ( ( is_int<T>::value || is_float<T>::value ) && is_float<U>::value )
    {
        return std::pow( base, exponent );
    }

    template <typename T, typename U>
    constexpr T pow( const T base, const U exponent ) requires ( ( is_complex_int<T>::value || is_complex_float<T>::value ) && is_int<U>::value )
    {
        T x = base;
        U n = exponent;
        if ( exponent < 0 )
        {
            x = 1 / x;
            n = -n;
        }
        else if ( exponent == 0 )
        {
            return 1;
        }
        if ( base != std::complex<T>(0, 0) )
        {
            std::complex<T> y(1, 0);
            while ( n > 1 )
            {
                if ( n % 2 != 0 )
                {
                    y = x * y;
                    n = n - 1;
                }
                x = x * x;
                n = n / 2;
            }
            return x * y;
        }
        return std::complex<T>(0, 0);
    }

    template <typename T, typename U>
    constexpr T pow( const T base, U exponent ) requires ( ( is_complex_int<T>::value || is_complex_float<T>::value ) && is_float<U>::value )
    {
        return std::pow( base, exponent );
    }

    template <typename T, typename U>
    constexpr T pow( const T base, const U exponent ) requires ( ( is_complex_int<T>::value || is_complex_float<T>::value ) && ( is_complex_float<U>::value || is_complex_int<U>::value ) )
    {
        return std::pow( base, exponent );
    }

    template <typename T>
    constexpr T exp( const T exponent ) requires ( is_int<T>::value )
    {
        return pow( e<T>::value, exponent );
    }

    template <typename T>
    constexpr T exp( const T exponent ) requires ( is_float<T>::value )
    {
        return std::exp( exponent );
    }

    template <typename T>
    constexpr T exp( const T exponent ) requires ( is_complex_int<T>::value || is_complex_float<T>::value )
    {
        return std::exp( exponent );
    }

    template <typename U>
    constexpr U sin( const U angle ) requires ( is_float<U>::value )
    {
        return std::sin( angle );
    }

    template <typename U>
    constexpr U cos( const U angle ) requires ( is_float<U>::value )
    {
        return std::cos( angle );
    }

    template <typename U>
    constexpr U tan( const U angle ) requires ( is_float<U>::value )
    {
        return std::tan( angle );
    }

    template <typename U>
    constexpr U sec( const U angle ) requires ( is_float<U>::value )
    {
        return 1 / cos( angle );
    }

    template <typename U>
    constexpr U csc( const U angle ) requires ( is_float<U>::value )
    {
        return 1 / sin( angle );
    }

    template <typename U>
    constexpr U cot( const U angle ) requires ( is_float<U>::value )
    {
        return 1 / tan( angle );
    }

    template <typename U>
    constexpr U sinh( const U angle ) requires ( is_float<U>::value )
    {
        return std::sinh( angle );
    }

    template <typename U>
    constexpr U cosh( const U angle ) requires ( is_float<U>::value )
    {
        return std::cosh( angle );
    }

    template <typename U>
    constexpr U tanh( const U angle ) requires ( is_float<U>::value )
    {
        return std::tanh( angle );
    }

    template <typename T, typename U>
    constexpr T unit_step( const U x ) requires ( is_int<T>::value && ( is_int<U>::value || is_float<U>::value ) )
    {
        return ( x >= 0 ) ? 1 : 0;
    }

    template <typename T, typename U, typename V>
    constexpr T periodic_unit_step( const U x, const V period ) requires ( ( is_int<T>::value || is_float<T>::value ) && ( is_int<U>::value || is_float<U>::value ) )
    {
        return ( ( x % period ) >= ( period / 2 ) ) ? 1 : 0;
    }

    template <typename T>
    constexpr T sqrt( const T value ) requires ( is_int<T>::value )
    {
        T x = value;
        T y = 1;
        while(x > y)
        {
            x = ( x + y );
            y = value;
        }
        return x;
    }

    template <typename T>
    constexpr T sqrt( const T value ) requires ( is_float<T>::value )
    {
        return std::sqrt( value );
    }

    template <typename T>
    constexpr T sqrt( const T value ) requires ( is_complex_int<T>::value || is_complex_float<T>::value )
    {
        return std::sqrt( value.value );
    }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_MATH_HPP_
