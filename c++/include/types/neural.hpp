// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_HPP_

//#define NEURAL_INT8
//#define NEURAL_INT16
#define NEURAL_INT32
//#define NEURAL_INT64
//#define NEURAL_INT128
//#define NEURAL_UINT8
//#define NEURAL_UINT16
//#define NEURAL_UINT32
//#define NEURAL_UINT64
//#define NEURAL_UINT128
//#define NEURAL_FLOAT16
#define NEURAL_FLOAT32
//#define NEURAL_FLOAT64
//#define NEURAL_FLOAT128
//#define NEURAL_COMPLEX_INT8
//#define NEURAL_COMPLEX_INT16
//#define NEURAL_COMPLEX_INT32
//#define NEURAL_COMPLEX_INT64
//#define NEURAL_COMPLEX_INT128
//#define NEURAL_COMPLEX_UINT8
//#define NEURAL_COMPLEX_UINT16
//#define NEURAL_COMPLEX_UINT32
//#define NEURAL_COMPLEX_UINT64
//#define NEURAL_COMPLEX_UINT128
//#define NEURAL_COMPLEX_FLOAT16
#define NEURAL_COMPLEX_FLOAT32
//#define NEURAL_COMPLEX_FLOAT64
//#define NEURAL_COMPLEX_FLOAT128
//#define NEURAL_BOOL

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

namespace nn
{
    template <typename T>
    class Neural
    {
        private:
            T value;

        public:
                                    constexpr Neural();
            template <typename U>   constexpr Neural( U value );

        public:
            template <typename U>   constexpr               void            operator=   ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value );
            template <typename U>   constexpr               void            operator=   ( const Complex<Neural<U>> other )  requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value );
            template <typename U>   constexpr               Neural<T>       operator+   ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator-   ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator*   ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator/   ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator+=  ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator-=  ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator*=  ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator/=  ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               bool            operator==  ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value );
            template <typename U>   constexpr               bool            operator!=  ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value );
            template <typename U>   constexpr               Neural<T>       operator<<  ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator<<= ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator>>  ( const Neural<U> other ) const     requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator>>= ( const Neural<U> other )           requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value );
                                    constexpr               Neural<T>       operator-   () const                            requires ( is_int<T>::value || is_float<T>::value );
                                    constexpr               Neural<T>       operator+   () const                            requires ( is_int<T>::value || is_float<T>::value );
                                    constexpr               Neural<T>       operator!   () const                            requires ( is_int<T>::value || is_bool<T>::value );
            template <typename U>   constexpr               bool            operator||  ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>       operator%   ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               bool            operator&&  ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator%=  ( const Neural<U> other )           requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>       operator&   ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator&=  ( const Neural<U> other )           requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>       operator|   ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator|=  ( const Neural<U> other )           requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>       operator^   ( const Neural<U> other ) const     requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator^=  ( const Neural<U> other )           requires ( is_int<T>::value && is_int<U>::value );
            template <typename U>   constexpr               Neural<T>       operator~   () const                            requires ( is_int<T>::value && is_int<U>::value );
                                    constexpr               Neural<T>&      operator++  ()                                  requires ( is_int<T>::value );
                                    constexpr               Neural<T>       operator++  ( T )                               requires ( is_int<T>::value );
                                    constexpr               Neural<T>&      operator--  ()                                  requires ( is_int<T>::value );
                                    constexpr               Neural<T>       operator--  ( T )                               requires ( is_int<T>::value );
            template <typename U>   constexpr               Neural<T>       operator+   ( const Neural<U> other ) const     requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator-   ( const Neural<U> other ) const     requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator*   ( const Neural<U> other ) const     requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>       operator/   ( const Neural<U> other ) const     requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator+=  ( const Neural<U> other )           requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator-=  ( const Neural<U> other )           requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator*=  ( const Neural<U> other )           requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
            template <typename U>   constexpr               Neural<T>&      operator/=  ( const Neural<U> other )           requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value );
                                    constexpr               operator        T           () const;
    };

    template <typename T>
    constexpr Neural<T>::Neural() : value( false ) {}

    template <typename T>
    template <typename U>
    constexpr Neural<T>::Neural( U value ) : value( value ) {}

    template <typename T>
    template <typename U>
    constexpr void Neural<T>::operator=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value )
    {
        this->value = other.value;
    }

    template <typename T>
    template <typename U>
    constexpr void Neural<T>::operator=( const Complex<Neural<U>> other ) requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value )
    {
        this->value = other.value.real();
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator+( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value + other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator-( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value - other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator*( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value * other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator/( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value / other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator+=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value += other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator-=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value -= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator*=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value *= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator/=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value /= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr bool Neural<T>::operator==( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value )
    {
        return this->value == other.value;
    }

    template <typename T>
    template <typename U>
    constexpr bool Neural<T>::operator!=( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value || is_bool<T>::value && is_int<U>::value || is_float<U>::value || is_bool<U>::value )
    {
        return this->value != other.value;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator<<( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value << other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator<<=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value <<= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator>>( const Neural<U> other ) const requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value >> other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator>>=( const Neural<U> other ) requires ( is_int<T>::value || is_float<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value >>= other.value;
        return *this;
    }

    template <typename T>
    constexpr Neural<T> Neural<T>::operator-() const requires ( is_int<T>::value || is_float<T>::value )
    {
        return Neural<T>( -this->value );
    }

    template <typename T>
    constexpr Neural<T> Neural<T>::operator+() const requires ( is_int<T>::value || is_float<T>::value )
    {
        return Neural<T>( +this->value );
    }

    template <typename T>
    constexpr Neural<T> Neural<T>::operator!() const requires ( is_int<T>::value || is_bool<T>::value )
    {
        return Neural<T>( !this->value );
    }

    template <typename T>
    template <typename U>
    constexpr bool Neural<T>::operator||( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return this->value || other.value;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator%( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return Neural<T>( this->value % other.value );
    }

    template <typename T>
    template <typename U>
    constexpr bool Neural<T>::operator&&( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return this->value && other.value;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator%=( const Neural<U> other ) requires ( is_int<T>::value && is_int<U>::value )
    {
        this->value %= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator&( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return Neural<T>( this->value & other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator&=( const Neural<U> other ) requires ( is_int<T>::value && is_int<U>::value )
    {
        this->value &= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator|( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return Neural<T>( this->value | other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator|=( const Neural<U> other ) requires ( is_int<T>::value && is_int<U>::value )
    {
        this->value |= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator^( const Neural<U> other ) const requires ( is_int<T>::value && is_int<U>::value )
    {
        return Neural<T>( this->value ^ other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator^=( const Neural<U> other ) requires ( is_int<T>::value && is_int<U>::value )
    {
        this->value ^= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator~() const requires ( is_int<T>::value && is_int<U>::value )
    {
        return Neural<T>( ~this->value );
    }

    template <typename T>
    constexpr Neural<T>& Neural<T>::operator++() requires ( is_int<T>::value )
    {
        ++this->value;
        return *this;
    }

    template <typename T>
    constexpr Neural<T> Neural<T>::operator++( T ) requires ( is_int<T>::value )
    {
        return Neural<T>( this->value++ );
    }

    template <typename T>
    constexpr Neural<T>& Neural<T>::operator--() requires ( is_int<T>::value )
    {
        --this->value;
        return *this;
    }

    template <typename T>
    constexpr Neural<T> Neural<T>::operator--( T ) requires ( is_int<T>::value )
    {
        return Neural<T>( this->value-- );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator+( const Neural<U> other ) const requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value || other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator-( const Neural<U> other ) const requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value || !other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator*( const Neural<U> other ) const requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value && other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T> Neural<T>::operator/( const Neural<U> other ) const requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        return Neural<T>( this->value && !other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator+=( const Neural<U> other ) requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value |= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator-=( const Neural<U> other ) requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value |= !other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator*=( const Neural<U> other ) requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value &= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Neural<T>& Neural<T>::operator/=( const Neural<U> other ) requires ( is_bool<T>::value && is_int<U>::value || is_float<U>::value )
    {
        this->value &= !other.value;
        return *this;
    }

    template <typename T>
    constexpr Neural<T>::operator T() const
    {
        return this->value;
    }

    #ifdef INT8
        #ifdef NEURAL_INT8
            typedef Neural<i8> ni8;
            template <typename T>
            struct is_ni8
            {
                static const bool value = false;
            };
            template <>
            struct is_ni8<ni8>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni8>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT8
    #endif // INT8
    #ifdef INT16
        #ifdef NEURAL_INT16
            typedef Neural<i16> ni16;
            template <typename T>
            struct is_ni16
            {
                static const bool value = false;
            };
            template <>
            struct is_ni16<ni16>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT16
    #endif // INT16
    #ifdef INT32
        #ifdef NEURAL_INT32
            typedef Neural<i32> ni32;
            template <typename T>
            struct is_ni32
            {
                static const bool value = false;
            };
            template <>
            struct is_ni32<ni32>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT32
    #endif // INT32
    #ifdef INT64
        #ifdef NEURAL_INT64
            typedef Neural<i64> ni64;
            template <typename T>
            struct is_ni64
            {
                static const bool value = false;
            };
            template <>
            struct is_ni64<ni64>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT64
    #endif // INT64
    #ifdef INT128
        #ifdef NEURAL_INT128
            typedef Neural<i128> ni128;
            template <typename T>
            struct is_ni128
            {
                static const bool value = false;
            };
            template <>
            struct is_ni128<ni128>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT128
    #endif // INT128
    #ifdef UINT8
        #ifdef NEURAL_UINT8
            typedef Neural<i8> ni8;
            template <typename T>
            struct is_ni8
            {
                static const bool value = false;
            };
            template <>
            struct is_ni8<ni8>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni8>
            {
                static const bool value = true;
            };
        #endif // NEURAL_UINT8
    #endif // UINT8
    #ifdef UINT16
        #ifdef NEURAL_UINT16
            typedef Neural<i16> ni16;
            template <typename T>
            struct is_ni16
            {
                static const bool value = false;
            };
            template <>
            struct is_ni16<ni16>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_UINT16
    #endif // UINT16
    #ifdef UINT32
        #ifdef NEURAL_UINT32
            typedef Neural<i32> ni32;
            template <typename T>
            struct is_ni32
            {
                static const bool value = false;
            };
            template <>
            struct is_ni32<ni32>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_UINT32
    #endif // UINT32
    #ifdef UINT64
        #ifdef NEURAL_UINT64
            typedef Neural<i64> ni64;
            template <typename T>
            struct is_ni64
            {
                static const bool value = false;
            };
            template <>
            struct is_ni64<ni64>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_UINT64
    #endif // UINT64
    #ifdef UINT128
        #ifdef NEURAL_UINT128
            typedef Neural<i128> ni128;
            template <typename T>
            struct is_ni128
            {
                static const bool value = false;
            };
            template <>
            struct is_ni128<ni128>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_UINT128
    #endif // UINT128
    #ifdef FLOAT16
        #ifdef NEURAL_FLOAT16
            typedef Neural<f16> nf16;
            template <typename T>
            struct is_nf16
            {
                static const bool value = false;
            };
            template <>
            struct is_nf16<nf16>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<nf16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_FLOAT16
    #endif // FLOAT16
    #ifdef FLOAT32
        #ifdef NEURAL_FLOAT32
            typedef Neural<f32> nf32;
            template <typename T>
            struct is_nf32
            {
                static const bool value = false;
            };
            template <>
            struct is_nf32<nf32>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<nf32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_FLOAT32
    #endif // FLOAT32
    #ifdef FLOAT64
        #ifdef NEURAL_FLOAT64
            typedef Neural<f64> nf64;
            template <typename T>
            struct is_nf64
            {
                static const bool value = false;
            };
            template <>
            struct is_nf64<nf64>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<nf64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_FLOAT64
    #endif // FLOAT64
    #ifdef FLOAT128
        #ifdef NEURAL_FLOAT128
            typedef Neural<f128> nf128;
            template <typename T>
            struct is_nf128
            {
                static const bool value = false;
            };
            template <>
            struct is_nf128<nf128>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<nf128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_FLOAT128
    #endif // FLOAT128
    #ifdef INT8
        #ifdef NEURAL_COMPLEX_INT8
            typedef Complex<Neural<i8>> cni8;
            template <typename T>
            struct is_cni8
            {
                static const bool value = false;
            };
            template <>
            struct is_cni8<cni8>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni8>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_INT8
    #endif // INT8
    #ifdef INT16
        #ifdef NEURAL_COMPLEX_INT16
            typedef Complex<Neural<i16>> cni16;
            template <typename T>
            struct is_ni16
            {
                static const bool value = false;
            };
            template <>
            struct is_ni16<ni16>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<ni16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_INT16
    #endif // INT16
    #ifdef INT32
        #ifdef NEURAL_COMPLEX_INT32
            typedef Complex<Neural<i32>> cni32;
            template <typename T>
            struct is_cni32
            {
                static const bool value = false;
            };
            template <>
            struct is_cni32<cni32>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_INT32
    #endif // INT32
    #ifdef INT64
        #ifdef NEURAL_COMPLEX_INT64
            typedef Complex<Neural<i64>> cni64;
            template <typename T>
            struct is_cni64
            {
                static const bool value = false;
            };
            template <>
            struct is_cni64<cni64>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_INT64
    #endif // INT64
    #ifdef INT128
        #ifdef NEURAL_COMPLEX_INT128
            typedef Complex<Neural<i128>> cni128;
            template <typename T>
            struct is_cni128
            {
                static const bool value = false;
            };
            template <>
            struct is_cni128<cni128>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_INT128
    #endif // INT128
    #ifdef UINT8
        #ifdef NEURAL_COMPLEX_UINT8
            typedef Complex<Neural<i8>> cni8;
            template <typename T>
            struct is_cni8
            {
                static const bool value = false;
            };
            template <>
            struct is_cni8<cni8>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni8>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_UINT8
    #endif // UINT8
    #ifdef UINT16
        #ifdef NEURAL_COMPLEX_UINT16
            typedef Complex<Neural<i16>> cni16;
            template <typename T>
            struct is_cni16
            {
                static const bool value = false;
            };
            template <>
            struct is_cni16<cni16>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_UINT16
    #endif // UINT16
    #ifdef UINT32
        #ifdef NEURAL_COMPLEX_UINT32
            typedef Complex<Neural<i32>> cni32;
            template <typename T>
            struct is_cni32
            {
                static const bool value = false;
            };
            template <>
            struct is_cni32<cni32>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_UINT32
    #endif // UINT32
    #ifdef UINT64
        #ifdef NEURAL_COMPLEX_UINT64
            typedef Complex<Neural<i64>> cni64;
            template <typename T>
            struct is_cni64
            {
                static const bool value = false;
            };
            template <>
            struct is_cni64<cni64>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_UINT64
    #endif // UINT64
    #ifdef UINT128
        #ifdef NEURAL_COMPLEX_UINT128
            typedef Complex<Neural<i128>> cni128;
            template <typename T>
            struct is_cni128
            {
                static const bool value = false;
            };
            template <>
            struct is_cni128<cni128>
            {
                static const bool value = true;
            };
            template <>
            struct is_int<cni128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_UINT128
    #endif // UINT128
    #ifdef FLOAT16
        #ifdef NEURAL_COMPLEX_FLOAT16
            typedef Complex<Neural<f16>> cnf16;
            template <typename T>
            struct is_cnf16
            {
                static const bool value = false;
            };
            template <>
            struct is_cnf16<cnf16>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<cnf16>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_FLOAT16
    #endif // FLOAT16
    #ifdef FLOAT32
        #ifdef NEURAL_COMPLEX_FLOAT32
            typedef Complex<Neural<f32>> cnf32;
            template <typename T>
            struct is_cnf32
            {
                static const bool value = false;
            };
            template <>
            struct is_cnf32<cnf32>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<cnf32>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_FLOAT32
    #endif // FLOAT32
    #ifdef FLOAT64
        #ifdef NEURAL_COMPLEX_FLOAT64
            typedef Complex<Neural<f64>> cnf64;
            template <typename T>
            struct is_cnf64
            {
                static const bool value = false;
            };
            template <>
            struct is_cnf64<cnf64>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<cnf64>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_FLOAT64
    #endif // FLOAT64
    #ifdef FLOAT128
        #ifdef NEURAL_COMPLEX_FLOAT128
            typedef Complex<Neural<f128>> cnf128;
            template <typename T>
            struct is_cnf128
            {
                static const bool value = false;
            };
            template <>
            struct is_cnf128<cnf128>
            {
                static const bool value = true;
            };
            template <>
            struct is_float<cnf128>
            {
                static const bool value = true;
            };
        #endif // NEURAL_COMPLEX_FLOAT128
    #endif // FLOAT128
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_HPP_

