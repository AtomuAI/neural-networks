// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_COMPLEX_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_COMPLEX_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"

namespace nn
{
    template <typename T>
    class Complex;

    template <typename T> struct is_complex;
    template <typename T> struct is_complex_int;
    template <typename T> struct is_complex_float;
    template <typename T> struct is_complex_signed_int;
    template <typename T> struct is_complex_unsigned_int;

    template <typename T>
    class Complex
    {
        protected:
            std::complex<T> value;

        public:
                                    constexpr           Complex();
                                    constexpr           Complex( const T& real, const T& imag );
            template <typename U>   constexpr explicit  Complex( const std::complex<U>& value );
            template <typename U>   constexpr explicit  Complex( const U scalar );

        public:
            constexpr T     real() const;
            constexpr T     imag() const;
            constexpr void  real( T value );
            constexpr void  imag( T value );

        public:
                                    constexpr void          operator=   ( const Complex<T> other );
            template <typename U>   constexpr void          operator=   ( const Complex<U> other );
            template <typename U>   constexpr void          operator=   ( const U scalar );
                                    constexpr Complex<T>    operator+   ( const Complex<T> other ) const;
            template <typename U>   constexpr Complex<T>    operator+   ( const Complex<U> other ) const;
            template <typename U>   constexpr Complex<T>    operator+   ( const U scalar ) const;
                                    constexpr Complex<T>    operator-   ( const Complex<T> other ) const;
            template <typename U>   constexpr Complex<T>    operator-   ( const Complex<U> other ) const;
            template <typename U>   constexpr Complex<T>    operator-   ( const U scalar ) const;
                                    constexpr Complex<T>    operator*   ( const Complex<T> other ) const;
            template <typename U>   constexpr Complex<T>    operator*   ( const Complex<U> other ) const;
            template <typename U>   constexpr Complex<T>    operator*   ( const U scalar ) const;
                                    constexpr Complex<T>    operator/   ( const Complex<T> other ) const;
            template <typename U>   constexpr Complex<T>    operator/   ( const Complex<U> other ) const;
            template <typename U>   constexpr Complex<T>    operator/   ( const U scalar ) const;
                                    constexpr Complex<T>&   operator+=  ( const Complex<T> other );
            template <typename U>   constexpr Complex<T>&   operator+=  ( const Complex<U> other );
            template <typename U>   constexpr Complex<T>&   operator+=  ( const U scalar );
                                    constexpr Complex<T>&   operator-=  ( const Complex<T> other );
            template <typename U>   constexpr Complex<T>&   operator-=  ( const Complex<U> other );
            template <typename U>   constexpr Complex<T>&   operator-=  ( const U scalar );
                                    constexpr Complex<T>&   operator*=  ( const Complex<T> other );
            template <typename U>   constexpr Complex<T>&   operator*=  ( const Complex<U> other );
            template <typename U>   constexpr Complex<T>&   operator*=  ( const U scalar );
                                    constexpr Complex<T>&   operator/=  ( const Complex<T> other );
            template <typename U>   constexpr Complex<T>&   operator/=  ( const Complex<U> other );
            template <typename U>   constexpr Complex<T>&   operator/=  ( const U scalar );
            template <typename U>   constexpr bool          operator==  ( const Complex<U> other ) const;
            template <typename U>   constexpr bool          operator!=  ( const Complex<U> other ) const;
            template <typename U>   constexpr bool          operator>   ( const Complex<U> other ) const;
            template <typename U>   constexpr bool          operator<   ( const Complex<U> other ) const;
            template <typename U>   constexpr bool          operator>=  ( const Complex<U> other ) const;
            template <typename U>   constexpr bool          operator<=  ( const Complex<U> other ) const;
                                    constexpr Complex<T>    operator-   () const;
                                    constexpr Complex<T>    operator+   () const;

            template <typename U>   friend std::ostream&    operator<<  ( std::ostream& os, const Complex<U>& c );
            template <typename U>   friend std::istream&    operator>>  ( std::istream& is, Complex<U>& c );
    };

    template <typename T>
    constexpr Complex<T>::Complex() : value( T( 0 ), T( 0 ) ) {}

    template <typename T>
    constexpr Complex<T>::Complex( const T& real, const T& imag ) : value( real, imag ) {}

    template <typename T>
    template <typename U>
    constexpr Complex<T>::Complex( const std::complex<U>& value ) : value( value ) {}

    template <typename T>
    template <typename U>
    constexpr Complex<T>::Complex( const U scalar ) : value( scalar, 0 ) {}

    template <typename T>
    constexpr T Complex<T>::real() const { return this->value.real(); }

    template <typename T>
    constexpr T Complex<T>::imag() const { return this->value.imag(); }

    template <typename T>
    constexpr void Complex<T>::real( T value ) { this->value.real( value ); }

    template <typename T>
    constexpr void Complex<T>::imag( T value ) { this->value.imag( value ); }

    template <typename T>
    constexpr void Complex<T>::operator=( const Complex<T> other )
    {
        this->value = other.value;
    }

    template <typename T>
    template <typename U>
    constexpr void Complex<T>::operator=( const Complex<U> other )
    {
        this->value = std::complex<T>( other.real(), other.imag() );
    }

    template <typename T>
    template <typename U>
    constexpr void Complex<T>::operator=( const U scalar )
    {
        this->value.real( scalar );
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator+( const Complex<T> other ) const
    {
        return Complex<T>( this->value + other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator+( const Complex<U> other ) const
    {
        return Complex<T>( this->value + std::complex<T>( other.real(), other.imag() ) );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator+( const U scalar ) const
    {
        return Complex<T>( this->value + scalar );
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator-( const Complex<T> other ) const
    {
        return Complex<T>( this->value - other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator-( const Complex<U> other ) const
    {
        return Complex<T>( this->value - std::complex<T>( other.real(), other.imag() ) );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator-( const U scalar ) const
    {
        return Complex<T>( this->value - scalar );
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator*( const Complex<T> other ) const
    {
        return Complex<T>( this->value * other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator*( const Complex<U> other ) const
    {
        return Complex<T>( this->value * std::complex<T>( other.real(), other.imag() ) );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator*( const U scalar ) const
    {
        return Complex<T>( this->value * scalar );
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator/( const Complex<T> other ) const
    {
        return Complex<T>( this->value / other.value );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator/( const Complex<U> other ) const
    {
        return Complex<T>( this->value / std::complex<T>( other.real(), other.imag() ) );
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T> Complex<T>::operator/( const U scalar ) const
    {
        return Complex<T>( this->value / scalar );
    }

    template <typename T>
    constexpr Complex<T>& Complex<T>::operator+=( const Complex<T> other )
    {
        this->value += other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator+=( const Complex<U> other )
    {
        this->value += std::complex<T>( other.real(), other.imag() );
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator+=( const U scalar )
    {
        this->value += scalar;
        return *this;
    }

    template <typename T>
    constexpr Complex<T>& Complex<T>::operator-=( const Complex<T> other )
    {
        this->value -= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator-=( const Complex<U> other )
    {
        this->value -= std::complex<T>( other.real(), other.imag() );
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator-=( const U scalar )
    {
        this->value -= scalar;
        return *this;
    }

    template <typename T>
    constexpr Complex<T>& Complex<T>::operator*=( const Complex<T> other )
    {
        this->value *= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator*=( const Complex<U> other )
    {
        this->value *= std::complex<T>( other.real(), other.imag() );
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator*=( const U scalar )
    {
        this->value *= scalar;
        return *this;
    }

    template <typename T>
    constexpr Complex<T>& Complex<T>::operator/=( const Complex<T> other )
    {
        this->value /= other.value;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator/=( const Complex<U> other )
    {
        this->value /= std::complex<T>( other.real(), other.imag() );
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr Complex<T>& Complex<T>::operator/=( const U scalar )
    {
        this->value /= scalar;
        return *this;
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator==( const Complex<U> other ) const
    {
        return this->value == other.value;
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator!=( const Complex<U> other ) const
    {
        return this->value != other.value;
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator>( const Complex<U> other ) const
    {
        return ( this->value.real() > other.value.real() ) && ( this->value.imag() > other.value.imag() );
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator<( const Complex<U> other ) const
    {
        return ( this->value.real() < other.value.real() ) && ( this->value.imag() < other.value.imag() );
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator>=( const Complex<U> other ) const
    {
        return this->value >= other.value;
    }

    template <typename T>
    template <typename U>
    constexpr bool Complex<T>::operator<=( const Complex<U> other ) const
    {
        return this->value <= other.value;
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator-() const
    {
        return Complex<T>( -this->value );
    }

    template <typename T>
    constexpr Complex<T> Complex<T>::operator+() const
    {
        return Complex<T>( +this->value );
    }

    template <typename U>
    std::ostream& operator<<( std::ostream& os, const Complex<U>& c )
    {
        os << c.value;
        return os;
    }

    template <typename U>
    std::istream& operator>>( std::istream& is, Complex<U>& c )
    {
        is >> c.value;
        return is;
    }

    template <typename T, typename U>   constexpr T     operator+   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T     operator-   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T     operator*   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value );
    template <typename T, typename U>   constexpr T     operator*   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T     operator/   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value );
    template <typename T, typename U>   constexpr T     operator/   ( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T&    operator+=  ( T& lhs, const Complex<U> rhs )      requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T&    operator-=  ( T& lhs, const Complex<U> rhs )      requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T&    operator*=  ( T& lhs, const Complex<U> rhs )      requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );
    template <typename T, typename U>   constexpr T&    operator/=  ( T& lhs, const Complex<U> rhs )      requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value );

    template <typename T, typename U>
    constexpr T operator+( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return lhs + rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator-( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return lhs - rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator*( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value )
    {
        Complex<T> c = Complex<T>( lhs, 0 ) * rhs;
        return c.real();
    }

    template <typename T, typename U>
    constexpr T operator*( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value )
    {
        return lhs * rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator/( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value )
    {
        Complex<T> c = Complex<T>( lhs, 0 ) / rhs;
        return c.real();
    }

    template <typename T, typename U>
    constexpr T operator/( const T lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value )
    {
        return lhs / rhs.real();
    }

    template <typename T, typename U>
    constexpr T& operator+=( T& lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        lhs += rhs.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator-=( T& lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        lhs -= rhs.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator*=( T& lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        Complex<T> c = lhs * rhs;
        lhs = c.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator/=( T& lhs, const Complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        Complex<T> c = lhs / rhs;
        lhs = c.real();
        return lhs;
    }

    template<typename T>
    struct is_complex
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex<Complex<T>>
    {
        static const bool value = true;
    };

    template<typename T>
    struct is_complex_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_int<Complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_float
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_float<Complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_signed_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_signed_int<Complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_unsigned_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_unsigned_int<Complex<T>>
    {
        static const bool value = false;
    };
    #ifdef INT8
        #ifdef COMPLEX_INT8
            typedef Complex<i8>    ci8;
            template<typename T>
            struct is_ci8
            {
                static const bool value = false;
            };
            template<>
            struct is_ci8<ci8>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<ci8>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_signed_int<ci8>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef INT16
        #ifdef COMPLEX_INT16
            typedef Complex<i16>   ci16;
            template<typename T>
            struct is_ci16
            {
                static const bool value = false;
            };
            template<>
            struct is_ci16<ci16>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<ci16>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_signed_int<ci16>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef INT32
        #ifdef COMPLEX_INT32
            typedef Complex<i32>   ci32;
            template<typename T>
            struct is_ci32
            {
                static const bool value = false;
            };
            template<>
            struct is_ci32<ci32>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<ci32>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_signed_int<ci32>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef INT64
        #ifdef COMPLEX_INT64
            typedef Complex<i64>   ci64;
            template<typename T>
            struct is_ci64
            {
                static const bool value = false;
            };
            template<>
            struct is_ci64<ci64>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<ci64>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_signed_int<ci64>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef INT128
        #ifdef COMPLEX_INT128
            typedef Complex<i128>  ci128;
            template<typename T>
            struct is_ci128
            {
                static const bool value = false;
            };
            template<>
            struct is_ci128<ci128>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<ci128>
            {
                static const bool value = true;
            };
            typedef Complex<i128>  ci128;
            template<>
            struct is_complex_signed_int<ci128>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT8
        #ifdef COMPLEX_UINT8
            typedef Complex<u8>    cu8;
            template<typename T>
            struct is_cu8
            {
                static const bool value = false;
            };
            template<>
            struct is_cu8<cu8>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<cu8>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_unsigned_int<cu8>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT16
        #ifdef COMPLEX_UINT16
            typedef Complex<u16>   cu16;
            template<typename T>
            struct is_cu16
            {
                static const bool value = false;
            };
            template<>
            struct is_cu16<cu16>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<cu16>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_unsigned_int<cu16>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT32
        #ifdef COMPLEX_UINT32
            typedef Complex<u32>   cu32;
            template<typename T>
            struct is_cu32
            {
                static const bool value = false;
            };
            template<>
            struct is_cu32<cu32>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<cu32>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_unsigned_int<cu32>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT64
        #ifdef COMPLEX_UINT64
            typedef Complex<u64>   cu64;
            template<typename T>
            struct is_cu64
            {
                static const bool value = false;
            };
            template<>
            struct is_cu64<cu64>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<cu64>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_unsigned_int<cu64>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT128
        #ifdef COMPLEX_UINT128
            typedef Complex<u128>  cu128;
            template<typename T>
            struct is_cu128
            {
                static const bool value = false;
            };
            template<>
            struct is_cu128<cu128>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_int<cu128>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_unsigned_int<cu128>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef FLOAT16
        #ifdef COMPLEX_FLOAT16
            typedef Complex<f16>   cf16;
            template<typename T>
            struct is_cf16
            {
                static const bool value = false;
            };
            template<>
            struct is_cf16<cf16>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_float<cf16>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef FLOAT32
        #ifdef COMPLEX_FLOAT32
            typedef Complex<f32>   cf32;
            template<typename T>
            struct is_cf32
            {
                static const bool value = false;
            };
            template<>
            struct is_cf32<cf32>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_float<cf32>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef FLOAT64
        #ifdef COMPLEX_FLOAT64
            typedef Complex<f64>   cf64;
            template<typename T>
            struct is_cf64
            {
                static const bool value = false;
            };
            template<>
            struct is_cf64<cf64>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_float<cf64>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef FLOAT128
        #ifdef COMPLEX_FLOAT128
            typedef Complex<f128>  cf128;
            template<typename T>
            struct is_cf128
            {
                static const bool value = false;
            };
            template<>
            struct is_cf128<cf128>
            {
                static const bool value = true;
            };
            template<>
            struct is_complex_float<cf128>
            {
                static const bool value = true;
            };
        #endif
    #endif

    template <typename T, typename U>
    constexpr T convert( U value ) requires ( !is_complex<T>::value && ( is_int<T>::value || is_float<T>::value || is_bool<T>::value ) && ( is_complex_int<U>::value || is_complex_float<U>::value ) )
    {
        T a( value.real() );
        return a;
    }

    template <typename T, typename U>
    constexpr T convert( U value ) requires ( ( is_complex_int<T>::value || is_complex_float<T>::value ) && ( is_complex_int<U>::value || is_complex_float<U>::value ) )
    {
        T a( value );
        return a;
    }

    template <typename T, typename U>
    constexpr T convert( U value ) requires ( ( is_complex_int<T>::value || is_complex_float<T>::value ) && !is_complex<U>::value && ( is_int<U>::value || is_float<U>::value || is_bool<U>::value ) )
    {
        T a( value );
        return a;
    }

    template <typename T, typename U>
    constexpr T convert( U value ) requires ( ( is_int<T>::value || is_float<T>::value || is_bool<T>::value ) && ( is_int<U>::value || is_float<U>::value || is_bool<U>::value  ) )
    {
        T a( value );
        return a;
    }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_COMPLEX_HPP_
