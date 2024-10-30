// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_FUNDAMENTAL_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_FUNDAMENTAL_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>

#define BOOL
#define INT8
#define INT16
#define INT32
#define INT64
//#define INT128
#define UINT8
#define UINT16
#define UINT32
#define UINT64
//#define UINT128
//#define FLOAT16
#define FLOAT32
#define FLOAT64
//#define FLOAT128
//#define COMPLEX_INT8
//#define COMPLEX_INT16
//#define COMPLEX_INT32
//#define COMPLEX_INT64
//#define COMPLEX_INT128
//#define COMPLEX_UINT8
//#define COMPLEX_UINT16
//#define COMPLEX_UINT32
//#define COMPLEX_UINT64
//#define COMPLEX_UINT128
//#define COMPLEX_FLOAT16
#define COMPLEX_FLOAT32
#define COMPLEX_FLOAT64
//#define COMPLEX_FLOAT128

namespace nn
{
    #ifdef BOOL
        typedef bool                b8;
    #endif
    #ifdef INT8
        typedef std::int8_t         i8;
    #endif
    #ifdef INT16
        typedef std::int16_t        i16;
    #endif
    #ifdef INT32
        typedef std::int32_t        i32;
    #endif
    #ifdef INT64
        typedef std::int64_t        i64;
    #endif
    #ifdef INT128
        typedef __int128_t          i128;
    #endif
    #ifdef UINT8
        typedef std::uint8_t        u8;
    #endif
    #ifdef UINT16
        typedef std::uint16_t       u16;
    #endif
    #ifdef UINT32
        typedef std::uint32_t       u32;
    #endif
    #ifdef UINT64
        typedef std::uint64_t       u64;
    #endif
    #ifdef UINT128
        typedef __uint128_t         u128;
    #endif
    #ifdef FLOAT16
        typedef _Float16            f16;
    #endif
    #ifdef FLOAT32
        typedef std::float_t        f32;
    #endif
    #ifdef FLOAT64
        typedef std::double_t       f64;
    #endif
    #ifdef FLOAT128
        typedef _Float128           f128;
    #endif

    template<typename T, typename U>
    struct is_type
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_type<T, T>
    {
        static const bool value = true;
    };

    template<typename T>
    struct is_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_signed_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_unsigned_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_float
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_bool
    {
        static const bool value = false;
    };

    template<>
    struct is_bool<bool>
    {
        static const bool value = true;
    };

    #ifdef INT8
        template<typename T>
        struct is_i8
        {
            static const bool value = false;
        };
        template<>
        struct is_i8<i8>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<i8>
        {
            static const bool value = true;
        };
        template<>
        struct is_signed_int<i8>
        {
            static const bool value = true;
        };
    #endif
    #ifdef INT16
        template<typename T>
        struct is_i16
        {
            static const bool value = false;
        };
        template<>
        struct is_i16<i16>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<i16>
        {
            static const bool value = true;
        };
        template<>
        struct is_signed_int<i16>
        {
            static const bool value = true;
        };
    #endif
    #ifdef INT32
        template<typename T>
        struct is_i32
        {
            static const bool value = false;
        };
        template<>
        struct is_i32<i32>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<i32>
        {
            static const bool value = true;
        };
        template<>
        struct is_signed_int<i32>
        {
            static const bool value = true;
        };
    #endif
    #ifdef INT64
        template<typename T>
        struct is_i64
        {
            static const bool value = false;
        };
        template<>
        struct is_i64<i64>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<i64>
        {
            static const bool value = true;
        };
        template<>
        struct is_signed_int<i64>
        {
            static const bool value = true;
        };
    #endif
    #ifdef INT128
        template<typename T>
        struct is_i128
        {
            static const bool value = false;
        };
        template<>
        struct is_i128<i128>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<i128>
        {
            static const bool value = true;
        };
        template<>
        struct is_signed_int<i128>
        {
            static const bool value = true;
        };
    #endif
    #ifdef UINT8
        template<typename T>
        struct is_u8
        {
            static const bool value = false;
        };
        template<>
        struct is_u8<u8>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<u8>
        {
            static const bool value = true;
        };
        template<>
        struct is_unsigned_int<u8>
        {
            static const bool value = true;
        };
    #endif
    #ifdef UINT16
        template<typename T>
        struct is_u16
        {
            static const bool value = false;
        };
        template<>
        struct is_u16<u16>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<u16>
        {
            static const bool value = true;
        };
        template<>
        struct is_unsigned_int<u16>
        {
            static const bool value = true;
        };
    #endif
    #ifdef UINT32
        template<typename T>
        struct is_u32
        {
            static const bool value = false;
        };
        template<>
        struct is_u32<u32>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<u32>
        {
            static const bool value = true;
        };
        template<>
        struct is_unsigned_int<u32>
        {
            static const bool value = true;
        };
    #endif
    #ifdef UINT64
        template<typename T>
        struct is_u64
        {
            static const bool value = false;
        };
        template<>
        struct is_u64<u64>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<u64>
        {
            static const bool value = true;
        };
        template<>
        struct is_unsigned_int<u64>
        {
            static const bool value = true;
        };
    #endif
    #ifdef UINT128
        template<typename T>
        struct is_u128
        {
            static const bool value = false;
        };
        template<>
        struct is_u128<u128>
        {
            static const bool value = true;
        };
        template<>
        struct is_int<u128>
        {
            static const bool value = true;
        };
        template<>
        struct is_unsigned_int<u128>
        {
            static const bool value = true;
        };
    #endif
    #ifdef FLOAT16
        template<typename T>
        struct is_f16
        {
            static const bool value = false;
        };
        template<>
        struct is_f16<f16>
        {
            static const bool value = true;
        };
        template<>
        struct is_float<f16>
        {
            static const bool value = true;
        };
    #endif
    #ifdef FLOAT32
        template<typename T>
        struct is_f32
        {
            static const bool value = false;
        };
        template<>
        struct is_f32<f32>
        {
            static const bool value = true;
        };
        template<>
        struct is_float<f32>
        {
            static const bool value = true;
        };
    #endif
    #ifdef FLOAT64
        template<typename T>
        struct is_f64
        {
            static const bool value = false;
        };
        template<>
        struct is_f64<f64>
        {
            static const bool value = true;
        };
        template<>
        struct is_float<f64>
        {
            static const bool value = true;
        };
    #endif
    #ifdef FLOAT128
        template<typename T>
        struct is_f128
        {
            static const bool value = false;
        };
        template<>
        struct is_f128<f128>
        {
            static const bool value = true;
        };
        template<>
        struct is_float<f128>
        {
            static const bool value = true;
        };
    #endif

    template <typename T>
    T operator+(T a, bool b) requires ( !is_bool<T>::value )
    {
        return a + b;
    }

    template <typename T>
    T operator+(bool a, T b) requires ( !is_bool<T>::value )
    {
        return a || b;
    }

    template <typename T>
    T operator+(T a, T b) requires ( is_bool<T>::value )
    {
        return a || b;
    }

    template <typename T>
    T operator-(T a, bool b) requires ( !is_bool<T>::value )
    {
        return a - b;
    }

    template <typename T>
    T operator-(bool a, T b) requires ( !is_bool<T>::value )
    {
        return a || !b;
    }

    template <typename T>
    T operator-(T a, T b) requires ( is_bool<T>::value )
    {
        return a || b;
    }

    template <typename T>
    T operator*(T a, bool b) requires ( !is_bool<T>::value )
    {
        return a * b;
    }

    template <typename T>
    T operator*(bool a, T b) requires ( !is_bool<T>::value )
    {
        return a && b;
    }

    template <typename T>
    T operator*(T a, T b) requires ( is_bool<T>::value )
    {
        return a && b;
    }

    template <typename T>
    T operator/(T a, bool b) requires ( !is_bool<T>::value )
    {
        return a / b;
    }

    template <typename T>
    T operator/(bool a, T b) requires ( !is_bool<T>::value )
    {
        return a && !b;
    }

    template <typename T>
    T operator/(T a, T b) requires ( is_bool<T>::value )
    {
        return a && !b;
    }

    template<typename T>
    struct is_complex
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex<std::complex<T>>
    {
        static const bool value = true;
    };

    template<typename T>
    struct is_complex_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_int<std::complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_float
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_float<std::complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_signed_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_signed_int<std::complex<T>>
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_unsigned_int
    {
        static const bool value = false;
    };

    template<typename T>
    struct is_complex_unsigned_int<std::complex<T>>
    {
        static const bool value = false;
    };
    #ifdef INT8
        #ifdef COMPLEX_INT8
            typedef std::complex<i8>    ci8;
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
            typedef std::complex<i16>   ci16;
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
            typedef std::complex<i32>   ci32;
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
            typedef std::complex<i64>   ci64;
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
            typedef std::complex<i128>  ci128;
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
            typedef std::complex<i128>  ci128;
            template<>
            struct is_complex_signed_int<ci128>
            {
                static const bool value = true;
            };
        #endif
    #endif
    #ifdef UINT8
        #ifdef COMPLEX_UINT8
            typedef std::complex<u8>    cu8;
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
            typedef std::complex<u16>   cu16;
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
            typedef std::complex<u32>   cu32;
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
            typedef std::complex<u64>   cu64;
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
            typedef std::complex<u128>  cu128;
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
            typedef std::complex<f16>   cf16;
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
            typedef std::complex<f32>   cf32;
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
            typedef std::complex<f64>   cf64;
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
            typedef std::complex<f128>  cf128;
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
    constexpr T operator+( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return lhs + rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator-( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return lhs - rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator*( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value )
    {
        std::complex<T> c = std::complex<T>( lhs, 0 ) * rhs;
        return c.real();
    }

    template <typename T, typename U>
    constexpr T operator*( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value )
    {
        return lhs * rhs.real();
    }

    template <typename T, typename U>
    constexpr T operator/( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value )
    {
        std::complex<T> c = std::complex<T>( lhs, 0 ) / rhs;
        return c.real();
    }

    template <typename T, typename U>
    constexpr T operator/( const T lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_int<T>::value || is_bool<T>::value )
    {
        return lhs / rhs.real();
    }

    template <typename T, typename U>
    constexpr T& operator+=( T& lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        lhs += rhs.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator-=( T& lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        lhs -= rhs.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator*=( T& lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        std::complex<T> c = lhs * rhs;
        lhs = c.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr T& operator/=( T& lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        std::complex<T> c = lhs / rhs;
        lhs = c.real();
        return lhs;
    }

    template <typename T, typename U>
    constexpr bool operator>( std::complex<T> lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return ( lhs.real() > rhs.real() ) && ( lhs.imag() > rhs.imag() );
    }

    template <typename T, typename U>
    constexpr bool operator<( std::complex<T> lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return ( lhs.real() < rhs.real() ) && ( lhs.imag() < rhs.imag() );
    }

    template <typename T, typename U>
    constexpr bool operator>=( std::complex<T> lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return ( lhs.real() >= rhs.real() ) && ( lhs.imag() >= rhs.imag() );
    }

    template <typename T, typename U>
    constexpr bool operator<=( std::complex<T> lhs, const std::complex<U> rhs ) requires ( !is_complex<T>::value && !is_complex<U>::value && is_float<T>::value || is_int<T>::value || is_bool<T>::value )
    {
        return ( lhs.real() <= rhs.real() ) && ( lhs.imag() <= rhs.imag() );
    }

    template <typename T, typename U>
    constexpr T cast( U value ) requires ( is_complex<T>::value && is_complex<U>::value )
    {
        T a( value );
        return a;
    }

    template <typename T, typename U>
    constexpr T cast( std::complex<U> value ) requires ( is_complex<T>::value && is_complex<U>::value )
    {
        T a( value.real() );
        return a;
    }

} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_FUNDAMENTAL_HPP_
