// Copyright 2024 Shane W. Mulcahy

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

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/dim.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T, u8 N> Tensor<T, N>::Tensor() : shape( Shape<N>() ) {}
        template <typename T, u8 N> Tensor<T, N>::Tensor( const Shape<N> shape ) : shape( shape ), data( shape.volume() ) {}
        template <typename T, u8 N> Tensor<T, N>::Tensor( const Shape<N> shape, const T scalar ) : shape( shape ), data( shape.volume(), scalar ) {}
        template <typename T, u8 N>
        Tensor<T, N>::Tensor( const Shape<N> shape, const std::vector<T>& data ) requires ( !is_type<T, bool>::value ) : shape( shape ), data( std::move( data ) )
        {
            Size size = shape.volume();
            if ( size != this->data.size() )
            {
                this->data.resize( size );
            }
        }
        template <typename T, u8 N>
        Tensor<T, N>::Tensor( const Shape<N> shape, const std::vector<T>& data ) requires ( is_type<T, bool>::value ) : shape( shape ), data( std::move( std::vector<u8>( data.begin(), data.end() ) ) )
        {
            Size size = shape.volume();
            if ( size != this->data.size() )
            {
                this->data.resize( size );
            }
        }

    //: Destructors
        template <typename T, u8 N> Tensor<T, N>::~Tensor() {}

    //: Methods
        template <typename T, u8 N>
        const Shape<N>& Tensor<T, N>::get_shape() const
        {
            return shape;
        }

        template <typename T, u8 N>
        Size Tensor<T, N>::get_size() const
        {
            return data.size();
        }

        template <typename T, u8 N>
        const std::vector<typename std::conditional<is_bool<T>::value, u8, T>::type>& Tensor<T, N>::get_vector() const
        {
            return this->data;
        }

        template <typename T, u8 N>
        const T* const Tensor<T, N>::get_ptr() const requires ( !is_type<T, bool>::value )
        {
            return this->data.data();
        }

        template <typename T, u8 N>
        const T* const Tensor<T, N>::get_ptr() const requires ( is_type<T, bool>::value )
        {
            return reinterpret_cast<const bool*>( this->data.data() );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::reshape( const Shape<N>& shape ) requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            if ( shape == this->shape ) { return; }
            this->shape.reshape( shape.dim );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::resize( const Shape<N>& shape )
        {
            if ( shape == this->shape ) { return; }
            this->shape.resize( shape.dim );
            this->data.resize( shape.volume() );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::resize( const Shape<N>& shape, T value ) requires ( !is_bool<T>::value )
        {
            if ( shape == this->shape ) { return; }
            this->shape.resize( shape.dim );
            data.resize( shape.volume(), value );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::fill( const T value ) requires ( !is_bool<T>::value )
        {
            std::fill( data.begin(), data.end(), value );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::set( const std::vector<T> set ) requires ( !is_bool<T>::value )
        {
            if ( data.size() != set.size() ) // Check that the shapes are the same.
            {
                throw std::invalid_argument( "Size must be the same to set Tensor" );
            }

            data = set;
        }

        template <typename T, u8 N>
        void Tensor<T, N>::zero() requires ( !is_bool<T>::value )
        {
            std::fill( data.begin(), data.end(), 0.0 );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::randomize( const T min, const T max ) requires ( !is_bool<T>::value )
        {
            std::generate( data.begin(), data.end(), [ & ]() -> T { return random_value( min, max ); } );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::resize( const Shape<N>& shape, T value ) requires ( is_bool<T>::value )
        {
            if ( shape == this->shape ) { return; }
            this->shape.resize( shape.dim );
            data.resize( shape.volume(), static_cast<u8>( value ) );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::fill( const T value ) requires ( is_bool<T>::value )
        {
            std::fill( data.begin(), data.end(), static_cast<u8>( value ) );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::set( std::vector<T> set ) requires ( is_bool<T>::value )
        {
            if ( data.size() != set.size() ) // Check that the shapes are the same.
            {
                throw std::invalid_argument( "Size must be the same to set Tensor" );
            }

            data = std::vector<u8>( set.begin(), set.end() );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::zero() requires ( is_bool<T>::value )
        {
            std::fill( data.begin(), data.end(), 0 );
        }

        template <typename T, u8 N>
        void Tensor<T, N>::randomize( const T min, const T max ) requires ( is_bool<T>::value )
        {
            std::generate( data.begin(), data.end(), [ & ]() -> T { return random_value(); } );
        }
} // namespace nn

//: Specializations
namespace nn
{
    #ifdef BOOL // bool
        template class Tensor<bool, 1>;
        template class Tensor<bool, 2>;
        template class Tensor<bool, 3>;
        template class Tensor<bool, 4>;
        template class Tensor<bool, 5>;
    #endif
    #ifdef INT8 // i8
        template class Tensor<i8, 1>;
        template class Tensor<i8, 2>;
        template class Tensor<i8, 3>;
        template class Tensor<i8, 4>;
        template class Tensor<i8, 5>;
    #endif
    #ifdef INT16 // i16
        template class Tensor<i16, 1>;
        template class Tensor<i16, 2>;
        template class Tensor<i16, 3>;
        template class Tensor<i16, 4>;
        template class Tensor<i16, 5>;
    #endif
    #ifdef INT32 // i32
        template class Tensor<i32, 1>;
        template class Tensor<i32, 2>;
        template class Tensor<i32, 3>;
        template class Tensor<i32, 4>;
        template class Tensor<i32, 5>;
    #endif
    #ifdef INT64 // i64
        template class Tensor<i64, 1>;
        template class Tensor<i64, 2>;
        template class Tensor<i64, 3>;
        template class Tensor<i64, 4>;
        template class Tensor<i64, 5>;
    #endif
    #ifdef INT128 // i128
        template class Tensor<i128, 1>;
        template class Tensor<i128, 2>;
        template class Tensor<i128, 3>;
        template class Tensor<i128, 4>;
        template class Tensor<i128, 5>;
    #endif
    #ifdef UINT8 // u8
        template class Tensor<u8, 1>;
        template class Tensor<u8, 2>;
        template class Tensor<u8, 3>;
        template class Tensor<u8, 4>;
        template class Tensor<u8, 5>;
    #endif
    #ifdef UINT16 // u16
        template class Tensor<u16, 1>;
        template class Tensor<u16, 2>;
        template class Tensor<u16, 3>;
        template class Tensor<u16, 4>;
        template class Tensor<u16, 5>;
    #endif
    #ifdef UINT32 // u32
        template class Tensor<u32, 1>;
        template class Tensor<u32, 2>;
        template class Tensor<u32, 3>;
        template class Tensor<u32, 4>;
        template class Tensor<u32, 5>;
    #endif
    #ifdef UINT64 // u64
        template class Tensor<u64, 1>;
        template class Tensor<u64, 2>;
        template class Tensor<u64, 3>;
        template class Tensor<u64, 4>;
        template class Tensor<u64, 5>;
    #endif
    #ifdef UINT128 // u128
        template class Tensor<u128, 1>;
        template class Tensor<u128, 2>;
        template class Tensor<u128, 3>;
        template class Tensor<u128, 4>;
        template class Tensor<u128, 5>;
    #endif
    #ifdef FLOAT16 // f16
        template class Tensor<f16, 1>;
        template class Tensor<f16, 2>;
        template class Tensor<f16, 3>;
        template class Tensor<f16, 4>;
        template class Tensor<f16, 5>;
    #endif
    #ifdef FLOAT32 // f32
        template class Tensor<f32, 1>;
        template class Tensor<f32, 2>;
        template class Tensor<f32, 3>;
        template class Tensor<f32, 4>;
        template class Tensor<f32, 5>;
    #endif
    #ifdef FLOAT64 // f64
        template class Tensor<f64, 1>;
        template class Tensor<f64, 2>;
        template class Tensor<f64, 3>;
        template class Tensor<f64, 4>;
        template class Tensor<f64, 5>;
    #endif
    #ifdef FLOAT128 // f128
        template class Tensor<f128, 1>;
        template class Tensor<f128, 2>;
        template class Tensor<f128, 3>;
        template class Tensor<f128, 4>;
        template class Tensor<f128, 5>;
    #endif
    #ifdef COMPLEX_INT8 // ci8
        template class Tensor<ci8, 1>;
        template class Tensor<ci8, 2>;
        template class Tensor<ci8, 3>;
        template class Tensor<ci8, 4>;
        template class Tensor<ci8, 5>;
    #endif
    #ifdef COMPLEX_INT16 // ci16
        template class Tensor<ci16, 1>;
        template class Tensor<ci16, 2>;
        template class Tensor<ci16, 3>;
        template class Tensor<ci16, 4>;
        template class Tensor<ci16, 5>;
    #endif
    #ifdef COMPLEX_INT32 // ci32
        template class Tensor<ci32, 1>;
        template class Tensor<ci32, 2>;
        template class Tensor<ci32, 3>;
        template class Tensor<ci32, 4>;
        template class Tensor<ci32, 5>;
    #endif
    #ifdef COMPLEX_INT64 // ci64
        template class Tensor<ci64, 1>;
        template class Tensor<ci64, 2>;
        template class Tensor<ci64, 3>;
        template class Tensor<ci64, 4>;
        template class Tensor<ci64, 5>;
    #endif
    #ifdef COMPLEX_INT128 // ci128
        template class Tensor<ci128, 1>;
        template class Tensor<ci128, 2>;
        template class Tensor<ci128, 3>;
        template class Tensor<ci128, 4>;
        template class Tensor<ci128, 5>;
    #endif
    #ifdef COMPLEX_UINT8 // cu8
        template class Tensor<cu8, 1>;
        template class Tensor<cu8, 2>;
        template class Tensor<cu8, 3>;
        template class Tensor<cu8, 4>;
        template class Tensor<cu8, 5>;
    #endif
    #ifdef COMPLEX_UINT16 // cu16
        template class Tensor<cu16, 1>;
        template class Tensor<cu16, 2>;
        template class Tensor<cu16, 3>;
        template class Tensor<cu16, 4>;
        template class Tensor<cu16, 5>;
    #endif
    #ifdef COMPLEX_UINT32 // cu32
        template class Tensor<cu32, 1>;
        template class Tensor<cu32, 2>;
        template class Tensor<cu32, 3>;
        template class Tensor<cu32, 4>;
        template class Tensor<cu32, 5>;
    #endif
    #ifdef COMPLEX_UINT64 // cu64
        template class Tensor<cu64, 1>;
        template class Tensor<cu64, 2>;
        template class Tensor<cu64, 3>;
        template class Tensor<cu64, 4>;
        template class Tensor<cu64, 5>;
    #endif
    #ifdef COMPLEX_UINT128 // cu128
        template class Tensor<cu128, 1>;
        template class Tensor<cu128, 2>;
        template class Tensor<cu128, 3>;
        template class Tensor<cu128, 4>;
        template class Tensor<cu128, 5>;
    #endif
    #ifdef COMPLEX_FLOAT16 // cf16
        template class Tensor<cf16, 1>;
        template class Tensor<cf16, 2>;
        template class Tensor<cf16, 3>;
        template class Tensor<cf16, 4>;
        template class Tensor<cf16, 5>;
    #endif
    #ifdef COMPLEX_FLOAT32 // cf32
        template class Tensor<cf32, 1>;
        template class Tensor<cf32, 2>;
        template class Tensor<cf32, 3>;
        template class Tensor<cf32, 4>;
        template class Tensor<cf32, 5>;
    #endif
    #ifdef COMPLEX_FLOAT64 // cf64
        template class Tensor<cf64, 1>;
        template class Tensor<cf64, 2>;
        template class Tensor<cf64, 3>;
        template class Tensor<cf64, 4>;
        template class Tensor<cf64, 5>;
    #endif
    #ifdef COMPLEX_FLOAT128 // cf128
        template class Tensor<cf128, 1>;
        template class Tensor<cf128, 2>;
        template class Tensor<cf128, 3>;
        template class Tensor<cf128, 4>;
        template class Tensor<cf128, 5>;
    #endif
} // namespace nn
