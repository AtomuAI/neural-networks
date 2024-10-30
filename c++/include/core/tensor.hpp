// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_TENSOR_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_TENSOR_HPP_

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
#include "bewusstsein_neural_networks/c++/include/core/random.hpp"

namespace nn
{
    template <typename T, u8 N>
    class Tensor;

    template <typename T> using Tensor1D = Tensor<T, 1>;
    template <typename T> using Tensor2D = Tensor<T, 2>;
    template <typename T> using Tensor3D = Tensor<T, 3>;
    template <typename T> using Tensor4D = Tensor<T, 4>;
    template <typename T> using Tensor5D = Tensor<T, 5>;

    template <typename T, u8 N>
    class Tensor
    {
        //: Members
        public:
            static const u8 dimensionality = N;
        private:
            Shape<N>                                                                shape;
            std::vector<typename std::conditional<is_bool<T>::value, u8, T>::type>  data;

        //: Constructors
        public:
                        Tensor();
            explicit    Tensor( const Shape<N> shape );
                        Tensor( const Shape<N> shape, T scalar );
                        Tensor( const Shape<N> shape, const std::vector<T>& data ) requires ( !is_type<T, bool>::value );
                        Tensor( const Shape<N> shape, const std::vector<T>& data ) requires ( is_type<T, bool>::value );

        //: Destructors
        public:
            virtual     ~Tensor();

        //: Methods
        public:
            const Shape<N>&         get_shape   () const;
            Size                    get_size    () const;
            const std::vector<typename std::conditional<is_bool<T>::value, u8, T>::type>&   get_vector  () const;
            const T* const          get_ptr() const requires ( !is_type<T, bool>::value );
            const T* const          get_ptr() const requires ( is_type<T, bool>::value );
            void                    reshape     ( const Shape<N>& shape ) requires ( ( N >= 2 ) && ( N <= 5 ) );
            void                    resize      ( const Shape<N>& shape );
            void                    resize      ( const Shape<N>& shape, T value ) requires ( !is_bool<T>::value );
            void                    fill        ( const T value ) requires ( !is_bool<T>::value );
            void                    set         ( std::vector<T> data ) requires ( !is_bool<T>::value );
            void                    zero        () requires ( !is_bool<T>::value );
            void                    randomize   ( const T min, const T max ) requires ( !is_bool<T>::value );
            void                    resize      ( const Shape<N>& shape, T value ) requires ( is_bool<T>::value );
            void                    fill        ( const T value ) requires ( is_bool<T>::value );
            void                    set         ( std::vector<T> data ) requires ( is_bool<T>::value );
            void                    zero        () requires ( is_bool<T>::value );
            void                    randomize   ( const T min, const T max ) requires ( is_bool<T>::value );

        //: Operators
        public:
            inline T                                    operator[]      ( const DimND<N>& indices ) const requires ( N == 2 );
            inline T                                    operator[]      ( const DimND<N>& indices ) const requires ( N == 3 );
            inline T                                    operator[]      ( const DimND<N>& indices ) const requires ( N == 4 );
            inline T                                    operator[]      ( const DimND<N>& indices ) const requires ( N == 5 );
            inline T                                    operator[]      ( const Idx index ) const requires ( !is_bool<T>::value );
            inline T                                    operator[]      ( const Idx index ) const requires ( is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 2 ) && !is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 2 ) && is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 3 ) && !is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 3 ) && is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 4 ) && !is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 4 ) && is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 5 ) && !is_bool<T>::value );
            inline T&                                   operator[]      ( const DimND<N>& indices ) requires ( ( N == 5 ) && is_bool<T>::value );
            inline T&                                   operator[]      ( const Idx index ) requires ( !is_bool<T>::value );
            inline T&                                   operator[]      ( const Idx index ) requires ( is_bool<T>::value );
            inline Tensor<T, N>&                        operator=       ( const Tensor<T, N>& other );
            template <typename U> inline Tensor<T, N>   operator+       ( const Tensor<U, N>& other ) const;
            template <typename U> inline Tensor<T, N>   operator-       ( const Tensor<U, N>& other ) const;
            template <typename U> inline Tensor<T, N>   operator*       ( const Tensor<U, N>& other ) const;
            template <typename U> inline Tensor<T, N>   operator/       ( const Tensor<U, N>& other ) const;
            template <typename U> inline Tensor<T, N>&  operator+=      ( const Tensor<U, N>& other );
            template <typename U> inline Tensor<T, N>&  operator-=      ( const Tensor<U, N>& other );
            template <typename U> inline Tensor<T, N>&  operator*=      ( const Tensor<U, N>& other );
            template <typename U> inline Tensor<T, N>&  operator/=      ( const Tensor<U, N>& other );
            template <typename U> inline Tensor<T, N>   operator+       ( const U& scalar ) const;
            template <typename U> inline Tensor<T, N>   operator-       ( const U& scalar ) const;
            template <typename U> inline Tensor<T, N>   operator*       ( const U& scalar ) const;
            template <typename U> inline Tensor<T, N>   operator/       ( const U& scalar ) const;
            template <typename U> inline Tensor<T, N>&  operator+=      ( const U& scalar );
            template <typename U> inline Tensor<T, N>&  operator-=      ( const U& scalar );
            template <typename U> inline Tensor<T, N>&  operator*=      ( const U& scalar );
            template <typename U> inline Tensor<T, N>&  operator/=      ( const U& scalar );
            template <typename U> inline bool           operator==      ( const Tensor<U, N>& other ) const;
            template <typename U> inline bool           operator<       ( const Tensor<U, N>& other ) const;
            template <typename U> inline bool           operator>       ( const Tensor<U, N>& other ) const;
            template <typename U> inline bool           operator<=      ( const Tensor<U, N>& other ) const;
            template <typename U> inline bool           operator>=      ( const Tensor<U, N>& other ) const;
            inline T                                    reduce_add      ();
            inline T                                    reduce_subtract ();
            inline T                                    reduce_multiply ();
            inline T                                    reduce_divide   ();
    };

    //: Operators
        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const DimND<N>& indices ) const requires ( N == 2 )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const DimND<N>& indices ) const requires ( N == 3 )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const DimND<N>& indices ) const requires ( N == 4 )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const DimND<N>& indices ) const requires ( N == 5 )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ], indices[ 4 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const Idx index ) const requires ( !is_bool<T>::value )
        {
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::operator[]( const Idx index ) const requires ( is_bool<T>::value )
        {
            return static_cast<bool>( data[ index ] );
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 2 ) && !is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 2 ) && is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ] );
            return reinterpret_cast<bool&>( data[ index ] );
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 3 ) && !is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 3 ) && is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ] );
            return reinterpret_cast<bool&>( data[ index ] );
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 4 ) && !is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 4 ) && is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ] );
            return reinterpret_cast<bool&>( data[ index ] );
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 5 ) && !is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ], indices[ 4 ] );
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const DimND<N>& indices ) requires ( ( N == 5 ) && is_bool<T>::value )
        {
            Idx index = this->shape.index( indices[ 0 ], indices[ 1 ], indices[ 2 ], indices[ 3 ], indices[ 4 ] );
            return reinterpret_cast<bool&>( data[ index ] );
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const Idx index ) requires ( !is_bool<T>::value )
        {
            return data[ index ];
        }

        template <typename T, u8 N>
        inline T& Tensor<T, N>::operator[]( const Idx index ) requires ( is_bool<T>::value )
        {
            return reinterpret_cast<bool&>( data[ index ] );
        }

        template <typename T, u8 N>
        inline Tensor<T, N>& Tensor<T, N>::operator=( const Tensor<T, N>& other )
        {
            this->shape = other.shape;
            this->data = other.data;
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator+( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( this->shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] + other.data[ i ];
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator-( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] - other.data[ i ];
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator*( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] * other.data[ i ];
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator/( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] / other.data[ i ];
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator+=( const Tensor<U, N>& other )
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] += other.data[ i ];
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator-=( const Tensor<U, N>& other )
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] -= other.data[ i ];
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator*=( const Tensor<U, N>& other )
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] *= other.data[ i ];
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator/=( const Tensor<U, N>& other )
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to add Tensors" );
            }

            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] /= other.data[ i ];
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator+( const U& scalar ) const
        {
            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] + scalar;
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator-( const U& scalar ) const
        {
            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] - scalar;
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator*( const U& scalar ) const
        {
            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] * scalar;
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N> Tensor<T, N>::operator/( const U& scalar ) const
        {
            Size size = data.size();

            // Add the data elements.
            Tensor<T, N> result_tensor( this->shape );
            for ( Idx i = 0; i < size; ++i )
            {
                result_tensor[ i ] = data[ i ] / scalar;
            }
            return result_tensor;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator+=( const U& scalar )
        {
            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] += scalar;
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator-=( const U& scalar )
        {
            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] -= scalar;
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator*=( const U& scalar )
        {
            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] *= scalar;
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline Tensor<T, N>& Tensor<T, N>::operator/=( const U& scalar )
        {
            Size size = data.size();

            // Add the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                data[ i ] /= scalar;
            }
            return *this;
        }

        template <typename T, u8 N>
        template <typename U>
        inline bool Tensor<T, N>::operator==( const Tensor<U, N>& other ) const
        {
            Size size = data.size();

            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to divide Tensors" );
            }

            bool comparison = true;
            // Compare the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                comparison = comparison && ( data[ i ] == other.data[ i ] );
            }

            return comparison;
        }

        template <typename T, u8 N>
        template <typename U>
        inline bool Tensor<T, N>::operator<( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to divide Tensors" );
            }

            Size size = data.size();

            bool comparison = true;
            // Compare the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                comparison = comparison && ( data[ i ] < other.data[ i ] );
            }

            return comparison;
        }

        template <typename T, u8 N>
        template <typename U>
        inline bool Tensor<T, N>::operator>( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to divide Tensors" );
            }

            Size size = data.size();

            bool comparison = true;
            // Compare the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                comparison = comparison && ( data[ i ] > other.data[ i ] );
            }

            return comparison;
        }

        template <typename T, u8 N>
        template <typename U>
        inline bool Tensor<T, N>::operator<=( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to divide Tensors" );
            }

            Size size = data.size();

            bool comparison = true;
            // Compare the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                comparison = comparison && ( data[ i ] <= other.data[ i ] );
            }

            return comparison;
        }

        template <typename T, u8 N>
        template <typename U>
        inline bool Tensor<T, N>::operator>=( const Tensor<U, N>& other ) const
        {
            // Check that the shapes are the same.
            if ( shape != other.shape )
            {
                throw std::invalid_argument( "Shapes must be the same to divide Tensors" );
            }

            Size size = data.size();

            bool comparison = true;
            // Compare the data elements.
            for ( Idx i = 0; i < size; ++i )
            {
                comparison = comparison && ( data[ i ] >= other.data[ i ] );
            }

            return comparison;
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::reduce_add()
        {
            Size size = data.size();

            T reduction( 0 );
            // Perform the operation.
            for ( Idx i = 0; i < size; ++i )
            {
                reduction += this->data[ i ];
            }
            return reduction;
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::reduce_subtract()
        {
            Size size = data.size();

            T reduction( 0 );
            // Perform the operation.
            for ( Idx i = 0; i < size; ++i )
            {
                reduction -= this->data[ i ];
            }
            return reduction;
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::reduce_multiply()
        {
            Size size = data.size();

            T reduction( 0 );
            // Perform the operation.
            for ( Idx i = 0; i < size; ++i )
            {
                reduction *= this->data[ i ];
            }
            return reduction;
        }

        template <typename T, u8 N>
        inline T Tensor<T, N>::reduce_divide()
        {
            Size size = data.size();

            T reduction( 0 );
            // Perform the operation.
            for ( Idx i = 0; i < size; ++i )
            {
                reduction /= this->data[ i ];
            }
            return reduction;
        }
} // namespace nn

//: Tensor Functions
namespace nn
{
    /*
    template <typename T, u8 N>
    constexpr TensorError element_wise_addition_strict( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] + a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 1>& c, const Tensor<T, 1>& b, const Tensor<T, 1>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim x = 0; x < c.width(); ++x )
        {
            c[ x ] = b[ x ] + a[ x ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 2>& c, const Tensor<T, 2>& b, const Tensor<T, 2>& a )
    {
        const Shape<2> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim y = 0; y < c.width(); ++y )
        {
            Idx y_idx = shape.height_index( y );

            for ( Dim x = 0; x < c.width(); ++x )
            {
                Idx x_idx = shape.width_index( y_idx, x );

                c[ x_idx ] = b[ x_idx ] + a[ x_idx ];
            }
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 3>& c, const Tensor<T, 3>& b, const Tensor<T, 3>& a )
    {
        const Shape<3> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim z = 0; z < c.depth(); ++z )
        {
            Idx z_idx = shape.depth_index( z );

            for ( Dim y = 0; y < c.width(); ++y )
            {
                Idx y_idx = shape.height_index( z_idx, y );

                for ( Dim x = 0; x < c.width(); ++x )
                {
                    Idx x_idx = shape.width_index( y_idx, x );

                    c[ x_idx ] = b[ x_idx ] + a[ x_idx ];
                }
            }
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 4>& c, const Tensor<T, 4>& b, const Tensor<T, 4>& a )
    {
        const Shape<4> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim c = 0; c < c.channels(); ++c )
        {
            Idx c_idx = shape.channel_index( c );

            for ( Dim z = 0; z < c.depth(); ++z )
            {
                Idx z_idx = shape.depth_index( c_idx, z );

                for ( Dim y = 0; y < c.width(); ++y )
                {
                    Idx y_idx = shape.height_index( z_idx, y );

                    for ( Dim x = 0; x < c.width(); ++x )
                    {
                        Idx x_idx = shape.width_index( y_idx, x );

                        c[ x_idx ] = b[ x_idx ] + a[ x_idx ];
                    }
                }
            }
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 5>& c, const Tensor<T, 4>& b, const Tensor<T, 5>& a )
    {
        const Shape<5> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim b = 0; b < c.channels(); ++b )
        {
            Idx b_idx = shape.channel_index( b );

            for ( Dim c = 0; c < c.channels(); ++c )
            {
                Idx c_idx = shape.channel_index( b_idx, c );

                for ( Dim z = 0; z < c.depth(); ++z )
                {
                    Idx z_idx = shape.depth_index( c_idx, z );

                    for ( Dim y = 0; y < c.width(); ++y )
                    {
                        Idx y_idx = shape.height_index( z_idx, y );

                        for ( Dim x = 0; x < c.width(); ++x )
                        {
                            Idx x_idx = shape.width_index( y_idx, x );

                            c[ x_idx ] = b[ x_idx ] + a[ x_idx ];
                        }
                    }
                }
            }
        }

        return TensorError::NO_ERROR;
    }

    template <typename T>
    constexpr TensorError element_wise_addition( Tensor<T, 5>& c, const Tensor<T, 5>& b, const Tensor<T, 5>& a )
    {
        const Shape<5> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Dim b = 0; b < c.channels(); ++b )
        {
            Idx b_idx = shape.channel_index( b );

            for ( Dim c = 0; c < c.channels(); ++c )
            {
                Idx c_idx = shape.channel_index( b_idx, c );

                for ( Dim z = 0; z < c.depth(); ++z )
                {
                    Idx z_idx = shape.depth_index( c_idx, z );

                    for ( Dim y = 0; y < c.width(); ++y )
                    {
                        Idx y_idx = shape.height_index( z_idx, y );

                        for ( Dim x = 0; x < c.width(); ++x )
                        {
                            Idx x_idx = shape.width_index( y_idx, x );

                            c[ x_idx ] = b[ x_idx ] + a[ x_idx ];
                        }
                    }
                }
            }
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr Tensor<T, 1> element_wise_addition( const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );

        Tensor<T, 1> c( shape );
        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] + a[ index ];
        }

        return c;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_subtraction_strict( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] - a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_subtraction( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] - a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr Tensor<T, 1> element_wise_subtraction( const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );

        Tensor<T, 1> c( shape );
        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] - a[ index ];
        }

        return c;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_multiplication_strict( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] * a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_multiplication( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < shape.volume(); ++index )
        {
            c[ index ] = b[ index ] * a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr Tensor<T, 1> element_wise_multiplication( const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );

        Tensor<T, 1> c( shape );
        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] * a[ index ];
        }

        return c;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_division_strict( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] / a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_division( Tensor<T, N>& c, const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );
        if ( shape != c.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] / a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr Tensor<T, 1> element_wise_division( const Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        const Shape<1> shape = min( a.get_shape(), b.get_shape() );

        Tensor<T, 1> c( shape );
        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            c[ index ] = b[ index ] / a[ index ];
        }

        return c;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_addition_assignment_strict( Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            b[ index ] += a[ index ];
        }

        return TensorError::NO_ERROR;
    }

    template <typename T, u8 N>
    constexpr TensorError element_wise_addition_assignment( Tensor<T, N>& b, const Tensor<T, N>& a )
    {
        if ( c.get_shape() != a.get_shape() || c.get_shape() != b.get_shape() ) { return TensorError::MISMATCHED_SHAPES; }
        const Shape<N> shape = min( a.get_shape(), b.get_shape() );

        for ( Idx index = 0; index < a.get_size(); ++index )
        {
            b[ index ] += a[ index ];
        }

        return TensorError::NO_ERROR;
    }
    */
}

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_TENSOR_HPP_
