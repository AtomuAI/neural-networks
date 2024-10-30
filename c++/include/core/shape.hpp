// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_SHAPE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_SHAPE_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <functional>
#include <utility>

//: Type Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/dim.hpp"

namespace nn
{
    template <u8 N>
    class Shape;

    using Shape1D = Shape<1>;
    using Shape2D = Shape<2>;
    using Shape3D = Shape<3>;
    using Shape4D = Shape<4>;
    using Shape5D = Shape<5>;

    template <u8 N>
    class Shape
    {
        //: Members
        public:
            static const u8     dimensionality = N;
        private:
            DimND<N>    dim;
            Size        size;

        //: Constructors
        public:
            explicit    Shape   ( const DimND<N>& shape );
            explicit    Shape   ( const Dim width = 0 )                                                                                             requires ( N == 1 );
            explicit    Shape   ( const Dim width = 0, const Dim height = 1 )                                                                       requires ( N == 2 );
            explicit    Shape   ( const Dim width = 0, const Dim height = 1, const Dim depth = 1 )                                                  requires ( N == 3 );
            explicit    Shape   ( const Dim width = 0, const Dim height = 1, const Dim depth = 1, const Dim channels = 1 )                          requires ( N == 4 );
            explicit    Shape   ( const Dim width = 0, const Dim height = 1, const Dim depth = 1, const Dim channels = 1, const Dim batches = 1 )   requires ( N == 5 );

        //: Destructors
        public:
            virtual     ~Shape();

        //: Operators
        public:
            inline Shape<N>&   operator=   ( const Shape<N>& other );
            inline bool        operator==  ( const Shape<N>& other ) const;
            inline bool        operator!=  ( const Shape<N>& other ) const;
            inline bool        operator<   ( const Shape<N>& other ) const;
            inline bool        operator>   ( const Shape<N>& other ) const;
            inline bool        operator<=  ( const Shape<N>& other ) const;
            inline bool        operator>=  ( const Shape<N>& other ) const;

        //: Methods
        public:
            void        resize          ( const Dim width )                                                                                 requires ( N == 1 );
            void        resize          ( const Dim width, const Dim height )                                                               requires ( N == 2 );
            void        resize          ( const Dim width, const Dim height, const Dim depth )                                              requires ( N == 3 );
            void        resize          ( const Dim width, const Dim height, const Dim depth, const Dim channels )                          requires ( N == 4 );
            void        resize          ( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches )       requires ( N == 5 );
            void        reshape         ( const Dim width, const Dim height )                                                               requires ( N == 2 );
            void        reshape         ( const Dim width, const Dim height, const Dim depth )                                              requires ( N == 3 );
            void        reshape         ( const Dim width, const Dim height, const Dim depth, const Dim channels )                          requires ( N == 4 );
            void        reshape         ( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches )       requires ( N == 5 );
            inline bool equal           ( const Dim width ) const                                                                           requires ( N == 1 );
            inline bool equal           ( const Dim width, const Dim height ) const                                                         requires ( N == 2 );
            inline bool equal           ( const Dim width, const Dim height, const Dim depth ) const                                        requires ( N == 3 );
            inline bool equal           ( const Dim width, const Dim height, const Dim depth, const Dim channels ) const                    requires ( N == 4 );
            inline bool equal           ( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches ) const requires ( N == 5 );
            inline Idx  width_index     ( const Idx height_index, const Dim x ) const                                                       requires ( ( N >= 2 ) && ( N <= 5 ) );
            inline Idx  height_index    ( const Idx depth_index, const Dim y ) const                                                        requires ( ( N >= 3 ) && ( N <= 5 ) );
            inline Idx  depth_index     ( const Idx channel_index, const Dim z ) const                                                      requires ( ( N >= 4 ) && ( N <= 5 ) );
            inline Idx  channel_index   ( const Idx batch_index, const Dim c ) const                                                        requires ( ( N == 5 ) );
            inline Idx  height_index    ( const Dim y ) const                                                                               requires ( N == 2 );
            inline Idx  depth_index     ( const Dim z ) const                                                                               requires ( N == 3 );
            inline Idx  channel_index   ( const Dim c ) const                                                                               requires ( N == 4 );
            inline Idx  batch_index     ( const Dim b ) const                                                                               requires ( N == 5 );
            inline Idx  index           ( const Dim x, const Dim y ) const                                                                  requires ( N == 2 );
            inline Idx  index           ( const Dim x, const Dim y, const Dim z ) const                                                     requires ( N == 3 );
            inline Idx  index           ( const Dim x, const Dim y, const Dim z, const Dim c ) const                                        requires ( N == 4 );
            inline Idx  index           ( const Dim x, const Dim y, const Dim z, const Dim c, const Dim b ) const                           requires ( N == 5 );
            inline bool within_width    ( const Dim x ) const                                                                               requires ( ( N >= 1 ) && ( N <= 5 ) );
            inline bool within_height   ( const Dim y ) const                                                                               requires ( ( N >= 2 ) && ( N <= 5 ) );
            inline bool within_depth    ( const Dim z ) const                                                                               requires ( ( N >= 3 ) && ( N <= 5 ) );
            inline bool within_channels ( const Dim c ) const                                                                               requires ( ( N == 4 ) || ( N == 5 ) );
            inline bool within_batches  ( const Dim b ) const                                                                               requires ( N == 5 );
            inline Dim  width           () const                                                                                            requires ( ( N >= 1 ) && ( N <= 5 ) );
            inline Dim  height          () const                                                                                            requires ( ( N >= 2 ) && ( N <= 5 ) );
            inline Dim  depth           () const                                                                                            requires ( ( N >= 3 ) && ( N <= 5 ) );
            inline Dim  channels        () const                                                                                            requires ( ( N == 4 ) || ( N == 5 ) );
            inline Dim  batches         () const                                                                                            requires ( N == 5 );
            inline Size volume          () const;
            inline Size distance        ( const u8 begin, const u8 end ) const;
        private:
            void        resize          ( const std::array<Dim, N> dims );
            void        reshape         ( const std::array<Dim, N> dims )                                                                   requires ( ( N >= 2 ) && ( N <= 5 ) );
            inline Dim& width           ()                                                                                                  requires ( ( N >= 1 ) && ( N <= 5 ) );
            inline Dim& height          ()                                                                                                  requires ( ( N >= 2 ) && ( N <= 5 ) );
            inline Dim& depth           ()                                                                                                  requires ( ( N >= 3 ) && ( N <= 5 ) );
            inline Dim& channels        ()                                                                                                  requires ( ( N == 4 ) || ( N == 5 ) );
            inline Dim& batches         ()                                                                                                  requires ( N == 5 );

        private:
            template <typename T, u8 O>
            friend class Tensor;
    };

    //: Inline Operators
        template <u8 N>
        inline Shape<N>& Shape<N>::operator=( const Shape<N>& other )
        {
            this->dim = other.dim;
            return *this;
        }

        template <u8 N>
        inline bool Shape<N>::operator==( const Shape<N>& other ) const
        {
            return this->dim == other.dim;
        }

        template <u8 N>
        inline bool Shape<N>::operator!=( const Shape<N>& other ) const
        {
            return this->dim < other.dim;
        }

        template <u8 N>
        inline bool Shape<N>::operator>( const Shape<N>& other ) const
        {
            return this->dim > other.dim;
        }

        template <u8 N>
        inline bool Shape<N>::operator<=( const Shape<N>& other ) const
        {
            return this->dim <= other.dim;
        }

        template <u8 N>
        inline bool Shape<N>::operator>=( const Shape<N>& other ) const
        {
            return this->dim >= other.dim;
        }

    //: Inline Methods
        template <u8 N>
        inline bool Shape<N>::equal( const Dim width ) const requires ( N == 1 )
        {
            return ( this->width() == width );
        }
        template <u8 N>
        inline bool Shape<N>::equal( const Dim width, const Dim height ) const requires ( N == 2 )
        {
            return ( ( this->width() == width ) && ( this->height() == height ) );
        }
        template <u8 N>
        inline bool Shape<N>::equal( const Dim width, const Dim height, const Dim depth ) const requires ( N == 3 )
        {
            return ( ( this->width() == width ) && ( this->height() == height ) && ( this->depth() == depth ) );
        }
        template <u8 N>
        inline bool Shape<N>::equal( const Dim width, const Dim height, const Dim depth, const Dim channels ) const requires ( N == 4 )
        {
            return ( ( this->width() == width ) && ( this->height() == height ) && ( this->depth() == depth ) && ( this->channels() == channels ) );
        }
        template <u8 N>
        inline bool Shape<N>::equal( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches ) const requires ( N == 5 )
        {
            return ( ( this->width() == width ) && ( this->height() == height ) && ( this->depth() == depth ) && ( this->channels() == channels ) && ( this->batches() == batches )  );
        }
        template <u8 N>
        inline Idx Shape<N>::width_index( const Idx height_index, const Dim x ) const requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            return ( height_index + x );
        }
        template <u8 N>
        inline Idx Shape<N>::height_index( const Idx depth_index, const Dim y ) const requires ( ( N >= 3 ) && ( N <= 5 ) )
        {
            return ( ( depth_index + y ) * this->width() );
        }
        template <u8 N>
        inline Idx Shape<N>::depth_index( const Idx channel_index, const Dim z ) const requires ( ( N >= 4 ) && ( N <= 5 ) )
        {
            return ( ( channel_index + z ) * this->height() );
        }
        template <u8 N>
        inline Idx Shape<N>::channel_index( const Idx batch_index, const Dim c ) const requires ( ( N == 5 ) )
        {
            return ( ( batch_index + c ) * this->depth() );
        }
        template <u8 N>
        inline Idx Shape<N>::height_index( const Dim y ) const requires ( N == 2 )
        {
            return ( y * this->width() );
        }
        template <u8 N>
        inline Idx Shape<N>::depth_index( const Dim z ) const requires ( N == 3 )
        {
            return ( z * this->height() );
        }
        template <u8 N>
        inline Idx Shape<N>::channel_index( const Dim c ) const requires ( N == 4 )
        {
            return ( c * this->depth() );
        }
        template <u8 N>
        inline Idx Shape<N>::batch_index( const Dim b ) const requires ( N == 5 )
        {
            return ( b * this->channels() );
        }
        template <u8 N>
        inline Idx Shape<N>::index( const Dim x, const Dim y ) const requires ( N == 2 )
        {
            return this->width_index( this->height_index( y ), x );
        }
        template <u8 N>
        inline Idx Shape<N>::index( const Dim x, const Dim y, const Dim z ) const requires ( N == 3 )
        {
            return this->width_index( this->height_index( this->depth_index( z ), y ), x );
        }
        template <u8 N>
        inline Idx Shape<N>::index( const Dim x, const Dim y, const Dim z, const Dim c ) const requires ( N == 4 )
        {
            return this->width_index( this->height_index( this->depth_index( this->channel_index( c ), z ), y ), x );
        }
        template <u8 N>
        inline Idx Shape<N>::index( const Dim x, const Dim y, const Dim z, const Dim c, const Dim b ) const requires ( N == 5 )
        {
            return this->width_index( this->height_index( this->depth_index( this->channel_index( this->batch_index( b ), c ), z ), y ), x );
        }
        template <u8 N>
        inline bool Shape<N>::within_width( const Dim x ) const requires ( ( N >= 1 ) && ( N <= 5 ) )
        {
            return ( x >= 0 ) && ( x < this->width() );
        }
        template <u8 N>
        inline bool Shape<N>::within_height( const Dim y ) const requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            return ( y >= 0 ) && ( y < this->height() );
        }
        template <u8 N>
        inline bool Shape<N>::within_depth( const Dim z ) const requires ( ( N >= 3 ) && ( N <= 5 ) )
        {
            return ( z >= 0 ) && ( z < this->depth() );
        }
        template <u8 N>
        inline bool Shape<N>::within_channels( const Dim c ) const requires ( ( N == 4 ) || ( N == 5 ) )
        {
            return ( c >= 0 ) && ( c < this->channels() );
        }
        template <u8 N>
        inline bool Shape<N>::within_batches( const Dim b ) const requires ( N == 5 )
        {
            return ( b >= 0 ) && ( b < this->batches() );
        }
        template <u8 N>
        inline Dim Shape<N>::width() const requires ( ( N >= 1 ) && ( N <= 5 ) )
        {
            return this->dim[ 0 ];
        }
        template <u8 N>
        inline Dim Shape<N>::height() const requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            return this->dim[ 1 ];
        }
        template <u8 N>
        inline Dim Shape<N>::depth() const requires ( ( N >= 3 ) && ( N <= 5 ) )
        {
            return this->dim[ 2 ];
        }
        template <u8 N>
        inline Dim Shape<N>::channels() const requires ( ( N == 4 ) || ( N == 5 ) )
        {
            return this->dim[ 3 ];
        }
        template <u8 N>
        inline Dim Shape<N>::batches() const requires ( N == 5 )
        {
            return this->dim[ 4 ];
        }
        template <u8 N>
        inline Size Shape<N>::volume() const
        {
            return this->size;
        }
        template <u8 N>
        inline Size Shape<N>::distance( const u8 begin, const u8 end ) const
        {
            if ( end > N ) { throw std::out_of_range( "Shape<" + std::to_string( N ) + ">::distance: end is out of range" ); }
            return std::accumulate( this->dim.begin() + begin, this->dim.begin() + end, 1, std::multiplies<Dim>() );
        }
        template <u8 N>
        inline Dim& Shape<N>::width() requires ( ( N >= 1 ) && ( N <= 5 ) )
        {
            return this->dim[ 0 ];
        }
        template <u8 N>
        inline Dim& Shape<N>::height() requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            return this->dim[ 1 ];
        }
        template <u8 N>
        inline Dim& Shape<N>::depth() requires ( ( N >= 3 ) && ( N <= 5 ) )
        {
            return this->dim[ 2 ];
        }
        template <u8 N>
        inline Dim& Shape<N>::channels() requires ( ( N == 4 ) || ( N == 5 ) )
        {
            return this->dim[ 3 ];
        }
        template <u8 N>
        inline Dim& Shape<N>::batches() requires ( N == 5 )
        {
            return this->dim[ 4 ];
        }
} // namespace nn

//: Shape Functions
namespace nn
{
    constexpr Shape<1> min(const Shape<1>& shape_a, const Shape<1>& shape_b)
    {
        Shape<1> shape_c
        (
            std::min( shape_a.width(), shape_b.width() )
        );
        return shape_c;
    }

    constexpr Shape<2> min(const Shape<1>& shape_a, const Shape<2>& shape_b)
    {
        Shape<2> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<3> min(const Shape<1>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<1>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<1>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<2> min(const Shape<2>& shape_a, const Shape<1>& shape_b)
    {
        Shape<2> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<2> min(const Shape<2>& shape_a, const Shape<2>& shape_b)
    {
        Shape<2> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() )
        );
        return shape_c;
    }

    constexpr Shape<3> min(const Shape<2>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<2>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<2>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<3> min(const Shape<3>& shape_a, const Shape<1>& shape_b)
    {
        Shape<3> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<3> min(const Shape<3>& shape_a, const Shape<2>& shape_b)
    {
        Shape<3> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<3> min(const Shape<3>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() )
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<3>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<3>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<4>& shape_a, const Shape<1>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<4>& shape_a, const Shape<2>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<4>& shape_a, const Shape<3>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<4> min(const Shape<4>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            std::min( shape_a.channels(), shape_b.channels() )
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<4>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            std::min( shape_a.channels(), shape_b.channels() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<5>& shape_a, const Shape<1>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            1,
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<5>& shape_a, const Shape<2>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            1,
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<5>& shape_a, const Shape<3>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            1,
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<5>& shape_a, const Shape<4>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            std::min( shape_a.channels(), shape_b.channels() ),
            1
        );
        return shape_c;
    }

    constexpr Shape<5> min(const Shape<5>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::min( shape_a.width(), shape_b.width() ),
            std::min( shape_a.height(), shape_b.height() ),
            std::min( shape_a.depth(), shape_b.depth() ),
            std::min( shape_a.channels(), shape_b.channels() ),
            std::min( shape_a.batches(), shape_b.batches() )
        );
        return shape_c;
    }

    constexpr Shape<1> max(const Shape<1>& shape_a, const Shape<1>& shape_b)
    {
        Shape<1> shape_c
        (
            std::max( shape_a.width(), shape_b.width() )
        );
        return shape_c;
    }

    constexpr Shape<2> max(const Shape<1>& shape_a, const Shape<2>& shape_b)
    {
        Shape<2> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_b.height()
        );
        return shape_c;
    }

    constexpr Shape<3> max(const Shape<1>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_b.height(),
            shape_b.depth()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<1>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_b.height(),
            shape_b.depth(),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<1>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_b.height(),
            shape_b.depth(),
            shape_b.channels(),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<2> max(const Shape<2>& shape_a, const Shape<1>& shape_b)
    {
        Shape<2> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_a.height()
        );
        return shape_c;
    }

    constexpr Shape<2> max(const Shape<2>& shape_a, const Shape<2>& shape_b)
    {
        Shape<2> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() )
        );
        return shape_c;
    }

    constexpr Shape<3> max(const Shape<2>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_b.depth()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<2>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_b.depth(),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<2>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_b.depth(),
            shape_b.channels(),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<3> max(const Shape<3>& shape_a, const Shape<1>& shape_b)
    {
        Shape<3> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_a.height(),
            shape_a.depth()
        );
        return shape_c;
    }

    constexpr Shape<3> max(const Shape<3>& shape_a, const Shape<2>& shape_b)
    {
        Shape<3> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_a.depth()
        );
        return shape_c;
    }

    constexpr Shape<3> max(const Shape<3>& shape_a, const Shape<3>& shape_b)
    {
        Shape<3> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() )
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<3>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<3>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            shape_b.channels(),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<4>& shape_a, const Shape<1>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_a.height(),
            shape_a.depth(),
            shape_a.channels()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<4>& shape_a, const Shape<2>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_a.depth(),
            shape_a.channels()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<4>& shape_a, const Shape<3>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            shape_a.channels()
        );
        return shape_c;
    }

    constexpr Shape<4> max(const Shape<4>& shape_a, const Shape<4>& shape_b)
    {
        Shape<4> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            std::max( shape_a.channels(), shape_b.channels() )
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<4>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            std::max( shape_a.channels(), shape_b.channels() ),
            shape_b.channels()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<5>& shape_a, const Shape<1>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            shape_a.height(),
            shape_a.depth(),
            shape_a.channels(),
            shape_a.batches()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<5>& shape_a, const Shape<2>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            shape_a.depth(),
            shape_a.channels(),
            shape_a.batches()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<5>& shape_a, const Shape<3>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            shape_a.channels(),
            shape_a.batches()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<5>& shape_a, const Shape<4>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            std::max( shape_a.channels(), shape_b.channels() ),
            shape_a.batches()
        );
        return shape_c;
    }

    constexpr Shape<5> max(const Shape<5>& shape_a, const Shape<5>& shape_b)
    {
        Shape<5> shape_c
        (
            std::max( shape_a.width(), shape_b.width() ),
            std::max( shape_a.height(), shape_b.height() ),
            std::max( shape_a.depth(), shape_b.depth() ),
            std::max( shape_a.channels(), shape_b.channels() ),
            std::max( shape_a.batches(), shape_b.batches() )
        );
        return shape_c;
    }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_SHAPE_HPP_
