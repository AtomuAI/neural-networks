// Copyright 2024 Shane W. Mulcahy

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

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/dim.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <u8 N>
        Shape<N>::Shape( const DimND<N>& shape ) : dim( shape ), size( N ) {}
        template <u8 N>
        Shape<N>::Shape( const Dim width ) requires ( N == 1 ) : dim({ width }), size( width ) {}
        template <u8 N>
        Shape<N>::Shape( const Dim width, const Dim height ) requires ( N == 2 ) : dim({ width, height }), size( width * height ) {}
        template <u8 N>
        Shape<N>::Shape( const Dim width, const Dim height, const Dim depth ) requires ( N == 3 ) : dim({ width, height, depth }), size( width * height * depth ) {}
        template <u8 N>
        Shape<N>::Shape( const Dim width, const Dim height, const Dim depth, const Dim channels ) requires ( N == 4 ) : dim({ width, height, depth, channels }), size( width * height * depth * channels ) {}
        template <u8 N>
        Shape<N>::Shape( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches ) requires ( N == 5 ) : dim({ width, height, depth, channels, batches }), size( width * height * depth * channels * batches ) {}

    //: Destructors
        template <u8 N>
        Shape<N>::~Shape() {}

    //: Methods
        template <u8 N>
        void Shape<N>::resize( const std::array<Dim, N> dim )
        {
            this->dim = dim;
            this->size = std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() );
        }

        template <u8 N>
        void Shape<N>::resize( const Dim width ) requires ( N == 1 )
        {
            this->width() = width;
            this->size = width;
        }

        template <u8 N>
        void Shape<N>::resize( const Dim width, const Dim height ) requires ( N == 2 )
        {
            this->width() = width;
            this->height() = height;
            this->size = width * height;
        }

        template <u8 N>
        void Shape<N>::resize( const Dim width, const Dim height, const Dim depth ) requires ( N == 3 )
        {
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
            this->size = width * height * depth;
        }

        template <u8 N>
        void Shape<N>::resize( const Dim width, const Dim height, const Dim depth, const Dim channels ) requires ( N == 4 )
        {
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
            this->channels() = channels;
            this->size = width * height * depth * channels;
        }

        template <u8 N>
        void Shape<N>::resize( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches ) requires ( N == 5 )
        {
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
            this->channels() = channels;
            this->batches() = batches;
            this->size = width * height * depth * channels * batches;
        }

        template <u8 N>
        void Shape<N>::reshape( const std::array<Dim, N> dim ) requires ( ( N >= 2 ) && ( N <= 5 ) )
        {
            if ( std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() ) != this->volume() ) { return; }
            this->dim = dim;
        }

        template <u8 N>
        void Shape<N>::reshape( const Dim width, const Dim height ) requires ( N == 2 )
        {
            std::array<Dim, 2> dim = { width, height };
            if ( std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() ) != this->volume() ) { return; }
            this->width() = width;
            this->height() = height;
        }

        template <u8 N>
        void Shape<N>::reshape( const Dim width, const Dim height, const Dim depth ) requires ( N == 3 )
        {
            std::array<Dim, 3> dim = { width, height, depth };
            if ( std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() ) != this->volume() ) { return; }
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
        }

        template <u8 N>
        void Shape<N>::reshape( const Dim width, const Dim height, const Dim depth, const Dim channels ) requires ( N == 4 )
        {
            std::array<Dim, 4> dim = { width, height, depth, channels };
            if ( std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() ) != this->volume() ) { return; }
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
            this->channels() = channels;
        }

        template <u8 N>
        void Shape<N>::reshape( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches ) requires ( N == 5 )
        {
            std::array<Dim, 5> dim = { width, height, depth, channels, batches };
            if ( std::accumulate( dim.begin(), dim.end(), 1, std::multiplies<Dim>() ) != this->volume() ) { return; }
            this->width() = width;
            this->height() = height;
            this->depth() = depth;
            this->channels() = channels;
            this->batches() = batches;
        }
} // namespace nn

//: Specializations
namespace nn
{
    template class Shape<1>;
    template class Shape<2>;
    template class Shape<3>;
    template class Shape<4>;
    template class Shape<5>;
} // namespace nn
