// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COST_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COST_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    template <typename U, typename V>
    U mean_squared_error                                ( const V target, const U value );
    template <typename U, typename V>
    U categorical_cross_entropy                         ( const V target, const U value );
    template <typename U, typename V>
    U hellinger_distance                                ( const V target, const U value );
    template <typename U, typename V>
    U kullback_leibler_divergence                       ( const V target, const U value );
    template <typename U, typename V>
    U generalized_kullback_leibler_divergence           ( const V target, const U value );
    template <typename U, typename V>
    U itakura_saito_distance                            ( const V target, const U value );
    template <typename U, typename V>
    U mean_squared_error_derivative                     ( const V target, const U value );
    template <typename U, typename V>
    U categorical_cross_entropy_derivative              ( const V target, const U value );
    /*
    template <typename U, typename V>
    U softmax_categorical_cross_entropy_derivative      ( const V target, const U value );
    */
    template <typename U, typename V>
    U hellinger_distance_derivative                     ( const V target, const U value );
    template <typename U, typename V>
    U kullback_leibler_divergence_derivative            ( const V target, const U value );
    template <typename U, typename V>
    U generalized_kullback_leibler_divergence_derivative( const V target, const U value );
    template <typename U, typename V>
    U itakura_saito_distance_derivative                 ( const V target, const U value );

    template <typename U, typename V>
    U mean_squared_error( const V target, const U value )
    {
        return pow( ( target - value ), 2 );
    };

    template <typename U, typename V>
    U categorical_cross_entropy( const V target, const U value )
    {
        return target * ( ( value > 0 ) ? log( value ) : log( 1e-8 ) );
    };

    template <typename U, typename V>
    U hellinger_distance( const V target, const U value )
    {
        return 0;
    };

    template <typename U, typename V>
    U kullback_leibler_divergence( const V target, const U value )
    {
        return 0;
    };

    template <typename U, typename V>
    U generalized_kullback_leibler_divergence( const V target, const U value )
    {
        return 0;
    };

    template <typename U, typename V>
    U itakura_saito_distance( const V target, const U value )
    {
        return 0;
    };

    template <typename U, typename V>
    U mean_squared_error_derivative( const V target, const U value )
    {
        return ( target - value );
    };

    template <typename U, typename V>
    U categorical_cross_entropy_derivative( const V target, const U value )
    {
        return -( target * ( ( value > 0 ) ? log( value ) : log( 1e-8 ) ) );
    };

    /*
    template <typename U, typename V>
    U softmax_categorical_cross_entropy_derivative( const V target, const U value )
    {
        return ( target - value );
    };
    */

    template <typename U, typename V>
    U hellinger_distance_derivative( const V target, const U value )
    {
        return ( ( sqrt( target ) - sqrt( value ) ) / ( /*sqrt(2)*/( 1.4142135623730950*sqrt( target ) ) + 1e-8 ) );
    };

    template <typename U, typename V>
    U kullback_leibler_divergence_derivative( const V target, const U value )
    {
        return -( value / ( target + 1e-8 ) );
    };

    template <typename U, typename V>
    U generalized_kullback_leibler_divergence_derivative( const V target, const U value )
    {
        return ( ( target - value ) / ( target + 1e-8 ) );
    };

    template <typename U, typename V>
    U itakura_saito_distance_derivative( const V target, const U value )
    {
        return ( ( target - value ) / ( pow( target, 2 ) + 1e-8 ) );
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COST_HPP_
