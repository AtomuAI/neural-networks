// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ACTIVATION_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ACTIVATION_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

#include "bewusstsein_neural_networks/c++/include/core/math.hpp"

namespace nn
{
    //: Sigmoid
        template <typename T>               inline T                sigmoid             ( const T value )                                                       requires ( is_int<T>::value );
        template <typename T>               inline T                sigmoid             ( const T value )                                                       requires ( is_float<T>::value );
        template <typename T>               inline std::complex<T>  sigmoid             ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  separable_sigmoid   ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline T                d_sigmoid           ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_sigmoid           ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_separable_sigmoid ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );

    //: HTan
        template <typename T>               inline T                htan                ( const T value )                                                       requires ( is_unsigned_int<T>::value );
        template <typename T>               inline T                htan                ( const T value )                                                       requires ( is_signed_int<T>::value );
        template <typename T>               inline T                htan                ( const T value )                                                       requires ( is_float<T>::value );
        template <typename T>               inline std::complex<T>  htan                ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  separable_htan      ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline T                d_htan              ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_htan              ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_separable_htan    ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );

    //: Relu
        template <typename T>               inline T                relu                ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  relu                ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline T                d_relu              ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_relu              ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );

    //: Leaky Relu
        template <typename T, typename U>   inline T                leaky_relu          ( const U leak_coefficient, const T value )                             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  leaky_relu          ( const std::complex<U> leak_coefficient, const T value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  leaky_relu          ( const U leak_coefficient, const std::complex<T> value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  leaky_relu          ( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline T                d_leaky_relu        ( const U leak_coefficient, const T value )                             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_leaky_relu        ( const std::complex<U> leak_coefficient, const T value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_leaky_relu        ( const U leak_coefficient, const std::complex<T> value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_leaky_relu        ( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value );

    //: Elu
        template <typename T, typename U>   inline T                elu                 ( const U leak_coefficient, const T value )                             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  elu                 ( const std::complex<U> leak_coefficient, const T value )               requires ( is_int<T>::value || is_float<T>::value);
        template <typename T, typename U>   inline std::complex<T>  elu                 ( const U leak_coefficient, const std::complex<T> value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  elu                 ( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline T                d_elu               ( const U leak_coefficient, const T value )                             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_elu               ( const std::complex<U> leak_coefficient, const T value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_elu               ( const U leak_coefficient, const std::complex<T> value )               requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_elu               ( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value );

    //: Swish
        template <typename T>               inline T                swish               ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  swish               ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  separable_swish     ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline T                d_swish             ( const T value )                                                       requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_swish             ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T>               inline std::complex<T>  d_separable_swish   ( const std::complex<T> value )                                         requires ( is_int<T>::value || is_float<T>::value );

    //: ESwish
        template <typename T, typename U>   inline T                eswish              ( const U beta, const T value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  eswish              ( const std::complex<U> beta, const T value )                           requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  eswish              ( const U beta, const std::complex<T> value )                           requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  eswish              ( const std::complex<U> beta, const std::complex<T> value )             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  separable_eswish    ( const U beta, const std::complex<T> value )                           requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline T                d_eswish            ( const U beta, const T value )                                         requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_eswish            ( const std::complex<U> beta, const T value )                           requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_eswish            ( const U beta, const std::complex<T> value )                           requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_eswish            ( const std::complex<U> beta, const std::complex<T> value )             requires ( is_int<T>::value || is_float<T>::value );
        template <typename T, typename U>   inline std::complex<T>  d_separable_eswish  ( const U beta, const std::complex<T> value )                           requires ( is_int<T>::value || is_float<T>::value );

    //: Sigmoid
        template <typename T>
        inline T sigmoid(const T value) requires ( is_int<T>::value )
        {
            return ( value >= 0 ) ? 1 : 0;
        }

        template <typename T>
        inline T sigmoid( const T value ) requires ( is_float<T>::value )
        {
            return 1 / ( (1) + nn::exp( -value ) );
        }

        template <typename T>
        inline std::complex<T> sigmoid( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return 1 / ( 1 + nn::exp( -value ) );
        }

        template <typename T>
        inline std::complex<T> separable_sigmoid( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( sigmoid( value.real() ), sigmoid( value.imag() ) );
        }

        template <typename T>
        inline T d_sigmoid( const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            T s = sigmoid( value );
            return s * ( 1 - s );
        }

        template <typename T>
        inline std::complex<T> d_sigmoid( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            T s = sigmoid( value );
            return s * ( 1 - s );
        }

        template <typename T>
        inline std::complex<T> d_separable_sigmoid( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_sigmoid( value.real() ), d_sigmoid( value.imag() ) );
        }

    //: HTan
        template <typename T>
        inline T htan( const T value ) requires ( is_unsigned_int<T>::value )
        {
            return ( value >= 1 ) ? 1 : 0;
        }
        template <typename T>
        inline T htan( const T value ) requires ( is_signed_int<T>::value )
        {
            if      ( value > 0 )   { return 1; }
            else if ( value < 0 )   { return -1; }
            else                    { return 0; }
        }
        template <typename T>
        inline T htan( const T value ) requires ( is_float<T>::value )
        {
            return nn::tanh( value );
        }
        template <typename T>
        inline std::complex<T> htan( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return nn::tanh( value );
        }
        template <typename T>
        inline std::complex<T> separable_htan( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( htan( value.real() ), htan( value.imag() ) );
        }

        template <typename T>
        inline T d_htan( const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return 1 - nn::pow( nn::tanh( value ), 2 );
        }
        template <typename T>
        inline std::complex<T> d_htan( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return 1 - nn::pow( nn::tanh( value ), 2 );
        }
        template <typename T>
        inline std::complex<T> d_separable_htan( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_htan( value.real() ), d_htan( value.imag() ) );
        }

    //: Relu
        template <typename T>
        inline T relu(const T value) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value < 0 ) ? 0 : value;
        }
        template <typename T>
        inline std::complex<T> relu(const std::complex<T> value) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( relu( value.real() ), relu( value.imag() ) );
        }

        template <typename T>
        inline T d_relu( const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value > 0 ) ? 1 : 0;
        }
        template <typename T>
        inline std::complex<T> d_relu( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_relu( value.real() ), d_relu( value.imag() ) );
        }

    //: Leaky Relu
        template <typename T, typename U>
        inline T leaky_relu( const U leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value < 0 ) ? ( leak_coefficient * value ) : value;
        }
        template <typename T, typename U>
        inline std::complex<T> leaky_relu( const std::complex<U> leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value < 0 ) ? ( leak_coefficient * value ) : value;
        }
        template <typename T, typename U>
        inline std::complex<T> leaky_relu( const U leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( leaky_relu( leak_coefficient, value.real() ), leaky_relu( leak_coefficient, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> leaky_relu( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( leaky_relu( leak_coefficient.real(), value.real() ), leaky_relu( leak_coefficient.imag(), value.imag() ) );
        }

        template <typename T, typename U>
        inline T d_leaky_relu( const U leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value > 0 ) ? 1 : leak_coefficient;
        }
        template <typename T, typename U>
        inline std::complex<T> d_leaky_relu( const std::complex<U> leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value > 0 ) ? 1 : leak_coefficient;
        }
        template <typename T, typename U>
        inline std::complex<T> d_leaky_relu( const U leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_leaky_relu( leak_coefficient, value.real() ), d_leaky_relu( leak_coefficient, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_leaky_relu( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_leaky_relu( leak_coefficient.real(), value.real() ), d_leaky_relu( leak_coefficient.imag(), value.imag() ) );
        }

    //: Elu
        template <typename T, typename U>
        inline T elu( const U leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value < 0 ) ? ( leak_coefficient * ( nn::exp( value ) - 1 ) ) : value;
        }
        template <typename T, typename U>
        inline std::complex<T> elu( const std::complex<U> leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value < 0 ) ? ( leak_coefficient * ( nn::exp( value ) - 1 ) ) : value;
        }
        template <typename T, typename U>
        inline std::complex<T> elu( const U leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( elu( leak_coefficient, value.real() ), elu( leak_coefficient, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> elu( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( elu( leak_coefficient.real(), value.real() ), elu( leak_coefficient.imag(), value.imag() ) );
        }

        template <typename T, typename U>
        inline T d_elu( const U leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value >= 0 ) ? 1 : ( leak_coefficient * nn::exp( value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_elu( const std::complex<U> leak_coefficient, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( value >= 0 ) ? 1 : ( leak_coefficient * nn::exp( value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_elu( const U leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_elu( leak_coefficient, value.real() ), d_elu( leak_coefficient, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_elu( const std::complex<U> leak_coefficient, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_elu( leak_coefficient.real(), value.real() ), d_elu( leak_coefficient.imag(), value.imag() ) );
        }

    //: Swish
        template <typename T>
        inline T swish( const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return value * sigmoid( value );
        }
        template <typename T>
        inline std::complex<T> swish( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return value * sigmoid( value );
        }
        template <typename T>
        inline std::complex<T> separable_swish( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( swish( value.real() ), swish( value.imag() ) );
        }

        template <typename T>
        inline T d_swish( const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            T swish_value = swish( value );
            return swish_value + ( sigmoid( value ) * ( 1 - swish_value ) );
        }
        template <typename T>
        inline std::complex<T> d_swish( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            std::complex<T> swish_value = swish( value );
            return swish_value + ( sigmoid( value ) * ( 1 - swish_value ) );
        }
        template <typename T>
        inline std::complex<T> d_separable_swish( const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_swish( value.real() ), d_swish( value.imag() ) );
        }

    //: ESwish
        template <typename T, typename U>
        inline T eswish( const U beta, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( beta * value ) * sigmoid( value );
        }
        template <typename T, typename U>
        inline std::complex<T> eswish( const std::complex<U> beta, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( beta * value ) * sigmoid( value );
        }
        template <typename T, typename U>
        inline std::complex<T> eswish( const U beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( beta * value ) * sigmoid( value );
        }
        template <typename T, typename U>
        inline std::complex<T> eswish( const std::complex<U> beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return ( beta * value ) * sigmoid( value );
        }
        template <typename T, typename U>
        inline std::complex<T> separable_eswish( const U beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( eswish( beta, value.real() ), eswish( beta, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> separable_eswish( const std::complex<U> beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( eswish( beta.real(), value.real() ), eswish( beta.imag(), value.imag() ) );
        }

        template <typename T, typename U>
        inline T d_eswish( const U beta, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            T eswish_value = eswish( beta, value );
            return eswish_value + ( sigmoid( value ) * ( beta - eswish_value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_eswish( const std::complex<U> beta, const T value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            T eswish_value = eswish( beta, value );
            return eswish_value + ( sigmoid( value ) * ( beta - eswish_value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_eswish( const U beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            std::complex<T> eswish_value = eswish( beta, value );
            return eswish_value + ( sigmoid( value ) * ( beta - eswish_value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_eswish( const std::complex<U> beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            std::complex<T> eswish_value = eswish( beta, value );
            return eswish_value + ( sigmoid( value ) * ( beta - eswish_value ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_separable_eswish( const U beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_eswish( beta, value.real() ), d_eswish( beta, value.imag() ) );
        }
        template <typename T, typename U>
        inline std::complex<T> d_separable_eswish( const std::complex<U> beta, const std::complex<T> value ) requires ( is_int<T>::value || is_float<T>::value )
        {
            return std::complex<T>( d_eswish( beta.real(), value.real() ), d_eswish( beta.imag(), value.imag() ) );
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ACTIVATION_HPP_
