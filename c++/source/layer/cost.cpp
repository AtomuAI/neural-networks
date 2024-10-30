// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"
#include "bewusstsein_neural_networks/c++/include/core/cost.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/cost.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <CostType C>
        CostLayer<C>::CostLayer( const Dim num_examples ) :
            BaseLayer( LayerType::cost_layer ), num_examples( num_examples ) {}

    //: Destructors
        template <CostType C>
        CostLayer<C>::~CostLayer() {}

    //: Methods
        template <CostType C>
        template <typename U, typename V>
        Error CostLayer<C>::inference( NodeLayer<U>& layer, const NodeLayer<V>& target ) const
        {
            const Shape<5> layer_shape = layer.get_shape();
            const Shape<5> target_shape = target.get_shape();
            if ( layer_shape != target_shape ) { return Error::MISMATCHED_SHAPES; }

            Size layer_size = layer.get_size();

            for ( Idx i = 0; i < layer_size; ++i )
            {
                layer.get_delta( i ) = this->cost( target.get_node( i ), layer.get_node( i ) );
            }

            return Error::NONE;
        }

        template <CostType C>
        template <typename U, typename V>
        Error CostLayer<C>::backpropagation( NodeLayer<U>& layer, const NodeLayer<V>& target ) const
        {
            const Shape<5> layer_shape = layer.get_shape();
            const Shape<5> target_shape = target.get_shape();
            if ( layer_shape != target_shape ) { return Error::MISMATCHED_SHAPES; }

            Size size = layer.get_size();

            for ( Idx i = 0; i < size; ++i )
            {
                layer.get_delta( i ) = this->d_cost( target.get_node( i ), layer.get_node( i ) );
            }

            return Error::NONE;
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::mean_squared_error )
        {
            return mean_squared_error( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::categorical_cross_entropy )
        {
            return categorical_cross_entropy( target, value );
        }

        /*
        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::softmax_categorical_cross_entropy )
        {
            return softmax_categorical_cross_entropy( target, value );
        }
        */

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::hellinger_distance )
        {
            return hellinger_distance( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::kullback_leibler_divergence )
        {
            return kullback_leibler_divergence( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::generalized_kullback_leibler_divergence )
        {
            return generalized_kullback_leibler_divergence( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::cost( const V target, const U value ) const requires ( C == CostType::itakura_saito_distance )
        {
            return itakura_saito_distance( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::mean_squared_error )
        {
            return mean_squared_error_derivative( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::categorical_cross_entropy )
        {
            return categorical_cross_entropy_derivative( target, value );
        }

        /*
        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::softmax_categorical_cross_entropy )
        {
            return softmax_categorical_cross_entropy_derivative( target, value );
        }
        */

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::hellinger_distance )
        {
            return hellinger_distance_derivative( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::kullback_leibler_divergence )
        {
            return kullback_leibler_divergence_derivative( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::generalized_kullback_leibler_divergence )
        {
            return generalized_kullback_leibler_divergence_derivative( target, value );
        }

        template <CostType C>
        template <typename U, typename V>
        inline U CostLayer<C>::d_cost( const V target, const U value ) const requires ( C == CostType::itakura_saito_distance )
        {
            return itakura_saito_distance_derivative( target, value );
        }

} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacroB( cost_type, type_a, type_b )\
        template Error CostLayer<cost_type>::inference( NodeLayer<type_a>& layer, const NodeLayer<type_b>& target ) const;\
        template Error CostLayer<cost_type>::backpropagation( NodeLayer<type_a>& layer, const NodeLayer<type_b>& target ) const;

    #define FunctionMacro( cost_type, type )\
        FunctionMacroB( cost_type, type, i32 )\
        FunctionMacroB( cost_type, type, i64 )\
        FunctionMacroB( cost_type, type, f32 )\
        FunctionMacroB( cost_type, type, f64 )

    #define ClassMacroB( cost_type )\
        template class CostLayer<cost_type>;\
        FunctionMacro( cost_type, i32 )\
        FunctionMacro( cost_type, i64 )\
        FunctionMacro( cost_type, f32 )\
        FunctionMacro( cost_type, f64 )

    #define ClassMacro\
        ClassMacroB( CostType::mean_squared_error )\
        ClassMacroB( CostType::categorical_cross_entropy )\
        /*ClassMacroB( CostType::softmax_categorical_cross_entropy )\*/
        ClassMacroB( CostType::hellinger_distance )\
        ClassMacroB( CostType::kullback_leibler_divergence )\
        ClassMacroB( CostType::generalized_kullback_leibler_divergence )\
        ClassMacroB( CostType::itakura_saito_distance )

    ClassMacro

    #undef FunctionMacroB
    #undef FunctionMacro
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn
