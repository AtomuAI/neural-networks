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
#include <functional>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/softmax.hpp"

namespace nn
{
    //: Constructors
        SoftmaxLayer::SoftmaxLayer() : BaseLayer( LayerType::softmax_layer ) {}

    //: Destructors
        SoftmaxLayer::~SoftmaxLayer() {}

    //: Methods
        template <typename U, typename V>
        Error SoftmaxLayer::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const Shape<5> input_shape = input_layer.get_shape();
            const Shape<5> output_shape = input_layer.get_shape();
            Size spacial_size = input_shape.distance( 0, 4 );

            if ( input_shape != output_shape ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim i = 0; i < input_shape.batches(); ++i )
            {
                U sum( 0 );
                Idx batch_index = i * spacial_size;
                for ( Idx j = 0; j < spacial_size; ++j )
                {
                    Idx index = batch_index + j;
                    U value = exp( input_layer.get_node( index ) );
                    output_layer.get_node( index ) = value;
                    sum += value;
                }
                for ( Idx j = 0; j < spacial_size; ++j )
                {
                    Idx index = batch_index + j;
                    if ( ( sum != INFINITY ) && ( sum != 0 ) && ( sum != -INFINITY ) )
                    {
                        output_layer.get_node( index ) /= sum;
                    }
                    else if ( sum == INFINITY )
                    {
                        output_layer.get_node( index ) = 0;
                    }
                    else
                    {
                        output_layer.get_node( index ) = std::numeric_limits<V>::max();
                    }
                }
            }

            return Error::NONE;
        }

        template <typename U, typename V>
        Error SoftmaxLayer::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const
        {
            const Shape<5> input_shape = input_layer.get_shape();
            const Shape<5> output_shape = input_layer.get_shape();
            Size spacial_size = input_shape.distance( 0, 4 );

            if ( input_shape != output_shape ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim i = 0; i < input_shape.batches(); ++i )
            {
                U sum( 0 );
                Idx batch_index = i * spacial_size;
                for ( Idx j = 0; j < spacial_size; ++j )
                {
                    Idx index = batch_index + j;
                    U delta = ( output_layer.get_delta( index ) > 0 ) ? 1 : 0;
                    input_layer.get_delta( index ) = output_layer.get_node( index ) * ( delta - input_layer.get_node( index ) );
                }
            }

            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacroB( type_a, type_b )\
        template Error SoftmaxLayer::inference( const NodeLayer<type_a>& input_layer, NodeLayer<type_b>& output_layer ) const;\
        template Error SoftmaxLayer::backpropagation( NodeLayer<type_a>& input_layer, const NodeLayer<type_b>& output_layer ) const;

    #define FunctionMacro( type )\
        FunctionMacroB( type, f32 )\
        FunctionMacroB( type, f64 )

    #define ClassMacro\
        FunctionMacro( f32 )\
        FunctionMacro( f64 )

    ClassMacro

    #undef FunctionMacroB
    #undef FunctionMacro
    #undef ClassMacro
} // namespace nn
