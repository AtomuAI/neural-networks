// Copyright 2024 Shane W. Mulcahy

#include <cassert>

#include "bewusstsein_neural_networks/c++/include/bewusstsein_nn.hpp"
#include "bewusstsein_neural_networks/c++/tests/core/print.hpp"

int main()
{
    nn::NodeLayer<nn::b8> layer_a
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        true
    );
    layer_a.randomize_nodes( false, true );
    print_tensor( layer_a.get_nodes() );

    nn::NodeLayer<nn::b8> layer_b
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        false
    );
    layer_b.set_training_mode( nn::TrainingMode::normal );
    print_tensor( layer_b.get_nodes() );

    nn::NodeLayer<nn::b8> target
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        {
            true, true,
            true, true,

            true, false,
            false, true,

            false, true,
            true, false,

            false, false,
            false, false,
        }
    );
    print_tensor( target.get_nodes() );

    nn::DenseLayer<nn::b8> dense_a
    (
        layer_a.get_shape(),
        layer_b.get_shape(),
        false
    );
    dense_a.set_training_mode( nn::TrainingMode::normal );
    dense_a.randomize_weights( false, true );
    print_tensor( dense_a.get_weights() );

    dense_a.inference( layer_a, layer_b );
    print_tensor( layer_b.get_nodes() );

    nn::CostLayer<nn::CostType::mean_squared_error> cost( 1 );

    for ( nn::i32 i = 0; i < 1000; i++ )
    {
        cost.backpropagation( layer_b, target );
        dense_a.backpropagation( layer_a, layer_b );
        dense_a.gradient_decent<nn::TrainingMode::normal>( 1, 0.01 );

        dense_a.inference( layer_a, layer_b );
        print_tensor( layer_b.get_nodes() );
    }
}
