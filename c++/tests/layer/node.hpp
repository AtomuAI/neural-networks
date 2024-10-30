// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_NODE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_NODE_HPP_

#include <cassert>

#include "bewusstsein_neural_networks/c++/include/bewusstsein_nn.hpp"

#include "bewusstsein_neural_networks/c++/tests/core/print.hpp"

template <typename T>
bool test_create_node_layer()
{
    nn::NodeLayer<T> layer_a
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        1
    );
    print_tensor( layer_a.get_nodes() );

    nn::NodeLayer<T> layer_b
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        {
            1, 1,
            1, 1,

            1, 1,
            1, 1,

            1, 1,
            1, 1,

            1, 1,
            1, 1,
        }
    );
    print_tensor( layer_b.get_nodes() );

    return true;
}

template <>
bool test_create_node_layer<nn::b8>()
{
    nn::NodeLayer<nn::b8> layer_a
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        true
    );
    print_tensor( layer_a.get_nodes() );

    nn::NodeLayer<nn::b8> layer_b
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
    print_tensor( layer_b.get_nodes() );

    return true;
}

bool test_node_layer()
{
    bool result = true;

    result &= test_create_node_layer<nn::b8>(); assert( true == result );
    result &= test_create_node_layer<nn::i32>(); assert( true == result );
    result &= test_create_node_layer<nn::i64>(); assert( true == result );
    result &= test_create_node_layer<nn::f32>(); assert( true == result );
    result &= test_create_node_layer<nn::f64>(); assert( true == result );

    return true;
}

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_NODE_HPP_
