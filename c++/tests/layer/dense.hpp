// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_DENSE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_DENSE_HPP_

#include <cassert>

#include "bewusstsein_neural_networks/c++/include/bewusstsein_nn.hpp"

#include "bewusstsein_neural_networks/c++/tests/core/print.hpp"

template <typename T>
bool test_create_dense_layer()
{
    nn::DenseLayer<T> dense_a
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        1
    );
    print_tensor( dense_a.get_weights() );

    nn::DenseLayer<T> dense_b
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        {
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        }
    );
    print_tensor( dense_b.get_weights() );

    return true;
}

template <>
bool test_create_dense_layer<nn::b8>()
{
    nn::DenseLayer<nn::b8> dense_a
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        1
    );
    print_tensor( dense_a.get_weights() );

    nn::DenseLayer<nn::b8> dense_b
    (
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        nn::Shape<5>( 2, 2, 2, 2, 1 ),
        {
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,

            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        }
    );
    print_tensor( dense_b.get_weights() );

    return true;
}

bool test_dense_layer()
{
    bool result = true;

    result &= test_create_dense_layer<nn::b8>(); assert( true == result );
    result &= test_create_dense_layer<nn::i32>(); assert( true == result );
    result &= test_create_dense_layer<nn::i64>(); assert( true == result );
    result &= test_create_dense_layer<nn::f32>(); assert( true == result );
    result &= test_create_dense_layer<nn::f64>(); assert( true == result );

    return true;
}

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_LAYER_DENSE_HPP_
