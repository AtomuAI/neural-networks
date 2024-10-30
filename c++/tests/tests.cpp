// Copyright 2024 Shane W. Mulcahy

#include <cassert>

#include "bewusstsein_neural_networks/c++/include/bewusstsein_nn.hpp"

#include "bewusstsein_neural_networks/c++/tests/layer/node.hpp"
#include "bewusstsein_neural_networks/c++/tests/layer/dense.hpp"

int main()
{
    bool result = true;

    result &= test_node_layer(); assert( true == result );
    result &= test_dense_layer(); assert( true == result );
}
