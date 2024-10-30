#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    nn::layers::nodelayer layer_0
    (
        {2,2,1,1,10},
        {
            0.0,0.0,
            0.0,0.0,

            0.1,0.1,
            0.1,0.1,

            0.2,0.2,
            0.2,0.2,

            0.3,0.3,
            0.3,0.3,

            0.4,0.4,
            0.4,0.4,

            0.5,0.5,
            0.5,0.5,

            0.6,0.6,
            0.6,0.6,

            0.7,0.7,
            0.7,0.7,

            0.8,0.8,
            0.8,0.8,

            0.9,0.9,
            0.9,0.9
        },
        "Layer_0",
        true
    );
    nn::layers::poolinglayer pool_0_1
    (
        {1,1,1}, 
        "Pool_1_2", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::pooling::max
    );
    nn::layers::convlayer conv_0_1
    (
        {1,1,1,2},
        1,
        "Convolution_0_1", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::padding::valid, 
        nn::core::padding_value::zero, 
        true, 
        true,
        true
    );
    nn::layers::denselayer dense_0_1
    (
        {2,2}, 
        {2,2},
        1,
        "Dense_0_1", 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_1
    (
        {2,2,1,2,10},
        "Layer_1",
        true
    );
    nn::layers::nodelayer target
    (
        {2,2,1,2,10},
        {
            0,0,
            0,0,

            0,0,
            0,0,

            1,1,
            1,1,

            1,1,
            1,1,

            2,2,
            2,2,

            2,2,
            2,2,

            3,3,
            3,3,

            3,3,
            3,3,

            4,4,
            4,4,

            4,4,
            4,4,

            5,5,
            5,5,

            5,5,
            5,5,

            6,6,
            6,6,

            6,6,
            6,6,

            7,7,
            7,7,

            7,7,
            7,7,

            8,8,
            8,8,

            8,8,
            8,8,

            9,9,
            9,9,

            9,9,
            9,9
        },
        "Target",
        true
    );
    nn::layers::costlayer cost
    (
        bewusstsein::nn::core::cost::softmax_categorical_cross_entropy,
        10
    );

    //dense_0_1.initialize(nn::core::initialization::he, util::core::distribution::normal);
    //conv_0_1.initialize(layer_0.get_shape(), layer_1.get_shape(), nn::core::initialization::he, util::core::distribution::normal);

    for (int i = 0; i < 1; i++)
    {

        //dense_0_1.inference(layer_0, layer_1);
        //conv_0_1.inference(layer_0, layer_1);
        pool_0_1.inference(layer_0, layer_1);

        cost.backpropagation(layer_1, target);
        //dense_0_1.backpropagation(layer_0, layer_1);
        //conv_0_1.backpropagation(layer_0, layer_1);
        pool_0_1.backpropagation(layer_0, layer_1);


        //dense_0_1.gradient_decent(10, 0.001);
        //conv_0_1.gradient_decent(10, 0.001);
    }

    //dense_0_1.print();
    //conv_0_1.print();

    layer_0.print();
    layer_1.print();

    target.print();

    //dense_0_1.print_jacobian();
    //conv_0_1.print_jacobian();

    layer_0.print_delta();
    layer_1.print_delta();


}