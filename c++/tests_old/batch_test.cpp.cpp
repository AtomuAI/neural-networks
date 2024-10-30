#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    long iterations = 100;

    nn::layers::nodelayer input_layer
    (
        {6,6,1,1,2}, 
        1,
        "Input", 
        true
    );
    nn::layers::normlayer input_norm_layer
    (
        nn::core::batch_norm,
        input_layer.get_shape(),
        true,
        true,
        true
    );
    nn::layers::dropoutlayer input_dropout_layer
    (
        {6,6,1,1},
        0.5
    );
    nn::layers::convlayer conv_layer
    (
        {3,3}, 
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 
        "Convolution", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::valid, 
        nn::core::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer output_layer
    (
        {4,4,1,1,2}, 
        0, 
        "Output", 
        true
    );
    nn::layers::biaslayer bias_layer
    (
        {4,4}, 
        0.1, 
        "Bias", 
        true,
        true,
        true
    );
    nn::layers::poolinglayer pool_layer
    (
        {2,2,1}, 
        "Pool", 
        {2,2,0}, 
        {1,1,1}, 
        nn::core::max
    );
    nn::layers::activationlayer activ_layer(bewusstsein::nn::core::activation::leakyrelu, 0.1);
    nn::layers::nodelayer pooled_layer
    (
        {2,2,1,1,2}, 
        0, 
        "Pooled", 
        true
    );
    nn::layers::nodelayer dense_output_layer
    (
        {2,2,1,1,2}, 
        0, 
        "Dense Output", 
        true
    );
    nn::layers::denselayer dense_layer
    (
        {2,2}, 
        {2,2},
        0.01, 
        "Dense", 
        true, 
        true,
        true
    );
    nn::layers::costlayer cost_layer
    {
        bewusstsein::nn::core::mean_squared_error,
        iterations
    };
    nn::layers::nodelayer target_layer
    (
        {2,2,1,1,2}, 
        10, 
        "Target", 
        true
    );

    input_layer.randomize(0.0, 1.0);
    target_layer.randomize(0.0, 1.0);

    conv_layer.randomize(-0.01, 0.01);
    dense_layer.randomize(-0.01, 0.01);
    bias_layer.randomize(0.01, 0.1);

    conv_layer.info();
    dense_layer.info();
    bias_layer.info();
    pool_layer.info();
    input_layer.info();
    output_layer.info();
    pooled_layer.info();

    conv_layer.print();
    dense_layer.print();
    bias_layer.print();
    input_layer.print();
    output_layer.print();
    pooled_layer.print();
    dense_output_layer.print();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        input_norm_layer.stat_analysis(input_layer);
        input_norm_layer.inference(input_layer);
        input_dropout_layer.inference(input_layer);
        conv_layer.inference(input_layer, output_layer);
        bias_layer.inference(output_layer);
        activ_layer.inference(output_layer);
        pool_layer.inference(output_layer, pooled_layer);
        dense_layer.inference(pooled_layer, dense_output_layer);

        cost_layer.backpropagation(dense_output_layer, target_layer);
        dense_layer.backpropagation(pooled_layer, dense_output_layer);
        pool_layer.backpropagation(output_layer, pooled_layer);
        activ_layer.backpropagation(output_layer);
        bias_layer.backpropagation(output_layer);
        conv_layer.backpropagation(input_layer, output_layer);
        input_dropout_layer.backpropagation(input_layer);
        input_norm_layer.backpropagation(input_layer);
        
        input_norm_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        conv_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        bias_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        dense_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

    conv_layer.print();
    dense_layer.print();
    input_layer.print();
    output_layer.print();
    pooled_layer.print();
    dense_output_layer.print();
    target_layer.print();
}