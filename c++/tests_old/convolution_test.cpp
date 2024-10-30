#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    long iterations = 1000;

    nn::layers::nodelayer input_layer
    (
        {10,10}, 
        {
            1,    2,  3,    4,   5,   6,   7,   8,   9,  10,
            11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
            21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
            31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
            41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
            51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
            61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
            71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
            81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
            91,  92,  93,  94,  95,  96,  97,  98,  99, 100
        },
        "Input", 
        true
    );
    nn::layers::nodelayer output_layer
    (
        {8,8}, 
        0, 
        "Output", 
        true
    );
    nn::layers::convlayer conv_layer
    (
        {3,3}, 
        1, 
        "Convolution", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::valid, 
        nn::core::zero, 
        true, 
        true,
        true
    );
    nn::layers::biaslayer bias_layer
    (
        {8,8}, 
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
        {4,4}, 
        0, 
        "Pooled", 
        true
    );
    nn::layers::nodelayer dense_output_layer
    (
        {4,4}, 
        0, 
        "Dense Output", 
        true
    );
    nn::layers::denselayer dense_layer
    (
        {4,4}, 
        {4,4},
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
        {4,4}, 
        {
            2, 2, 2, 2,
            2, 2, 2, 2,
            2, 2, 2, 2,
            2, 2, 2, 2
        }, 
        "Pooled", 
        true
    );

    conv_layer.randomize(-0.01, -0.01);
    dense_layer.randomize(-0.01, 0.01);
    bias_layer.randomize(0.01, 0.1);

    conv_layer.info(); std::cout << std::endl;
    dense_layer.info(); std::cout << std::endl;
    bias_layer.info(); std::cout << std::endl;
    pool_layer.info(); std::cout << std::endl;
    input_layer.info(); std::cout << std::endl;
    output_layer.info(); std::cout << std::endl;
    pooled_layer.info(); std::cout << std::endl;

    conv_layer.print(); std::cout << std::endl;
    dense_layer.print(); std::cout << std::endl;
    bias_layer.print(); std::cout << std::endl;
    input_layer.print(); std::cout << std::endl;
    output_layer.print(); std::cout << std::endl;
    pooled_layer.print(); std::cout << std::endl;
    dense_output_layer.print(); std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
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

        dense_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        bias_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        conv_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

    conv_layer.print(); std::cout << std::endl;
    dense_layer.print(); std::cout << std::endl;
    input_layer.print(); std::cout << std::endl;
    output_layer.print(); std::cout << std::endl;
    pooled_layer.print(); std::cout << std::endl;
    dense_output_layer.print(); std::cout << std::endl;
}