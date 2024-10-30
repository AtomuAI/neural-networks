#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    long iterations = 50;

    nn::layers::nodelayer input_layer
    (
        {2,2,1,1,4}, 
        1,
        "Input", 
        true
    );
    nn::layers::normlayer input_norm_layer
    {
        nn::core::batch_norm,
        input_layer.get_shape(),
        true,
        true,
        true
    };
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
    nn::layers::nodelayer dense_output_layer
    (
        {2,2,1,1,4}, 
        0, 
        "Dense Output", 
        true
    );
    nn::layers::costlayer cost_layer
    {
        bewusstsein::nn::core::mean_squared_error,
        iterations
    };
    nn::layers::nodelayer target_layer
    (
        {2,2,1,1,4}, 
        1, 
        "Target", 
        true
    );

    input_layer.randomize(0.0, 1.0);
    target_layer.randomize(0.0, 1.0);

    dense_layer.randomize(-0.01, 0.01);

    dense_layer.info();
    input_layer.info();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        input_layer.print();
        input_norm_layer.stat_analysis(input_layer);
        input_norm_layer.inference(input_layer);
        input_layer.print();
        dense_layer.inference(input_layer, dense_output_layer);

        dense_layer.print();
        dense_output_layer.print();
        target_layer.print();

        cost_layer.backpropagation(dense_output_layer, target_layer);
        dense_layer.backpropagation(input_layer, dense_output_layer);
        input_norm_layer.backpropagation(input_layer);
        
        input_norm_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
        dense_layer.gradient_decent_adam(0.001, 0.9, 0.99, 1e-8);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
}