#include "../source/nn.h"

using namespace bewusstsein;

int main()
{
    nn::layers::nodelayer layer_0({10,10,1,1,2}, "layer_0", false);
    nn::layers::nodelayer layer_1({10,10,1,1,2}, "layer_1", false);
    nn::layers::nodelayer layer_2({10,10,1,1,2}, "layer_2", false);
    nn::layers::convlayer conv_0_1({3,3}, 1.0, {1,1,0}, {1,1,0}, {1,1,1}, nn::core::padding::zero, false, false, false);
    nn::layers::denselayer dense_1_2({10,10}, {10,10}, 1.0, false, false, false);
    nn::layers::biaslayer bias_2({10,10}, 1.0, false, false, false);
    
    layer_0.randomize(0.0, 1.0);

    layer_0.info();
    layer_1.info();
    layer_2.info();

    layer_0.print();
    layer_1.print();
    layer_2.print();
    
    auto start = std::chrono::high_resolution_clock::now();

    conv_0_1.inference(layer_0, layer_1);
    dense_1_2.inference(layer_1, layer_2);
    bias_2.inference(layer_2);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "ms: " << elapsed.count() << std::endl;
    
    layer_0.print();
    layer_1.print();
    layer_2.print();
}