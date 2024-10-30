#include "../source/bewusstsein_nn.h"

int main()
{
    srand(time(0));

    bewusstsein::nn::layers::node_layer<long> layer_0
    (
        "Layer_0",
        bewusstsein::util::core::shape(10, 10, 1, 1, 1),
        bewusstsein::nn::core::training_mode::adam
    );

    bewusstsein::nn::layers::node_layer<long> layer_1
    (
        "Layer_1",
        bewusstsein::util::core::shape(10, 10, 1, 1, 1),
        bewusstsein::nn::core::training_mode::adam
    );

    bewusstsein::nn::layers::dense_layer<long> dense_0_1
    (
        "Dense_0_1",
        bewusstsein::util::core::shape(10, 10, 1, 1, 1),
        bewusstsein::util::core::shape(10, 10, 1, 1, 1),
        bewusstsein::nn::core::training_mode::adam
    );

    bewusstsein::nn::layers::cost_layer<long> cost
    (
        "Cost",
        bewusstsein::nn::core::cost::mean_squared_error,
        1
    );

    bewusstsein::nn::layers::node_layer<long> target
    (
        "Target",
        bewusstsein::util::core::shape(10, 10, 1, 1, 1)
    );

    layer_0.randomize(0, 100);
    layer_1.zero();
    target.randomize(0, 100);
    target.randomize(0, 100);
    //dense_0_1.initialize(bewusstsein::nn::core::initialization::he, bewusstsein::util::core::distribution::normal);

    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    while (true)
    {
        start = std::chrono::high_resolution_clock::now();

        dense_0_1.inference(layer_0, layer_1);
        cost.backpropagation(layer_1, target);
        dense_0_1.backpropagation(layer_0, layer_1);
        dense_0_1.gradient_decent_adam(1, 0.001, 0.9, 0.99, std::numeric_limits<double>::epsilon());

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        //if(system("cls")){throw("Terminal clear failure");}
        if(system("clear")){throw("Terminal clear failure");}
        std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
        layer_0.print();
        fflush(stdout);
        layer_1.print();
        fflush(stdout);
        target.print();
        fflush(stdout);
    }
}