#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    long iterations = 1000;

    nn::core::neural_network network({10,10, 1, 1, 1}, {10,10, 1, 1, 1});

    nn::layers::nodelayer layer
    (
        {10,10, 1, 1, 1}, 
        1,
        "Input", 
        true
    );

    network.addLayer(&layer);
    network.removeLayer(1);
}