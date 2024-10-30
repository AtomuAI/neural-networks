#include "../source/bewusstsein_nn.h"

int main()
{
    bewusstsein::nn::core::neural_network network;

    network.add_layer<bewusstsein::nn::layers::node_layer<float>>("Layer", bewusstsein::util::core::shape(10,10,1,1,1));

    //bewusstsein::nn::layers::node_layer<float> layer = network.get_layer<bewusstsein::nn::layers::node_layer<float>>(0);

    bewusstsein::nn::layers::node_layer<float> layer = network.get_layer_by_name<bewusstsein::nn::layers::node_layer<float>>("Layer");

    while (true) {}
}