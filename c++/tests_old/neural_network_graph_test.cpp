#include "../../bewusstsein_computational_graphs/source/bewusstsein_comp_graph.h"
#include "../source/bewusstsein_nn.h"

int main()
{
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    bewusstsein::cmpg::graph network;

    auto layer_0 = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "Layer_0",
        bewusstsein::util::core::shape(4,4),
        bewusstsein::nn::core::training_mode::adam
    );
    auto conv_0_1 = std::make_shared<bewusstsein::nn::layers::convolution_layer<float, float, float>>
    (
        "Conv_0_1",
        bewusstsein::util::core::shape(2,2,1),
        bewusstsein::nn::core::convolution_type::normal,
        bewusstsein::nn::core::padding_type::circular,
        bewusstsein::nn::core::padding_size::same,
        bewusstsein::util::core::shape(1,1,1),
        bewusstsein::util::core::shape(1,1,0),
        bewusstsein::util::core::shape(1,1,1),
        bewusstsein::nn::core::training_mode::adam
    );
    auto layer_1 = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "Layer_1",
        bewusstsein::util::core::shape(4,4),
        bewusstsein::nn::core::training_mode::adam
    );
    auto dense_1_2 = std::make_shared<bewusstsein::nn::layers::dense_layer<float, float, float>>
    (
        "Dense_1_2",
        bewusstsein::util::core::shape(4,4),
        bewusstsein::util::core::shape(4,4),
        bewusstsein::nn::core::training_mode::adam
    );
    auto layer_2 = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "Layer_2",
        bewusstsein::util::core::shape(4,4),
        bewusstsein::nn::core::training_mode::adam
    );
    auto bias_2 = std::make_shared<bewusstsein::nn::layers::bias_layer<float, float>>
    (
        "Bias_2",
        bewusstsein::util::core::shape(4,4),
        bewusstsein::nn::core::training_mode::adam
    );
    auto activ_2 = std::make_shared<bewusstsein::nn::layers::activation_layer<float, float>>
    (
        "Activ_2",
        bewusstsein::nn::core::activation::pararelu,
        bewusstsein::util::core::shape(1), 
        0.1,
        bewusstsein::nn::core::training_mode::adam
    );
    auto cost = std::make_shared<bewusstsein::nn::layers::cost_layer<float, float, float>>
    (
        "Cost",
        bewusstsein::nn::core::cost::mean_squared_error,
        1
    );
    auto target = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "Target",
        bewusstsein::util::core::shape(4,4),
        std::vector<float>
        {
            0.1, 0.2, 0.3, 0.4,
            1.1, 1.2, 1.3, 1.4,
            2.1, 2.2, 2.3, 2.4,
            3.1, 3.2, 3.3, 3.4,
        },
        bewusstsein::nn::core::training_mode::adam
    );

    layer_0->randomize(0, 1);

    conv_0_1->initialize(layer_0->get_shape(), layer_1->get_shape(), bewusstsein::nn::core::initialization::he, bewusstsein::util::core::distribution::normal);
    dense_1_2->initialize(bewusstsein::nn::core::initialization::he, bewusstsein::util::core::distribution::normal);

    conv_0_1->set_input(layer_0);
    conv_0_1->set_output(layer_1);

    dense_1_2->set_input(layer_1);
    dense_1_2->set_output(layer_2);

    bias_2->set_input(layer_2);

    activ_2->set_input(layer_2);

    cost->set_input(layer_2);
    cost->set_output(target);

    network.append_node(layer_0);
    network.append_node(conv_0_1);
    network.append_node(layer_1);
    network.append_node(dense_1_2);
    network.append_node(layer_2);
    network.append_node(bias_2);
    network.append_node(activ_2);
    network.append_node(cost);

    while (1)
    {
        start = std::chrono::high_resolution_clock::now();

        network.forward();
        network.backward();
        
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

        bewusstsein::util::core::sleep(20);

        //if(system("cls")){throw("Terminal clear failure");}
        if(system("clear")){throw("Terminal clear failure");}
        std::cout << "-------------------------" << std::endl;
        layer_0->print();
        conv_0_1->print();
        conv_0_1->print_jacobian();
        layer_1->print();
        dense_1_2->print();
        dense_1_2->print_jacobian();
        layer_2->print();
        target->print();
        std::cout << "-------------------------" << std::endl;
        fflush(stdout);

        network.optimize();
    }
}