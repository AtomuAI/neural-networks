#include "../source/bewusstsein_nn.h"

int main()
{
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    auto neural_network = std::make_shared<bewusstsein::cmpg::core::graph>("neural_network");
    neural_network->set_num_threads(1);

    auto layer_0 = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "layer_0",
        bewusstsein::util::core::shape(10,10),
        bewusstsein::nn::core::training_mode::adam
    );

    auto layer_1 = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "layer_1",
        bewusstsein::util::core::shape(10,10),
        bewusstsein::nn::core::training_mode::adam
    );

    auto dense_0_1 = std::make_shared<bewusstsein::nn::layers::dense_layer<float, float, float>>
    (
        "dense_0_1",
        bewusstsein::util::core::shape(10,10),
        bewusstsein::util::core::shape(10,10),
        bewusstsein::nn::core::training_mode::adam
    );

    auto cost = std::make_shared<bewusstsein::nn::layers::cost_layer<float, float, float>>
    (
        "layer_1",
        bewusstsein::nn::core::cost_type::mean_squared_error,
        1
    );

    auto target = std::make_shared<bewusstsein::nn::layers::node_layer<float>>
    (
        "target",
        bewusstsein::util::core::shape(10,10),
        bewusstsein::nn::core::training_mode::adam
    );

    dense_0_1->input_variables.append_pin("input_layer");
    dense_0_1->output_variables.append_pin("output_layer");

    cost->input_variables.append_pin("output_layer");
    cost->output_variables.append_pin("target_layer");

    connect_nodes(neural_network->get_entry_node(), dense_0_1);
    connect_nodes(dense_0_1, cost);
    connect_nodes(cost, neural_network->get_exit_node());

    bewusstsein::cmpg::core::connect_var_op_in(layer_0, dense_0_1, 0);
    bewusstsein::cmpg::core::connect_op_out_var(dense_0_1, 0, layer_1);

    bewusstsein::cmpg::core::connect_var_op_in(layer_1, cost, 0);
    bewusstsein::cmpg::core::connect_op_out_var(cost, 0, target);

    target->randomize(0, 1);
    layer_0->randomize(0, 1);
    layer_1->zero();
    dense_0_1->initialize(bewusstsein::nn::core::initialization_type::he, bewusstsein::util::core::distribution::normal);

    std::cout << "Before: " << std::endl;

    layer_0->print();
    layer_0->print_delta();

    layer_1->print();
    layer_1->print_delta();

    start = std::chrono::high_resolution_clock::now();

    neural_network->forward();

    neural_network->backward();

    neural_network->optimize();

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    std::cout << std::endl << "ms: " << elapsed.count() << std::endl << std::endl;

    std::cout << "After: " << std::endl;

    layer_0->print();
    layer_0->print_delta();

    layer_1->print();
    layer_1->print_delta();

    std::cout << "Target: " << std::endl;

    target->print();

    //neural_network->generate_dot("neural_network_dot.dot");
}