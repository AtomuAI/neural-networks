#include "../source/bewusstsein_nn.h"

int main()
{
    //std::string working_directory = "C:/Users/shane/Documents/VSCode/bewusstsein_api/";
    std::string working_directory = "/home/shane/Documents/VSCode/bewusstsein_api/";

    using namespace bewusstsein;

    nn::core::neural_network network;

    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10, 10, 1, 3, 1}, 
            "Input", 
            true
        )
    );

    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10, 10, 1, 3, 1}, 
            "Output", 
            true
        )
    );

    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10, 10, 1, 3, 1}, 
            "Target", 
            true
        )
    );

    network.add_layer
    (
        nn::layers::denselayer
        (
            {10, 10, 1, 3}, 
            {10, 10, 1, 3},
            "Dense", 
            true, 
            true,
            true
        ),
        "Input",
        "Output"
    );

    network.add_layer
    (
        nn::layers::biaslayer
        (
            {10,10,1,3},
            "Bias",
            true, 
            true,
            true
        ),
        "Output"
    );

    network.add_layer
    (
        nn::layers::activationlayer
        (
            nn::core::activation::pararelu,
            {1,1,1,3},
            0.1,
            "Activation",
            true,
            true,
            true
        ),
        "Output"
    );

    network.add_layer
    (
        nn::layers::costlayer
        (
            bewusstsein::nn::core::cost::mean_squared_error,
            "Cost",
            1
        ),
        "Output",
        "Target"
    );

    while (true) {}

    network.set_input("Input");
    network.set_output("Output");
    network.set_target("Target");

    nn::layers::denselayer* dense = network.get_denselayer_ptr("Dense");
    dense->initialize(nn::core::initialization::he, util::core::distribution::normal);

    nn::layers::nodelayer* input = network.get_input_ptr();
    nn::layers::nodelayer* output = network.get_output_ptr();
    nn::layers::nodelayer* target = network.get_target_ptr();

    input->randomize(-1.0, 1.0);
    target->randomize(0.1, 1.0);

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = std::numeric_limits<float>::epsilon();
    int batch_size = 1;

    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    //input->create_window(1280, 720);
    //output->create_window(1280, 720);

    for (int i = 0; i < 1000; i++)
    {
        start = std::chrono::high_resolution_clock::now();

        network.train_adam(batch_size, step_size, beta1, beta2, epsilon);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        //input->show_batch_video();
        //output->show_batch_video();

        /*
        if(system("cls")){throw("Terminal clear failure");}
        //if(system("clear")){throw("Terminal clear failure");}
        std::cout << "Training: " << elapsed.count() << std::endl << std::endl;
        std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
        input->print();
        output->print();
        target->print();
        fflush(stdout);
        */
    }

    std::cout << "Save Model" << std::endl;
    network.save_model(working_directory + "bewusstsein_neural_networks/models/test_model_win64");

    nn::core::neural_network network2;

    std::cout << "Load Model" << std::endl;
    network2.load_model(working_directory + "bewusstsein_neural_networks/models/test_model_win64.bewusstseinmodel");

    std::cout << "Get Pointers" << std::endl;
    nn::layers::nodelayer* input2 = network2.get_input_ptr();
    nn::layers::nodelayer* output2 = network2.get_output_ptr();
    nn::layers::nodelayer* target2 = network2.get_target_ptr();

    output2->zero();

    input2->create_window(1280, 720);
    output2->create_window(1280, 720);

    while (true)
    {
        input2->randomize(0, 1);
        start = std::chrono::high_resolution_clock::now();

        network2.inference();

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        input2->show_batch_video();
        output2->show_batch_video();
        
        /*
        if(system("cls")){throw("Terminal clear failure");}
        //if(system("clear")){throw("Terminal clear failure");}
        std::cout << "Inference: " << elapsed.count() << std::endl << std::endl;
        std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
        input2->print();
        output2->print();
        target2->print();
        fflush(stdout);
        */
    }

    input->destroy_window();
    output->destroy_window();
    input2->destroy_window();
    output2->destroy_window();
}