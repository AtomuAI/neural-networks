#include "../source/bewusstsein_nn.h"

std::vector<int> load_mnist_labels(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) 
    {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return {};
    }

    // Read the magic number and number of labels
    int magic_number;
    int num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(int));

    // Convert to little-endian format
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    // Check if the magic number is correct
    if (magic_number != 0x00000801) 
    {
        std::cerr << "Error: Invalid magic number in file " << filename << "\n";
        return {};
    }

    // Read the labels
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; i++) 
    {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}


int main()
{
    //std::string working_directory = "C:/Users/shane/Documents/VSCode/bewusstsein_api/";
    std::string working_directory = "/home/shane/Documents/VSCode/bewusstsein_api/";

    using namespace bewusstsein;

    std::vector<int> labels = load_mnist_labels("/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_datasets/training/train-labels-idx1-ubyte");
    std::vector<int> eval_labels = load_mnist_labels("/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_datasets/inference/t10k-labels-idx1-ubyte");
    
    int batch_size = 100;
    int samples = 2;
    int epochs = 40;
    int examples = labels.size();
    int evaluations = eval_labels.size();
    
    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = 1e-8;

    nn::core::neural_network network;

    // 0 : Input
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {28,28,1,1,batch_size},
            "Layer_0",
            true
        )
    );
    //--------------------------------------------------

    // 1 : Feature Extraction
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {26,26,1,samples,batch_size},
            "Layer_1",
            true
        )
    );
    network.add_layer
    (
        nn::layers::convlayer
        (
            {3,3,1,samples}, 
            "Convolution_0_1", 
            {1,1,0},
            {1,1,1},
            nn::core::convolution_type::normal,
            nn::core::padding::valid, 
            nn::core::padding_value::zero, 
            {1,1,1},
            true, 
            true,
            true
        ),
        "Layer_0",
        "Layer_1"
    );
    network.add_layer
    (
        nn::layers::biaslayer
        (
            {26,26,1,samples},
            "Bias_1",
            true, 
            true,
            true
        ),
        "Layer_1"
    );
    network.add_layer
    (
        nn::layers::activationlayer
        (
            nn::core::activation::pararelu,
            {1,1,1,samples},
            0.0,
            "Activation_1",
            true,
            true,
            true
        ),
        "Layer_1"
    );
    //--------------------------------------------------

    // 1 : Feature Extraction
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {24,24,1,samples*2,batch_size},
            "Layer_2",
            true
        )
    );
    network.add_layer
    (
        nn::layers::convlayer
        (
            {3,3,1,samples*2},
            "Convolution_1_2",
            {1,1,0},
            {1,1,1},
            nn::core::convolution_type::normal,
            nn::core::padding::valid,
            nn::core::padding_value::zero,
            {1,1,1},
            true,
            true,
            true
        ),
        "Layer_1",
        "Layer_2"
    );
    network.add_layer
    (
        nn::layers::biaslayer
        (
            {24,24,1,samples*2},
            "Bias_2",
            true, 
            true,
            true
        ),
        "Layer_2"
    );
    network.add_layer
    (
        nn::layers::activationlayer
        (
            nn::core::activation::pararelu,
            {1,1,1,samples*2},
            0.0,
            "Activation_2",
            true,
            true,
            true
        ),
        "Layer_2"
    );
    //--------------------------------------------------

    // 3 : Feature Pool
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {12,12,1,samples*2,batch_size},
            "Layer_3",
            true
        )
    );
    network.add_layer
    (
        nn::layers::poolinglayer
        (
            {2,2,1}, 
            "Pool_2_3", 
            {2,2,0}, 
            {1,1,1}, 
            nn::core::pooling::max
        ),
        "Layer_2",
        "Layer_3"
    );
    //--------------------------------------------------

    // 4 : Feature Extraction
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10,10,1,samples*4,batch_size},
            "Layer_4",
            true
        )
    );
    network.add_layer
    (
        nn::layers::convlayer
        (
            {3,3,1,samples*4}, 
            "Convolution_3_4", 
            {1,1,0}, 
            {1,1,1}, 
            nn::core::convolution_type::normal,
            nn::core::padding::valid, 
            nn::core::padding_value::zero, 
            {1,1,1},
            true, 
            true,
            true
        ),
        "Layer_3",
        "Layer_4"
    );
    network.add_layer
    (
        nn::layers::biaslayer
        (
            {10,10,1,samples*4},
            "Bias_4",
            true, 
            true,
            true
        ),
        "Layer_4"
    );
    network.add_layer
    (
        nn::layers::activationlayer
        (
            nn::core::activation::pararelu,
            {1,1,1,samples*4},
            0.0,
            "Activation_4",
            true,
            true,
            true
        ),
        "Layer_4"
    );
    //--------------------------------------------------

    // 5 : Feature Projection
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10*samples*4,1,1,1,batch_size}, 
            "Layer_5", 
            true
        )
    );
    network.add_layer
    (
        nn::layers::denselayer
        (
            {10,10,1,samples*4}, 
            {10*samples*4,1,1,1},
            "Dense_4_5", 
            true, 
            true,
            true
        ),
        "Layer_4",
        "Layer_5"
    );
    network.add_layer
    (
        nn::layers::biaslayer
        (
            {10*samples*4,1,1,1},
            "Bias_5",
            true, 
            true,
            true
        ),
        "Layer_5"
    );
    network.add_layer
    (
        nn::layers::activationlayer
        (
            nn::core::activation::pararelu,
            {1,1,1,1},
            0.0,
            "Activation_5",
            true,
            true,
            true
        ),
        "Layer_5"
    );
    //--------------------------------------------------

    // 6 : Feature Projection
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10,1,1,1,batch_size}, 
            "Layer_6", 
            true
        )
    );
    network.add_layer
    (
        nn::layers::denselayer
        (
            {10,1,1,samples*4}, 
            {10,1,1,1},
            "Dense_5_6", 
            true, 
            true,
            true
        ),
        "Layer_5",
        "Layer_6"
    );
    network.add_layer
    (
        nn::layers::biaslayer
        (
            {10,1,1,1},
            "Bias_6",
            true, 
            true,
            true
        ),
        "Layer_6"
    );
    network.add_layer
    (
        nn::layers::dropoutlayer
        (
            {10,1,1,1},
            "Dropout_6",
            0.5
        ),
        "Layer_6"
    );
    //--------------------------------------------------

    // 7 : Error
    //--------------------------------------------------
    network.add_layer
    (
        nn::layers::nodelayer
        (
            {10,1,1,1,batch_size}, 
            "Target", 
            true
        )
    );
    network.add_layer
    (
        nn::layers::costlayer
        (
            bewusstsein::nn::core::cost::softmax_categorical_cross_entropy,
            "Cost",
            examples
        ),
        "Layer_6",
        "Target"
    );
    //--------------------------------------------------

    while (true) {}

    /*

    network.get_convlayer("Convolution_0_1").initialize(network.get_nodelayer("Layer_0").get_shape(), network.get_nodelayer("Layer_1").get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    network.get_convlayer("Convolution_1_2").initialize(network.get_nodelayer("Layer_1").get_shape(), network.get_nodelayer("Layer_2").get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    network.get_convlayer("Convolution_3_4").initialize(network.get_nodelayer("Layer_3").get_shape(), network.get_nodelayer("Layer_4").get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    network.get_denselayer("Dense_4_5").initialize(nn::core::initialization::he, util::core::distribution::normal);
    network.get_denselayer("Dense_5_6").initialize(nn::core::initialization::he, util::core::distribution::normal);

    network.set_input("Layer_0");
    network.set_output("Layer_6");
    network.set_target("Target");

    int iterations = examples/batch_size;
    std::vector<std::string> batch(batch_size);

    network.get_input().create_window(1280, 720);
    network.get_output().create_window(1280, 720);

    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < iterations; i++)
        {
            start = std::chrono::high_resolution_clock::now();

            batch.clear();
            for (int b = 0; b < batch_size; b++)
            {
                std::string filename = "/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/" + std::string(5 - std::to_string(((i * batch_size) + b) + 1).length(), '0') + std::to_string(((i * batch_size) + b) + 1) + ".bmp";
                batch.push_back(filename);
                network.get_target()[(b * 10) + labels[(i * batch_size) + b]] = 1.0;
            }

            network.get_input().load_images(batch, util::core::image_type::GRAYSCALE);

            network.train_adam(batch_size, step_size, beta1, beta2, epsilon);

            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;

            //if(system("cls")){throw("Terminal clear failure");}
            if(system("clear")){throw("Terminal clear failure");}
            std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
            network.get_input().show_batch_video();
            network.get_output().show_batch_video();
            fflush(stdout);
        }
    }

    network.get_input().destroy_window();
    network.get_output().destroy_window();

    */
}

