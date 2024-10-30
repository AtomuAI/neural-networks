#include "../source/bewusstsein_nn.h"

using namespace bewusstsein;

int main()
{
    nn::layers::nodelayer layer_0
    (
        {28,28,1,1,4},
        "Layer 2",
        true
    );

    nn::layers::normlayer norm_0
    (
        nn::core::normalization::batch_norm,
        layer_0.get_shape(),
        "Normalization 2",
        true,
        true,
        true
    );

    nn::layers::normlayer norm_1
    (
        nn::core::normalization::batch_norm,
        layer_0.get_shape(),
        "Normalization 2",
        true,
        true,
        true
    );

    int batch_size = 4;

    std::vector<std::string> batch(batch_size);
    std::vector<std::string> output(batch_size);

    batch.clear();
    output.clear();
    for (int b = 0; b < batch_size; b++)
    {
        std::string filename = "/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/" + std::string(5 - std::to_string(b + 1).length(), '0') + std::to_string(b + 1) + ".bmp";
        batch.push_back(filename);
        filename = "/home/shane/Documents/VSCode/bewusstsein_api/bewusstsein_neural_networks/images/before_" + std::string(5 - std::to_string(b + 1).length(), '0') + std::to_string(b + 1) + ".bmp";
        output.push_back(filename);
    }

    layer_0.load_images(batch, util::core::image_type::GRAYSCALE);

    //layer_0.save_images(output);
    layer_0.print();

    norm_0.stat_analysis(layer_0);
    norm_0.inference(layer_0);
    norm_0.print();

    layer_0 *= sqrt(139283);
    layer_0 += 0.119045;

    output.clear();
    for (int b = 0; b < batch_size; b++)
    {
        std::string filename = "/home/shane/Documents/VSCode/bewusstsein_api/bewusstsein_neural_networks/images/after_" + std::string(5 - std::to_string(b + 1).length(), '0') + std::to_string(b + 1) + ".bmp";
        output.push_back(filename);
    }

    layer_0.save_images(output);
    layer_0.print();
}