#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;

    nn::layers::nodelayer layer_0
    (
        {28,28,1,1,1},
        "layer_0",
        true
    );

    nn::layers::nodelayer layer_1
    (
        {28,28,1,1,1},
        "layer_1",
        true
    );

    nn::layers::nodelayer layer_0_1
    (
        {28,28,1,1,2},
        "layer_0_1",
        true
    );

    layer_0.load_images({"/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/00001.bmp"}, util::core::image_type::GRAYSCALE);

    layer_1.load_images({"/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/00002.bmp"}, util::core::image_type::GRAYSCALE);

    layer_0_1.load_images({"/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/00001.bmp", "/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/00002.bmp"}, util::core::image_type::GRAYSCALE);

    layer_0.print();

    layer_1.print();

    layer_0_1.print();
}