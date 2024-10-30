#include "../source/bewusstsein_nn.h"

int main()
{
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    using namespace bewusstsein;

    nn::layers::nodelayer layer_0
    (
        {28,28,1,3,1},
        "Layer_0",
        true
    );

    nn::layers::translationlayer trans_0_1
    (
        {-28,-28,0},
        "Rotation_0_1"
    );

    nn::layers::nodelayer layer_1
    (
        {28,28,1,3,1},
        "Layer_1",
        true
    );

    layer_0.create_window(1280, 720);
    layer_1.create_window(1280, 720);

    //layer_0.load_images({"/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/00001.bmp"}, bewusstsein::util::core::image_type::COLOR);

    layer_0.randomize(0, 1);

    while(true)
    {
        layer_1.zero();

        trans_0_1.get_translation()[0] += 1;
        trans_0_1.get_translation()[1] += 1;
        if 
        (
            (trans_0_1.get_translation()[0] >= 28)
            ||
            (trans_0_1.get_translation()[1] >= 28)
        )
        {
            trans_0_1.get_translation()[0] = -28;
            trans_0_1.get_translation()[1] = -28;
        }
        
        start = std::chrono::high_resolution_clock::now();

        trans_0_1.backpropagation(layer_1, layer_0);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

        layer_0.show_batch_video();
        layer_1.show_batch_video();
    }
}