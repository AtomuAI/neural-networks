#include "../source/bewusstseinnn.h"

using namespace bewusstsein::nn;

int main()
{
    layers::nodelayer layer_1({178, 218, 1, 3, 4}, "layer_1", false);

    layer_1.info();

    std::vector<std::string> images = 
    {
        "/home/shane/Downloads/archive(1)/img_align_celeba/img_align_celeba/000001.jpg",
        "/home/shane/Downloads/archive(1)/img_align_celeba/img_align_celeba/000002.jpg",
        "/home/shane/Downloads/archive(1)/img_align_celeba/img_align_celeba/000003.jpg",
        "/home/shane/Downloads/archive(1)/img_align_celeba/img_align_celeba/000004.jpg"
    };
    layer_1.load_images(images);
    layer_1.show_batch();
    //layer_1.save_images({"img_1.png", "img_2.png", "img_3.png", "img_4.png"});
    
}