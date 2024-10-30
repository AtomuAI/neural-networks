#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;
    namespace plt = matplotlibcpp;

    int s = 1080;

    nn::layers::nodelayer layer_0
    (
        {2,1,1,1,s},
        "Layer_0",
        true
    );

    nn::layers::denselayer dense_0_1
    (
        {2,1,1,1,s}, 
        {2,1,1,1,s},
        "Dense_0_1", 
        true, 
        true,
        true
    );

    nn::layers::biaslayer bias_1
    (
        {2,1,1,1},
        "Bias_1",
        true, 
        true,
        true
    );

    nn::layers::activationlayer activ_1
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_1",
        true,
        true,
        true
    );

    nn::layers::nodelayer layer_1
    (
        {2,1,1,1,s},
        "Layer_1",
        true
    );

    nn::layers::fdbalayer fdba_0_1
    (
        dense_0_1,
        bias_1,
        activ_1
    );

    fdba_0_1.initialize(nn::core::initialization::he, util::core::distribution::normal);

    nn::layers::odernnlayer ode_1
    (
        {2,1,1,1,s},
        fdba_0_1,
        0.1,
        1
    );

    nn::layers::nodelayer target
    (
        {2,1,1,1,s},
        "Target",
        true
    );

    nn::layers::costlayer target_cost
    (
        bewusstsein::nn::core::cost::mean_squared_error,
        1
    );

    float r = 1;
    for (int i = 0; i < (s - 1); i++)
    {
        layer_0[i*2] = 1 - r * cos(util::ops::degrees_to_radians((float)i));
        layer_0[(i*2)+1] = r * sin(util::ops::degrees_to_radians((float)i));
        target[i*2] = 1 - (r-0.001) * cos(util::ops::degrees_to_radians((float)i+1));
        target[(i*2)+1] = (r-0.001) * sin(util::ops::degrees_to_radians((float)i+1));
        //target[(i*2)+2] = z_;

        //std::cout << "[" << target_1[i] << "," << target_1[i+1] << "]" << std::endl;
        r -= 0.001;
        //z_ -= 1/samples;
    }

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = std::numeric_limits<float>::epsilon();
    int batch_size = s;

    std::vector<double> x, y, xt, yt;

    while (true)
    {
        //--------------------------------------------------
        ode_1.inference(layer_0, layer_1);
        //--------------------------------------------------

        //--------------------------------------------------
        target_cost.backpropagation(layer_1, target);
        ode_1.backpropagation(layer_0, layer_1);
        //--------------------------------------------------

        //--------------------------------------------------
        ode_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //--------------------------------------------------

        x.clear();
        y.clear();
        xt.clear();
        yt.clear();

        for (int i = 0; i < (s - 1); i++)
        {
            x.push_back(layer_1[2*i]);
            y.push_back(layer_1[(2*i)+1]);
            xt.push_back(target[2*i]);
            yt.push_back(target[(2*i)+1]);
        }

        layer_1.print();
        target.print();
        
        plt::plot(x, y);
        plt::plot(xt, yt);
        plt::pause(0.01);
        plt::clf();
    }
}