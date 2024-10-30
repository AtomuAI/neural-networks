#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;
    namespace plt = matplotlibcpp;

    nn::layers::nodelayer layer_0
    (
        {1,1,1,1,1},
        "Layer_0",
        true
    );

    nn::layers::nodelayer layer_0_inf
    (
        {1,1,1,1,1},
        "Layer_0",
        true
    );

    nn::layers::denselayer dense_0_1
    (
        {2,1,1,1,1}, 
        {100,1,1,1,1},
        "Dense_0_1", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_1
    (
        {100,1,1,1,1},
        "Layer_1",
        true
    );

    nn::layers::nodelayer layer_1_inf
    (
        {100,1,1,1,1},
        "Layer_1",
        true
    );

    nn::layers::biaslayer bias_1
    (
        {100,1,1,1},
        "Bias 1",
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

    nn::layers::denselayer dense_1_2
    (
        {100,1,1,1,1}, 
        {100,1,1,1,1},
        "Dense_1_2", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_2
    (
        {100,1,1,1,1},
        "Layer_2",
        true
    );

    nn::layers::nodelayer layer_2_inf
    (
        {100,1,1,1,1},
        "Layer_2",
        true
    );

    nn::layers::biaslayer bias_2
    (
        {100,1,1,1},
        "Bias 2",
        true, 
        true,
        true
    );

    nn::layers::activationlayer activ_2
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_2",
        true,
        true,
        true
    );

    nn::layers::denselayer dense_2_3
    (
        {100,1,1,1,1}, 
        {2,1,1,1,1},
        "Dense_2_3", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_3
    (
        {2,1,1,1,1},
        "Layer_3",
        true
    );

    nn::layers::nodelayer layer_3_inf
    (
        {2,1,1,1,1},
        "Layer_3",
        true
    );

    nn::layers::biaslayer bias_3
    (
        {2,1,1,1},
        "Bias 3",
        true, 
        true,
        true
    );

    nn::layers::activationlayer activ_3
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_3",
        true,
        true,
        true
    );

    nn::layers::odernnlayer ode_3
    (
        {2,1,1,1,1},
        "Ltc_1"
    );

    nn::layers::nodelayer target
    (
        {2,1,1,1,1},
        "Target_6",
        true
    );

    nn::layers::costlayer target_cost
    (
        bewusstsein::nn::core::cost::mean_squared_error,
        1
    );

    dense_0_1.initialize(nn::core::initialization::he, util::core::distribution::normal);
    dense_1_2.initialize(nn::core::initialization::he, util::core::distribution::normal);

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = std::numeric_limits<float>::epsilon();
    int batch_size = 1080;

    std::vector<double> x, y, z, xt, yt, zt;

    while (true)
    {
        float r = 1;
        for (int i = 0; i < 1080; i++)
        {
            target[0] = 1 - (r * cos(util::ops::degrees_to_radians((float)i)));
            target[1] = r * sin(util::ops::degrees_to_radians((float)i));
            r -= 0.001;

            //--------------------------------------------------
            dense_0_1.inference(layer_3, layer_1);
            bias_1.inference(layer_1);
            activ_1.inference(layer_1);

            dense_1_2.inference(layer_1, layer_2);
            bias_1.inference(layer_2);
            activ_2.inference(layer_2);

            dense_2_3.inference(layer_2, layer_3);
            bias_3.inference(layer_3);
            activ_3.inference(layer_3);

            ode_3.inference(layer_3);
            //--------------------------------------------------

            //--------------------------------------------------
            target_cost.backpropagation(layer_3, target);

            ode_3.backpropagation(layer_3);

            activ_3.backpropagation(layer_3);
            bias_3.backpropagation(layer_3);
            dense_2_3.backpropagation(layer_2, layer_3);

            activ_2.backpropagation(layer_2);
            bias_2.backpropagation(layer_2);
            dense_1_2.backpropagation(layer_1, layer_2);

            activ_1.backpropagation(layer_1);
            bias_1.backpropagation(layer_1);
            dense_0_1.backpropagation(layer_3, layer_1);
            //--------------------------------------------------

            //--------------------------------------------------
            dense_0_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
            bias_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

            dense_1_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
            bias_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

            dense_2_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
            bias_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

            //ode_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
            //--------------------------------------------------

            x.push_back(layer_3[0]);
            y.push_back(layer_3[1]);
            xt.push_back(target[0]);
            yt.push_back(target[1]);
            
            plt::plot(x, y);
            plt::plot(xt, yt);
            plt::pause(0.01);
            plt::clf();
        }

        ode_3.zero();

        x.clear();
        y.clear();
        xt.clear();
        yt.clear();
    }
}