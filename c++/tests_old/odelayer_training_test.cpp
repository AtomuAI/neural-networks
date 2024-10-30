#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;
    namespace plt = matplotlibcpp;

    nn::layers::nodelayer layer_0
    (
        {2,1,1,1,360},
        "Layer_0",
        true
    );

    nn::layers::denselayer dense_0_1
    (
        {2,1,1,1,360}, 
        {100,1,1,1,360},
        "Dense_0_1", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_1
    (
        {100,1,1,1,360},
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

    nn::layers::odernnlayer ode_1
    (
        {100,1,1,1,360},
        "Ltc_1"
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
        {100,1,1,1,360}, 
        {100,1,1,1,360},
        "Dense_1_2", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_2
    (
        {100,1,1,1,360},
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

    nn::layers::odernnlayer ode_2
    (
        {100,1,1,1,360}, 
        "Ltc_2"
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
        {100,1,1,1,360}, 
        {100,1,1,1,360},
        "Dense_2_3", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_3
    (
        {100,1,1,1,360},
        "Layer_3",
        true
    );

    nn::layers::biaslayer bias_3
    (
        {100,1,1,1},
        "Bias 3",
        true, 
        true,
        true
    );

    nn::layers::odernnlayer ode_3
    (
        {100,1,1,1,360}, 
        "Ltc_3"
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

    nn::layers::denselayer dense_3_4
    (
        {100,1,1,1,360}, 
        {100,1,1,1,360},
        "Dense_3_4", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_4
    (
        {100,1,1,1,360},
        "Layer_4",
        true
    );

    nn::layers::biaslayer bias_4
    (
        {100,1,1,1},
        "Bias 4",
        true, 
        true,
        true
    );

    nn::layers::odernnlayer ode_4
    (
        {100,1,1,1,360}, 
        "Ltc_4"
    );

    nn::layers::activationlayer activ_4
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_4",
        true,
        true,
        true
    );

    nn::layers::denselayer dense_4_5
    (
        {100,1,1,1,360}, 
        {100,1,1,1,360},
        "Dense_4_5", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_5
    (
        {100,1,1,1,360},
        "Layer_5",
        true
    );

    nn::layers::biaslayer bias_5
    (
        {100,1,1,1},
        "Bias 5",
        true, 
        true,
        true
    );

    nn::layers::odernnlayer ode_5
    (
        {100,1,1,1,360},
        "Ltc_5"
    );

    nn::layers::activationlayer activ_5
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_5",
        true,
        true,
        true
    );

    nn::layers::denselayer dense_5_6
    (
        {100,1,1,1,360}, 
        {2,1,1,1,360},
        "Dense_5_6", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_6
    (
        {2,1,1,1,360},
        "Layer_6",
        true
    );

    nn::layers::biaslayer bias_6
    (
        {2,1,1,1},
        "Bias 6",
        true, 
        true,
        true
    );

    nn::layers::odernnlayer ode_6
    (
        {2,1,1,1,360}, 
        "Ltc_6"
    );

    nn::layers::activationlayer activ_6
    (
        nn::core::activation::tanh,
        {1,1,1,1},
        0.1,
        "Activation_6",
        true,
        true,
        true
    );

    nn::layers::nodelayer target_6
    (
        {2,1,1,1,360},
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
    dense_2_3.initialize(nn::core::initialization::he, util::core::distribution::normal);
    dense_3_4.initialize(nn::core::initialization::he, util::core::distribution::normal);
    dense_4_5.initialize(nn::core::initialization::he, util::core::distribution::normal);
    dense_5_6.initialize(nn::core::initialization::he, util::core::distribution::normal);

    float r = 1;
    for (int i = 0; i < 360; i++)
    {
        layer_0[i*2] = 1 - (r * cos(util::ops::degrees_to_radians((float)i-1)));
        layer_0[(i*2)+1] = r * sin(util::ops::degrees_to_radians((float)i-1));
        target_6[i*2] = 1 - (r * cos(util::ops::degrees_to_radians((float)i)));
        target_6[(i*2)+1] = r * sin(util::ops::degrees_to_radians((float)i));
        //std::cout << "[" << target_1[i] << "," << target_1[i+1] << "]" << std::endl;
        r -= 0.001;
    }

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = std::numeric_limits<float>::epsilon();
    int batch_size = 360;

    std::vector<double> x, y, xt, yt;

    while (true)
    {
        dense_0_1.inference(layer_0, layer_1);
        bias_1.inference(layer_1);
        activ_1.inference(layer_1);
        ode_1.inference(layer_1);

        //dense_1_2.inference(layer_1, layer_2);
        //bias_2.inference(layer_2);
        //activ_2.inference(layer_2);
        //ode_2.inference(layer_2);

        //dense_2_3.inference(layer_2, layer_3);
        //bias_3.inference(layer_3);
        //activ_3.inference(layer_3);
        //ode_3.inference(layer_3);

        //dense_3_4.inference(layer_3, layer_4);
        //bias_4.inference(layer_4);
        //activ_4.inference(layer_4);
        //ode_4.inference(layer_4);

        //dense_4_5.inference(layer_4, layer_5);
        //bias_5.inference(layer_5);
        //activ_5.inference(layer_5);
        //ode_5.inference(layer_5);

        dense_5_6.inference(layer_1, layer_6);
        bias_6.inference(layer_6);
        activ_6.inference(layer_6);
        ode_6.inference(layer_6);

        target_cost.backpropagation(layer_6, target_6);

        ode_6.backpropagation(layer_6);
        activ_6.backpropagation(layer_6);
        bias_6.backpropagation(layer_6);
        dense_5_6.backpropagation(layer_1, layer_6);

        //ode_5.backpropagation(layer_5);
        //activ_5.backpropagation(layer_5);
        //bias_5.backpropagation(layer_5);
        //dense_4_5.backpropagation(layer_4, layer_5);

        //ode_4.backpropagation(layer_4);
        //activ_4.backpropagation(layer_4);
        //bias_4.backpropagation(layer_4);
        //dense_3_4.backpropagation(layer_3, layer_4);

        //ode_3.backpropagation(layer_3);
        //activ_3.backpropagation(layer_3);
        //bias_3.backpropagation(layer_3);
        //dense_2_3.backpropagation(layer_2, layer_3);

        //ode_2.backpropagation(layer_2);
        //activ_2.backpropagation(layer_2);
        //bias_2.backpropagation(layer_2);
        //dense_1_2.backpropagation(layer_1, layer_2);

        ode_1.backpropagation(layer_1);
        activ_1.backpropagation(layer_1);
        bias_1.backpropagation(layer_1);
        dense_0_1.backpropagation(layer_0, layer_1);
        
        //ode_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //ode_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //ode_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //ode_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //ode_5.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //ode_6.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

        dense_0_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //dense_1_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //dense_2_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //dense_3_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //dense_4_5.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        dense_5_6.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

        bias_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //bias_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //bias_3.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //bias_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        //bias_5.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
        bias_6.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);

        ode_1.zero();
        //ode_2.zero();
        //ode_3.zero();
        //ode_4.zero();
        //ode_5.zero();
        ode_6.zero();
        
        if(system("clear"))
        {
            throw("Terminal clear failure");
        }
        target_6.print();
        layer_6.print();
        fflush(stdout);

        x.clear();
        y.clear();
        xt.clear();
        yt.clear();

        for (int i = 0; i < 360; i++)
        {
            x.push_back(layer_6[2*i]);
            y.push_back(layer_6[(2*i)+1]);
            xt.push_back(target_6[2*i]);
            yt.push_back(target_6[(2*i)+1]);
        }

        plt::plot(x, y);
        plt::plot(xt, yt);
        plt::pause(0.01);
        plt::clf();
    }
}