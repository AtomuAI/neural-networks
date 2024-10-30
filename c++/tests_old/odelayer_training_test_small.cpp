#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;
    namespace plt = matplotlibcpp;

    int samples = 1080;

    nn::layers::nodelayer layer_0
    (
        {1,1,1,1,samples},
        "Layer_0",
        true
    );

    nn::layers::nodelayer layer_0_inf
    (
        {1,1,1,1,samples},
        "Layer_0",
        true
    );

    nn::layers::denselayer dense_0_1
    (
        {2,1,1,1,samples}, 
        {100,1,1,1,samples},
        "Dense_0_1", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_1
    (
        {100,1,1,1,samples},
        "Layer_1",
        true
    );

    nn::layers::nodelayer layer_1_inf
    (
        {100,1,1,1,samples},
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
        {100,1,1,1,samples}, 
        {100,1,1,1,samples},
        "Dense_1_2", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_2
    (
        {100,1,1,1,samples},
        "Layer_2",
        true
    );

    nn::layers::nodelayer layer_2_inf
    (
        {100,1,1,1,samples},
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
        {100,1,1,1,samples}, 
        {2,1,1,1,samples},
        "Dense_2_3", 
        true, 
        true,
        true
    );

    nn::layers::nodelayer layer_3
    (
        {2,1,1,1,samples},
        "Layer_3",
        true
    );

    nn::layers::nodelayer layer_3_inf
    (
        {2,1,1,1,samples},
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
        {2,1,1,1,samples},
        "Ltc_1"
    );

    nn::layers::nodelayer target
    (
        {2,1,1,1,samples},
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

    for (int i = 0; i < samples; i++)
    {
        layer_0[i] = i + 1;
    }

    float r = 1;
    //float z_ = 1;
    for (int i = 0; i < (samples - 1); i++)
    {
        target[i*2] = 1 - (r * cos(util::ops::degrees_to_radians((float)i)));
        target[(i*2)+1] = r * sin(util::ops::degrees_to_radians((float)i));
        //target[(i*2)+2] = z_;

        //std::cout << "[" << target_1[i] << "," << target_1[i+1] << "]" << std::endl;
        r -= 0.001;
        //z_ -= 1/samples;
    }

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = std::numeric_limits<float>::epsilon();
    int batch_size = samples;

    std::vector<double> x, y, z, xt, yt, zt;

    while (true)
    {
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

        ode_3.zero();
        
        /*
        if(system("clear"))
        {
            throw("Terminal clear failure");
        }
        std::cout << i << std::endl;
        //target.print();
        //layer_1.print();
        fflush(stdout);
        */

        x.clear();
        y.clear();
        //z.clear();
        xt.clear();
        yt.clear();
        //zt.clear();

        for (int i = 0; i < (samples - 1); i++)
        {
            x.push_back(layer_3[2*i]);
            y.push_back(layer_3[(2*i)+1]);
            //z.push_back(layer_1[(2*i)+2]);
            xt.push_back(target[2*i]);
            yt.push_back(target[(2*i)+1]);
            //zt.push_back(target[(2*i)+2]);
        }

        plt::plot(x, y);
        plt::plot(xt, yt);
        plt::pause(0.01);
        plt::clf();

        /*
        // Plot the joints
        std::map<std::string, std::string> keywords;
        keywords["color"] = "r";
        keywords["marker"] = "o";
        plt::plot3(x_values, y_values, z_values, keywords);
        plt::xlabel("X-axis");
        plt::ylabel("Y-axis");
        plt::set_zlabel("Z-axis");
        plt::xlim(-3, 3); // Set x-axis limits from -10 to 10
        plt::ylim(-3, 3); // Set y-axis limits from -10 to 10
        plt::set_zlim(-3, 3); // Set y-axis limits from -10 to 10
        
        plt::show();
        */
    }

    /*
    x.clear();
    y.clear();

    for (int i = 0; i < samples; i++)
    {
        dense_0_1.inference(layer_2_inf, layer_1_inf);
        bias_1.inference(layer_1_inf);
        dense_0_1.inference(layer_1_inf, layer_2_inf);
        bias_1.inference(layer_2_inf);
        activ_2.inference(layer_2_inf);
        ode_2.inference(layer_2_inf);

        x.push_back(layer_2[2*i]);
        y.push_back(layer_2[(2*i)+1]);

        plt::plot(x, y);
        plt::clf();
    }
    */
}