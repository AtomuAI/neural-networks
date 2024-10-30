#include "../source/bewusstsein_nn.h"

int main()
{
    using namespace bewusstsein;
    
    int batch_size = 100;
    int epochs = 1;
    int iterations = 1;

    float step_size = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.99;
    float epsilon = 1e-8;

    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::duration<double, std::milli> elapsed;

    // 0 : Input
    //--------------------------------------------------
    nn::layers::nodelayer layer_0
    (
        {28,28,1,1,batch_size},
        0,
        "Layer 0",
        true
    );
    nn::layers::normlayer norm_0
    (
        nn::core::batch_norm,
        layer_0.get_shape(),
        true,
        true,
        true
    );
    //--------------------------------------------------

    // 1 : Feature Extraction
    //--------------------------------------------------
    nn::layers::convlayer conv_0_1
    (
        {3,3,1,64}, 
        0,
        "Conv 0 -> 1", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::valid, 
        nn::core::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_1
    (
        {26,26,1,64,batch_size},
        0,
        "Layer 1",
        true
    );
    nn::layers::biaslayer bias_1
    (
        {26,26,1,64},
        0,
        "Bias 1",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_1
    (
        nn::core::leakyrelu,
        0.1
    );
    nn::layers::normlayer norm_1
    (
        nn::core::batch_norm,
        layer_1.get_shape(),
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_1
    (
        {26,26,1,64},
        0.1
    );
    //--------------------------------------------------

    // 2 : Feature Extraction
    //--------------------------------------------------
    nn::layers::convlayer conv_1_2
    (
        {3,3,1,128}, 
        0,
        "Conv 0 -> 1", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::valid, 
        nn::core::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_2
    (
        {24,24,1,128,batch_size},
        0,
        "Layer 1",
        true
    );
    nn::layers::biaslayer bias_2
    (
        {24,24,1,128},
        0,
        "Bias 1",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_2
    (
        nn::core::leakyrelu,
        0.1
    );
    nn::layers::normlayer norm_2
    (
        nn::core::batch_norm,
        layer_2.get_shape(),
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_2
    (
        {24,24,1,128},
        0.1
    );
    //--------------------------------------------------

    // 3 : Feature Pool
    //--------------------------------------------------
    nn::layers::poolinglayer pool_2_3
    (
        {2,2,1}, 
        "Pool", 
        {2,2,0}, 
        {1,1,1}, 
        nn::core::max
    );
    nn::layers::nodelayer layer_3
    (
        {12,12,1,128,batch_size},
        0,
        "Layer 1",
        true
    );
    //--------------------------------------------------

    // 4 : Feature Extraction
    //--------------------------------------------------
    nn::layers::convlayer conv_3_4
    (
        {3,3,1,256}, 
        0,
        "Conv 0 -> 1", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::valid, 
        nn::core::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_4
    (
        {10,10,1,256,batch_size},
        0,
        "Layer 1",
        true
    );
    nn::layers::biaslayer bias_4
    (
        {10,10,1,256},
        0,
        "Bias 1",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_4
    (
        nn::core::leakyrelu,
        0.1
    );
    nn::layers::normlayer norm_4
    (
        nn::core::batch_norm,
        layer_4.get_shape(),
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_4
    (
        {10,10,1,256},
        0.1
    );
    //--------------------------------------------------

    // 5 : Feature Projection
    //--------------------------------------------------
    nn::layers::denselayer dense_4_5
    (
        {10,10,1,256}, 
        {10,1,1,1},
        0.01, 
        "Dense", 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_5
    (
        {10,1,1,1,batch_size}, 
        0, 
        "Dense Output", 
        true
    );
    nn::layers::biaslayer bias_5
    (
        {10,1,1,1},
        0,
        "Bias 1",
        true, 
        true,
        true
    );
    nn::layers::dropoutlayer dropout_5
    (
        {10,1,1,1},
        0.1
    );
    //--------------------------------------------------

    // 6 : Classification
    //--------------------------------------------------
    nn::layers::softmaxlayer softmax_5_6;
    nn::layers::nodelayer layer_6
    (
        {10,1,1,1,batch_size}, 
        0, 
        "Dense Output", 
        true
    );
    //--------------------------------------------------

    // 7 : Error
    //--------------------------------------------------
    nn::layers::nodelayer target
    (
        {10,1,1,1,batch_size}, 
        0, 
        "Target", 
        true
    );
    nn::layers::costlayer cost_target
    (
        bewusstsein::nn::core::softmax_categorical_cross_entropy,
        iterations
    );
    //--------------------------------------------------   

    // 0 : Input
    //--------------------------------------------------
    layer_0.info();
    //norm_0.info();
    //--------------------------------------------------

    // 1 : Feature Extraction
    //--------------------------------------------------
    conv_0_1.info();
    layer_1.info();
    bias_1.info();
    //activ_1.info();
    //norm_1.info();
    dropout_1.info();
    //--------------------------------------------------

    // 2 : Feature Extraction
    //--------------------------------------------------
    conv_1_2.info();
    layer_2.info();
    bias_2.info();
    //activ_2.info();
    //norm_2.info();
    dropout_2.info();
    //--------------------------------------------------

    // 3 : Feature Pool
    //--------------------------------------------------
    pool_2_3.info();
    layer_3.info();
    //--------------------------------------------------

    // 4 : Feature Extraction
    //--------------------------------------------------
    conv_3_4.info();
    layer_4.info();
    bias_4.info();
    //activ_4.info();
    //norm_4.info();
    dropout_4.info();
    //--------------------------------------------------

    // 5 : Feature Projection
    //--------------------------------------------------
    dense_4_5.info();
    layer_5.info();
    bias_5.info();
    dropout_5.info();
    //--------------------------------------------------

    // 6 : Classification
    //--------------------------------------------------
    //softmax_5_6.info();
    layer_6.info();
    //--------------------------------------------------

    // 7 : Error
    //--------------------------------------------------
    target.info();
    //cost_target.info();
    //--------------------------------------------------     

    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < iterations; i++)
        {
            // Load Batch
            //==================================================
            std::vector<std::string> batch(batch_size);
            for (long b = 0; b < batch_size; b++)
            {
                //batch[b] = 
            }
            //==================================================

            // Inference
            //==================================================
            
                // 0 : Input
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                layer_0.randomize(-1.0, 1.0);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "layer_0.randomize" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                //layer_0.load_images();
                    start = std::chrono::high_resolution_clock::now();
                norm_0.stat_analysis(layer_0);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "norm_0.stat_analysis" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                norm_0.inference(layer_0);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "norm_0.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                conv_0_1.inference(layer_0, layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "conv_0_1.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                bias_1.inference(layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "bias_1.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                activ_1.inference(layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "activ_1.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                norm_1.stat_analysis(layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "norm_1.stat_analysis" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                norm_1.inference(layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "norm_1.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                dropout_1.inference(layer_1);
                    end = std::chrono::high_resolution_clock::now();elapsed = end - start;
                    std::cout << "dropout_1.inference" << std::endl;std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                conv_1_2.inference(layer_1, layer_2);
                    start = std::chrono::high_resolution_clock::now();
                bias_2.inference(layer_2);
                    start = std::chrono::high_resolution_clock::now();
                activ_2.inference(layer_2);
                    start = std::chrono::high_resolution_clock::now();
                norm_2.stat_analysis(layer_2);
                    start = std::chrono::high_resolution_clock::now();
                norm_2.inference(layer_2);
                    start = std::chrono::high_resolution_clock::now();
                dropout_2.inference(layer_2);
                //--------------------------------------------------

                // 3 : Feature Pool
                //-------------------------------------------------- 
                    start = std::chrono::high_resolution_clock::now();       
                pool_2_3.inference(layer_2, layer_3);
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                conv_3_4.inference(layer_3, layer_4);
                    start = std::chrono::high_resolution_clock::now();
                bias_4.inference(layer_4);
                    start = std::chrono::high_resolution_clock::now();
                activ_4.inference(layer_4);
                    start = std::chrono::high_resolution_clock::now();
                norm_4.stat_analysis(layer_4);
                    start = std::chrono::high_resolution_clock::now();
                norm_4.inference(layer_4);
                    start = std::chrono::high_resolution_clock::now();
                dropout_4.inference(layer_4);
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                dense_4_5.inference(layer_4, layer_5);
                    start = std::chrono::high_resolution_clock::now();
                bias_5.inference(layer_5);
                    start = std::chrono::high_resolution_clock::now();
                dropout_5.inference(layer_5);
                //--------------------------------------------------

                // 6 : Classification
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();
                softmax_5_6.inference(layer_5, layer_6);
                //--------------------------------------------------

                // 7 : Error
                //--------------------------------------------------
                    start = std::chrono::high_resolution_clock::now();      
                cost_target.inference(layer_6, target);
                //--------------------------------------------------
            
            //==================================================

            

            start = std::chrono::high_resolution_clock::now();

            // Backpropagation
            //==================================================

                // 7 : Error
                //--------------------------------------------------
                cost_target.backpropagation(layer_6, target);
                //--------------------------------------------------
            
                // 6 : Classification
                //--------------------------------------------------
                //softmax_5_6.backpropagation(layer_5, layer_6);
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                dropout_5.backpropagation(layer_5);
                bias_5.backpropagation(layer_5);
                dense_4_5.backpropagation(layer_4, layer_5);
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                dropout_4.backpropagation(layer_4);
                norm_4.backpropagation(layer_4);
                activ_4.backpropagation(layer_4);
                bias_4.backpropagation(layer_4);
                conv_3_4.backpropagation(layer_3, layer_4);
                //--------------------------------------------------

                // 3 : Feature Pool
                //--------------------------------------------------
                pool_2_3.backpropagation(layer_2, layer_3);
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                dropout_2.backpropagation(layer_2);
                norm_2.backpropagation(layer_2);
                activ_2.backpropagation(layer_2);
                bias_2.backpropagation(layer_2);
                conv_1_2.backpropagation(layer_1, layer_2);
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                dropout_1.backpropagation(layer_1);
                norm_1.backpropagation(layer_1);
                activ_1.backpropagation(layer_1);
                bias_1.backpropagation(layer_1);
                conv_0_1.backpropagation(layer_0, layer_1);
                //--------------------------------------------------

                // 0 : Input
                //--------------------------------------------------
                norm_0.backpropagation(layer_0);
                //--------------------------------------------------

            //==================================================

            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Backpropagation" << std::endl;
            std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

            start = std::chrono::high_resolution_clock::now();

            // Gradient Decent
            //==================================================

                // 0 : Input
                //--------------------------------------------------
                norm_0.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                conv_0_1.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                bias_1.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //activ_1.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                norm_1.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                conv_1_2.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                bias_2.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //activ_2.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                norm_2.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                conv_3_4.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                bias_4.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //activ_4.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                norm_4.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                dense_4_5.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                bias_5.gradient_decent_adam(step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

            //==================================================

            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Gradient Decent" << std::endl;
            std::cout << "ms: " << elapsed.count() << std::endl << std::endl;
        }

        // Print
        //==================================================
            
            // 0 : Input
            //--------------------------------------------------
            //layer_0.print();
            //norm_0.print();
            //--------------------------------------------------

            // 1 : Feature Extraction
            //--------------------------------------------------
            //conv_0_1.print();
            //layer_1.print();
            //bias_1.print();
            //activ_1.print();
            //norm_1.print();
            //dropout_1.print();
            //--------------------------------------------------

            // 2 : Feature Extraction
            //--------------------------------------------------
            //conv_1_2.print();
            //layer_2.print();
            //bias_2.print();
            //activ_2.print();
            //norm_2.print();
            //dropout_2.print();
            //--------------------------------------------------

            // 3 : Feature Pool
            //--------------------------------------------------
            //layer_3.print();
            //--------------------------------------------------

            // 4 : Feature Extraction
            //--------------------------------------------------
            //conv_3_4.print();
            //layer_4.print();
            //bias_4.print();
            //activ_4.print();
            //norm_4.print();
            //dropout_4.print();
            //--------------------------------------------------

            // 5 : Feature Projection
            //--------------------------------------------------
            //dense_4_5.print();
            //layer_5.print();
            //bias_5.print();
            //dropout_5.print();
            //--------------------------------------------------

            // 6 : Classification
            //--------------------------------------------------
            //layer_6.print();
            //--------------------------------------------------

        //==================================================
    }
}