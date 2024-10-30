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
    using namespace bewusstsein;

    std::vector<int> labels = load_mnist_labels("/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_datasets/training/train-labels-idx1-ubyte");
    std::vector<int> eval_labels = load_mnist_labels("/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_datasets/inference/t10k-labels-idx1-ubyte");
    
    int batch_size = 100;
    int samples = 1;
    int epochs = 40;
    int examples = labels.size();
    int evaluations = eval_labels.size();
    
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
        "Layer 0",
        true
    );
    nn::layers::nodelayer layer_0_inference
    (
        {28,28,1,1,1},
        "Layer 0",
        true
    );
    nn::layers::normlayer norm_0
    (
        nn::core::normalization::batch_norm,
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
        {3,3,1,samples}, 
        "Convolution 0 -> 1", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::padding::valid, 
        nn::core::padding_value::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_1
    (
        {26,26,1,samples,batch_size},
        "Layer 1",
        true
    );
    nn::layers::nodelayer layer_1_inference
    (
        {26,26,1,samples,1},
        "Layer 1",
        true
    );
    nn::layers::biaslayer bias_1
    (
        {26,26,1,samples},
        "Bias 1",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_1
    (
        nn::core::activation::pararelu,
        {1,1,1,samples},
        0.1,
        "Activation 1",
        true,
        true,
        true
    );
    nn::layers::normlayer norm_1
    (
        nn::core::normalization::batch_norm,
        layer_1.get_shape(),
        "Normalization 1",
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_1
    (
        {26,26,1,samples},
        "Dropout 1",
        0.1
    );
    //--------------------------------------------------

    // 2 : Feature Extraction
    //--------------------------------------------------
    nn::layers::convlayer conv_1_2
    (
        {3,3,1,samples*2}, 
        "Convolution 1 -> 2", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::padding::valid, 
        nn::core::padding_value::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_2
    (
        {24,24,1,samples*2,batch_size},
        "Layer 2",
        true
    );
    nn::layers::nodelayer layer_2_inference
    (
        {24,24,1,samples*2,1},
        "Layer 2",
        true
    );
    nn::layers::biaslayer bias_2
    (
        {24,24,1,samples*2},
        "Bias 2",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_2
    (
        nn::core::activation::pararelu,
        {1,1,1,samples*2},
        0.1,
        "Activation 2",
        true,
        true,
        true
    );
    nn::layers::normlayer norm_2
    (
        nn::core::normalization::batch_norm,
        layer_2.get_shape(),
        "Normalization 2",
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_2
    (
        {24,24,1,samples*2},
        "Dropout 2",
        0.1
    );
    //--------------------------------------------------

    // 3 : Feature Pool
    //--------------------------------------------------
    nn::layers::poolinglayer pool_2_3
    (
        {2,2,1}, 
        "Pool 2 -> 3", 
        {2,2,0}, 
        {1,1,1}, 
        nn::core::pooling::max
    );
    nn::layers::nodelayer layer_3
    (
        {12,12,1,samples*2,batch_size},
        "Layer 3",
        true
    );
    nn::layers::nodelayer layer_3_inference
    (
        {12,12,1,samples*2,1},
        "Layer 3",
        true
    );
    //--------------------------------------------------

    // 4 : Feature Extraction
    //--------------------------------------------------
    nn::layers::convlayer conv_3_4
    (
        {3,3,1,samples*4}, 
        "Convolution 3 -> 4", 
        {1,1,0}, 
        {1,1,1}, 
        nn::core::padding::valid, 
        nn::core::padding_value::zero, 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_4
    (
        {10,10,1,samples*4,batch_size},
        "Layer 4",
        true
    );
    nn::layers::nodelayer layer_4_inference
    (
        {10,10,1,samples*4,1},
        "Layer 4",
        true
    );
    nn::layers::biaslayer bias_4
    (
        {10,10,1,samples*4},
        "Bias 4",
        true, 
        true,
        true
    );
    nn::layers::activationlayer activ_4
    (
        nn::core::activation::pararelu,
        {1,1,1,samples*4},
        0.1,
        "Activation 4",
        true,
        true,
        true
    );
    nn::layers::normlayer norm_4
    (
        nn::core::normalization::batch_norm,
        layer_4.get_shape(),
        "Normalization 4",
        true,
        true,
        true
    );
    nn::layers::dropoutlayer dropout_4
    (
        {10,10,1,samples*4},
        "Dropout 4",
        0.1
    );
    //--------------------------------------------------

    // 5 : Feature Projection
    //--------------------------------------------------
    nn::layers::denselayer dense_4_5
    (
        {10,10,1,samples*4}, 
        {10,1,1,1},
        "Dense 4 -> 5", 
        true, 
        true,
        true
    );
    nn::layers::nodelayer layer_5
    (
        {10,1,1,1,batch_size}, 
        "Layer 5", 
        true
    );
    nn::layers::nodelayer layer_5_inference
    (
        {10,1,1,1,1}, 
        "Layer 5", 
        true
    );
    nn::layers::biaslayer bias_5
    (
        {10,1,1,1},
        "Bias 5",
        true, 
        true,
        true
    );
    nn::layers::dropoutlayer dropout_5
    (
        {10,1,1,1},
        "Dropout 5",
        0.1
    );
    //--------------------------------------------------

    // 6 : Classification
    //--------------------------------------------------
    nn::layers::softmaxlayer softmax_5_6;
    //--------------------------------------------------

    // 7 : Error
    //--------------------------------------------------
    nn::layers::nodelayer target
    (
        {10,1,1,1,batch_size}, 
        "Target", 
        true
    );
    nn::layers::nodelayer target_inference
    (
        {10,1,1,1,1}, 
        "Target", 
        true
    );
    nn::layers::costlayer cost_target
    (
        bewusstsein::nn::core::cost::softmax_categorical_cross_entropy,
        examples
    );
    //--------------------------------------------------   

    // 1 : Feature Extraction
    //--------------------------------------------------
    conv_0_1.initialize(layer_0.get_shape(), layer_1.get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    layer_1.info();
    bias_1.randomize(0.01, 0.1);
    //--------------------------------------------------

    // 2 : Feature Extraction
    //--------------------------------------------------
    conv_1_2.initialize(layer_1.get_shape(), layer_2.get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    bias_2.randomize(0.01, 0.1);
    //--------------------------------------------------

    // 4 : Feature Extraction
    //--------------------------------------------------
    conv_3_4.initialize(layer_3.get_shape(), layer_4.get_shape(), nn::core::initialization::he, util::core::distribution::normal);
    bias_4.randomize(0.01, 0.1);
    //--------------------------------------------------

    // 5 : Feature Projection
    //--------------------------------------------------
    dense_4_5.initialize(nn::core::initialization::he, util::core::distribution::normal);
    bias_5.randomize(0.01, 0.1);
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

    // 7 : Error
    //--------------------------------------------------
    target.info();
    //cost_target.info();
    //--------------------------------------------------

    int best_epoch = 0;
    float best_accuracy = 0;

    std::vector<int> epochs_;
    std::vector<float> accuracies;

    int iterations = examples/batch_size;
    std::vector<std::string> batch(batch_size);

    float dropout_scale = 1.0f/(float)epochs;

    dropout_1.set_dropout(0.1);
    dropout_2.set_dropout(0.1);
    dropout_4.set_dropout(0.1);
    dropout_5.set_dropout(0.75);

    for (int e = 0; e < epochs; e++)
    {
        //float dropout_rate = e * dropout_scale;
        //dropout_1.set_dropout(dropout_rate);
        //dropout_2.set_dropout(dropout_rate);
        //dropout_4.set_dropout(dropout_rate);
        //dropout_5.set_dropout(dropout_rate);
    
        for (int i = 0; i < iterations; i++)
        {
            start = std::chrono::high_resolution_clock::now();

            target.zero();

            // Load Batch
            //==================================================
            batch.clear();
            for (int b = 0; b < batch_size; b++)
            {
                std::string filename = "/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/training/" + std::string(5 - std::to_string(((i * batch_size) + b) + 1).length(), '0') + std::to_string(((i * batch_size) + b) + 1) + ".bmp";
                batch.push_back(filename);
                target[(b * 10) + labels[(i * batch_size) + b]] = 1.0;
                //std::cout << "target[" << (b * 10) << " + " << labels[(i * batch_size) + b] << "] = " << 1.0 << std::endl;
            }
            //==================================================

            // Inference
            //==================================================
            
                // 0 : Input
                //--------------------------------------------------
                layer_0.load_images(batch, util::core::image_type::GRAYSCALE);\
                layer_0 * 255;
                //norm_0.stat_analysis(layer_0);
                //norm_0.inference(layer_0);
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                conv_0_1.inference(layer_0, layer_1);
                bias_1.inference(layer_1);
                activ_1.inference(layer_1);
                //norm_1.stat_analysis(layer_1);
                //norm_1.inference(layer_1);
                dropout_1.inference(layer_1);
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                conv_1_2.inference(layer_1, layer_2);
                bias_2.inference(layer_2);
                activ_2.inference(layer_2);
                //norm_2.stat_analysis(layer_2);
                //norm_2.inference(layer_2);
                dropout_2.inference(layer_2);
                //--------------------------------------------------

                // 3 : Feature Pool
                //--------------------------------------------------
                pool_2_3.inference(layer_2, layer_3);
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                conv_3_4.inference(layer_3, layer_4);
                bias_4.inference(layer_4);
                activ_4.inference(layer_4);
                //norm_4.stat_analysis(layer_4);
                //norm_4.inference(layer_4);
                dropout_4.inference(layer_4);
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                dense_4_5.inference(layer_4, layer_5);
                bias_5.inference(layer_5);
                dropout_5.inference(layer_5);
                //--------------------------------------------------

                // 6 : Classification
                //--------------------------------------------------
                softmax_5_6.inference(layer_5, layer_5);
                //--------------------------------------------------

                // 7 : Error
                //--------------------------------------------------
                //cost_target.inference(layer_5, target);
                //--------------------------------------------------
            
            //==================================================

            // Backpropagation
            //==================================================

                // 7 : Error
                //--------------------------------------------------
                cost_target.backpropagation(layer_5, target);
                //--------------------------------------------------
            
                // 6 : Classification
                //--------------------------------------------------
                //softmax_5_6.backpropagation(layer_5, layer_5);
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
                //norm_4.backpropagation(layer_4);
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
                //norm_2.backpropagation(layer_2);
                activ_2.backpropagation(layer_2);
                bias_2.backpropagation(layer_2);
                conv_1_2.backpropagation(layer_1, layer_2);
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                dropout_1.backpropagation(layer_1);
                //norm_1.backpropagation(layer_1);
                activ_1.backpropagation(layer_1);
                bias_1.backpropagation(layer_1);
                conv_0_1.backpropagation(layer_0, layer_1);
                //--------------------------------------------------

                // 0 : Input
                //--------------------------------------------------
                //norm_0.backpropagation(layer_0);
                //--------------------------------------------------

            //==================================================

            if(system("clear"))
            {
                throw("Terminal clear failure");
            }
            std::cout << "[";
            for (int p = 0; p < epochs_.size(); p++)
            {
                std::cout << "(" << epochs_[p] + 1 << ", " << accuracies[p] << ")"; 
                if (p < (epochs_.size() - 1)) {std::cout << ", ";}
            }
            std::cout << "]" << std::endl;
            std::cout << "Best Epoch: " << best_epoch + 1 << ", " << "Accuracy: " << best_accuracy << std::endl;
            std::cout << "Epoch: " << e + 1 << "/" << epochs << ", Batch: " << i + 1 << "/" << iterations << ", Example: " << (i * batch_size) + 1 << "/" << examples << std::endl;
            //std::cout << "Dropout Rate:" << dropout_rate << std::endl;
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "ms: " << elapsed.count() << std::endl << std::endl;

            // Print
            //==================================================
                
                // 0 : Input
                //--------------------------------------------------
                //norm_0.print();
                //layer_0.print();
                //layer_0.print_delta();
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                //conv_0_1.print();
                //conv_0_1.print_jacobian();
                //bias_1.print();
                //bias_1.print_jacobian();
                //activ_1.print();
                //activ_1.print_jacobian();
                //norm_1.print();
                //dropout_1.print();
                //layer_1.print();
                //layer_1.print_delta();
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                //conv_1_2.print();
                //conv_1_2.print_jacobian();
                //bias_2.print();
                //bias_2.print_jacobian();
                //activ_2.print();
                //activ_2.print_jacobian();
                //norm_2.print();
                //dropout_2.print();
                //layer_2.print();
                //layer_2.print_delta();
                //--------------------------------------------------

                // 3 : Feature Pool
                //--------------------------------------------------
                //layer_3.print();
                //layer_3.print_delta();
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                //conv_3_4.print();
                //conv_3_4.print_jacobian();
                //bias_4.print();
                //bias_4.print_jacobian();
                //activ_4.print();
                //activ_4.print_jacobian();
                //norm_4.print();
                //dropout_4.print();
                //layer_4.print();
                //layer_4.print_delta();
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                //dense_4_5.print();
                //dense_4_5.print_jacobian();
                //bias_5.print();
                //bias_5.print_jacobian();
                //dropout_5.print();
                //layer_5.print();
                //layer_5.print_delta();
                //--------------------------------------------------

                // 6 : Classification
                //--------------------------------------------------
                //target.print();
                //cost_target.info();
                //--------------------------------------------------

            //==================================================
            
            fflush(stdout);
            //std::this_thread::sleep_for(std::chrono::milliseconds(500));

            // Gradient Decent
            //==================================================

                // 0 : Input
                //--------------------------------------------------
                //norm_0.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 1 : Feature Extraction
                //--------------------------------------------------
                conv_0_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                bias_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                activ_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //norm_1.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 2 : Feature Extraction
                //--------------------------------------------------
                conv_1_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                bias_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                activ_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //norm_2.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 4 : Feature Extraction
                //--------------------------------------------------
                conv_3_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                bias_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                activ_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //norm_4.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

                // 5 : Feature Projection
                //--------------------------------------------------
                dense_4_5.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                bias_5.gradient_decent_adam(batch_size, step_size, beta1, beta2, epsilon);
                //--------------------------------------------------

            //==================================================
        }

        // Save
        //==================================================

            std::string path = "/home/shane/Documents/VSCode/bewusstsein_api/bewusstsein_neural_networks/models/test_2/";

            // 0 : Input
            //--------------------------------------------------
            //layer_0.save();
            norm_0.save(path + "norm_0_epoch[" + std::to_string(e) + "].norm");
            //--------------------------------------------------

            // 1 : Feature Extraction
            //--------------------------------------------------
            conv_0_1.save(path + "conv_0_1_epoch[" + std::to_string(e) + "].conv");
            //layer_1.save();
            bias_1.save(path + "bias_1_epoch[" + std::to_string(e) + "].bias");
            activ_1.save(path + "activ_1_epoch[" + std::to_string(e) + "].activ");
            norm_1.save(path + "norm_1_epoch[" + std::to_string(e) + "].norm");
            //dropout_1.save();
            //--------------------------------------------------

            // 2 : Feature Extraction
            //--------------------------------------------------
            conv_1_2.save(path + "conv_1_2_epoch[" + std::to_string(e) + "].conv");
            //layer_2.save();
            bias_2.save(path + "bias_2_epoch[" + std::to_string(e) + "].bias");
            activ_2.save(path + "activ_2_epoch[" + std::to_string(e) + "].activ");
            norm_2.save(path + "norm_2_epoch[" + std::to_string(e) + "].norm");
            //dropout_2.save();
            //--------------------------------------------------

            // 3 : Feature Pool
            //--------------------------------------------------
            //pool_2_3.save());
            //layer_3.save();
            //--------------------------------------------------

            // 4 : Feature Extraction
            //--------------------------------------------------
            conv_3_4.save(path + "conv_3_4_epoch[" + std::to_string(e) + "].conv");
            //layer_4.save();
            bias_4.save(path + "bias_4_epoch[" + std::to_string(e) + "].bias");
            activ_4.save(path + "activ_4_epoch[" + std::to_string(e) + "].activ");
            norm_4.save(path + "norm_4_epoch[" + std::to_string(e) + "].norm");
            //dropout_4.save(std::ofstream();
            //--------------------------------------------------

            // 5 : Feature Projection
            //--------------------------------------------------
            dense_4_5.save(path + "dense_4_5_epoch[" + std::to_string(e) + "].dense");
            //layer_5.save();
            bias_5.save(path + "bias_5_epoch[" + std::to_string(e) + "].bias");
            //dropout_5.save();
            //--------------------------------------------------

            // 6 : Classification
            //--------------------------------------------------
            //softmax_5_6.info();
            //layer_6.save();
            //--------------------------------------------------

            // 7 : Error
            //--------------------------------------------------
            //target.save();
            //cost_target.info();
            //--------------------------------------------------   

        //==================================================

        std::vector<std::string> file_name;
        int correct_count = 0;
        float inference_error = 0;
        float accuracy = 0;

        for (int i = 0; i < evaluations; i++)
        {
            file_name.clear();
            target_inference.zero();
            
            file_name.push_back("/home/shane/Documents/VSCode/AI/Neural_Network_API_Current/mnist_digit_images/inference/" + std::string(5 - std::to_string(i + 1).length(), '0') + std::to_string(i + 1) + ".bmp");
            //std::cout << file_name[0] << std::endl << std::endl;
            target_inference[eval_labels[i]] = 1.0;

        // Inference Test
        //==================================================

            // 0 : Input
            //--------------------------------------------------
            layer_0_inference.load_images(file_name, util::core::image_type::GRAYSCALE);
            norm_0.inference(layer_0);
            //--------------------------------------------------

            // 1 : Feature Extraction
            //--------------------------------------------------
            conv_0_1.inference(layer_0_inference, layer_1_inference);
            bias_1.inference(layer_1_inference);
            activ_1.inference(layer_1_inference);
            norm_1.inference(layer_1);
            //--------------------------------------------------

            // 2 : Feature Extraction
            //--------------------------------------------------
            conv_1_2.inference(layer_1_inference, layer_2_inference);
            bias_2.inference(layer_2_inference);
            activ_2.inference(layer_2_inference);
            norm_2.inference(layer_2);
            //--------------------------------------------------

            // 3 : Feature Pool
            //--------------------------------------------------
            pool_2_3.inference(layer_2_inference, layer_3_inference);
            //--------------------------------------------------

            // 4 : Feature Extraction
            //--------------------------------------------------
            conv_3_4.inference(layer_3_inference, layer_4_inference);
            bias_4.inference(layer_4_inference);
            activ_4.inference(layer_4_inference);
            norm_4.inference(layer_4);
            //--------------------------------------------------

            // 5 : Feature Projection
            //--------------------------------------------------
            dense_4_5.inference(layer_4_inference, layer_5_inference);
            bias_5.inference(layer_5_inference);
            //--------------------------------------------------

            // 6 : Classification
            //--------------------------------------------------
            softmax_5_6.inference(layer_5_inference, layer_5_inference);
            //--------------------------------------------------

            // 7 : Error
            //--------------------------------------------------
            //cost_target.inference(layer_5, target);
            //--------------------------------------------------
        
        //==================================================

            float max_node = 0;
            int max_index = 0;
            for (int j = 0; j < layer_5_inference.get_size(); j++)
            {
                float node = layer_5_inference[j];
                if (node > max_node)
                {
                    max_node = node;
                    max_index = j;
                }
            }
            if (max_index == eval_labels[i])
            {
                correct_count++;
            }
            accuracy = ((double)correct_count / (double)(i + 1)) * (double)100;

            if(system("clear"))
            {
                throw("Terminal clear failure");
            }
            std::cout << "[";
            for (int p = 0; p < epochs_.size(); p++)
            {
                std::cout << "(" << epochs_[p] + 1 << ", " << accuracies[p] << ")"; 
                if (p < (epochs_.size() - 1)) {std::cout << ", ";}
            }
            std::cout << "]" << std::endl;
            std::cout << "Best Epoch: " << best_epoch + 1 << ", " << "Accuracy: " << best_accuracy << std::endl;
            std::cout << "example: " << i + 1 << "/" << iterations << std::endl;
            std::cout << "accuracy: " << accuracy << " = " << correct_count << "/" << i + 1 << std::endl;
            std::cout << "correct: " << correct_count << "/" << iterations << std::endl;
            fflush(stdout);

        // Print
        //==================================================
            
            // 0 : Input
            //--------------------------------------------------
            //norm_0.print();
            //layer_0.print();
            //layer_0.print_delta();
            //--------------------------------------------------

            // 1 : Feature Extraction
            //--------------------------------------------------
            //conv_0_1.print();
            //bias_1.print();
            //activ_1.print();
            //norm_1.print();
            //layer_1.print();
            //layer_1.print_delta();
            //--------------------------------------------------

            // 2 : Feature Extraction
            //--------------------------------------------------
            //conv_1_2.print();
            //bias_2.print();
            //activ_2.print();
            //norm_2.print();
            //layer_2.print();
            //layer_2.print_delta();
            //--------------------------------------------------

            // 3 : Feature Pool
            //--------------------------------------------------
            //layer_3.print();
            //layer_3.print_delta();
            //--------------------------------------------------

            // 4 : Feature Extraction
            //--------------------------------------------------
            //conv_3_4.print();
            //bias_4.print();
            //activ_4.print();
            //norm_4.print();
            //layer_4.print();
            //layer_4.print_delta();
            //--------------------------------------------------

            // 5 : Feature Projection
            //--------------------------------------------------
            //dense_4_5.print();
            //bias_5.print();
            //layer_5_inference.print();
            //std::cout << std::endl << "Prediction: " << max_index << "Truth: " << labels[i] << std::endl;
            //layer_5.print_delta();
            //--------------------------------------------------

            // 6 : Classification
            //--------------------------------------------------
            //target_inference.print();
            //--------------------------------------------------

        //==================================================

        }

        if (accuracy > best_accuracy)
        {
            best_accuracy = accuracy;
            best_epoch = e;
        }

        accuracies.push_back(accuracy);
        epochs_.push_back(e);
    }

    if(system("clear"))
    {
        throw("Terminal clear failure");
    }
    std::cout << "[";
    for (int p = 0; p < epochs_.size(); p++)
    {
        std::cout << "(" << epochs_[p] + 1 << ", " << accuracies[p] << ")"; 
        if (p < (epochs_.size() - 1)) {std::cout << ", ";}
    }
    std::cout << "]" << std::endl;
    std::cout << "Best Epoch: " << best_epoch + 1 << ", " << "Accuracy: " << best_accuracy << std::endl;
    fflush(stdout);
}