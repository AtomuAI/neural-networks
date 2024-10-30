// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_CORE_PRINT_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_CORE_PRINT_HPP_

#include <vector>
#include <iostream>
#include <iomanip>

#include "bewusstsein_neural_networks/c++/include/bewusstsein_nn.hpp"

template <typename T>
void print_value(const T& value)
{
    std::cout << value;
}

template <>
void print_value<bool>(const bool& value)
{
    std::cout << std::boolalpha << value;
}

template <>
void print_value<int>(const int& value)
{
    std::cout << value;
}

template <>
void print_value<float>(const float& value)
{
    std::cout << std::fixed << std::setprecision(6) << value;
}

template <>
void print_value<double>(const double& value)
{
    std::cout << std::fixed << std::setprecision(12) << value;
}

template <typename T>
void print_tensor_info(const nn::Tensor<T, 1>& tensor)
{
    const nn::Shape<1>& shape = tensor.get_shape();
    const std::vector<typename std::conditional<nn::is_bool<T>::value, nn::u8, T>::type>& data = tensor.get_vector();

    std::cout << "Data Type: " << typeid(T).name() << std::endl;
    std::cout << "Dimensions: " << 1 << std::endl;
    std::cout << "Shape: [ ";
    std::cout << "Width: " << shape.width() << ", ";
    std::cout << " ]" << std::endl;
    std::cout << "Total Elements: " << data.size() << std::endl;
    std::cout << "Memory Usage: " << data.size() * sizeof(T) << " bytes" << std::endl;
}

template <typename T>
void print_tensor_info(const nn::Tensor<T, 2>& tensor)
{
    const nn::Shape<2>& shape = tensor.get_shape();
    const std::vector<typename std::conditional<nn::is_bool<T>::value, nn::u8, T>::type>& data = tensor.get_vector();

    std::cout << "Data Type: " << typeid(T).name() << std::endl;
    std::cout << "Dimensions: " << 2 << std::endl;
    std::cout << "Shape: [ ";
    std::cout << "Width: " << shape.width() << ", ";
    std::cout << "Height: " << shape.height() << ", ";
    std::cout << " ]" << std::endl;
    std::cout << "Total Elements: " << data.size() << std::endl;
    std::cout << "Memory Usage: " << data.size() * sizeof(T) << " bytes" << std::endl;
}

template <typename T>
void print_tensor_info(const nn::Tensor<T, 3>& tensor)
{
    const nn::Shape<3>& shape = tensor.get_shape();
    const std::vector<typename std::conditional<nn::is_bool<T>::value, nn::u8, T>::type>& data = tensor.get_vector();

    std::cout << "Data Type: " << typeid(T).name() << std::endl;
    std::cout << "Dimensions: " << 3 << std::endl;
    std::cout << "Shape: [ ";
    std::cout << "Width: " << shape.width() << ", ";
    std::cout << "Height: " << shape.height() << ", ";
    std::cout << "Depth: " << shape.depth() << ", ";
    std::cout << " ]" << std::endl;
    std::cout << "Total Elements: " << data.size() << std::endl;
    std::cout << "Memory Usage: " << data.size() * sizeof(T) << " bytes" << std::endl;
}

template <typename T>
void print_tensor_info(const nn::Tensor<T, 4>& tensor)
{
    const nn::Shape<4>& shape = tensor.get_shape();
    const std::vector<typename std::conditional<nn::is_bool<T>::value, nn::u8, T>::type>& data = tensor.get_vector();

    std::cout << "Data Type: " << typeid(T).name() << std::endl;
    std::cout << "Dimensions: " << 4 << std::endl;
    std::cout << "Shape: [ ";
    std::cout << "Width: " << shape.width() << ", ";
    std::cout << "Height: " << shape.height() << ", ";
    std::cout << "Depth: " << shape.depth() << ", ";
    std::cout << "Channels: " << shape.channels() << ", ";
    std::cout << " ]" << std::endl;
    std::cout << "Total Elements: " << data.size() << std::endl;
    std::cout << "Memory Usage: " << data.size() * sizeof(T) << " bytes" << std::endl;
}

template <typename T>
void print_tensor_info(const nn::Tensor<T, 5>& tensor)
{
    const nn::Shape<5>& shape = tensor.get_shape();
    const std::vector<typename std::conditional<nn::is_bool<T>::value, nn::u8, T>::type>& data = tensor.get_vector();

    std::cout << "Data Type: " << typeid(T).name() << std::endl;
    std::cout << "Dimensions: " << 5 << std::endl;
    std::cout << "Shape: [ ";
    std::cout << "Width: " << shape.width() << ", ";
    std::cout << "Height: " << shape.height() << ", ";
    std::cout << "Depth: " << shape.depth() << ", ";
    std::cout << "Channels: " << shape.channels() << ", ";
    std::cout << "Batches: " << shape.batches();
    std::cout << " ]" << std::endl;
    std::cout << "Total Elements: " << data.size() << std::endl;
    std::cout << "Memory Usage: " << data.size() * sizeof(T) << " bytes" << std::endl;
}

template <typename T>
void print_tensor_data(const nn::Tensor<T, 1>& tensor) {
    std::cout << "[ ";
    for (nn::Dim i = 0; i < tensor.get_shape().width(); ++i) {
        print_value( tensor[i] ); std::cout << (i != tensor.get_shape().width() - 1 ? "," : "");
    }
    std::cout << " ]" << std::endl;
}

template <typename T>
void print_tensor_data(const nn::Tensor<T, 2>& tensor) {
    std::cout << "[ ";
    for (nn::Dim i = 0; i < tensor.get_shape().height(); ++i) {
        nn::Idx i_idx = tensor.get_shape().height_index(i);
        std::cout << "[ ";
        for (nn::Dim j = 0; j < tensor.get_shape().width(); ++j) {
            print_value( tensor[tensor.get_shape().width_index(i_idx, j)] ); std::cout << (j != tensor.get_shape().width() - 1 ? "," : "");
        }
        std::cout << " ]" << (i != tensor.get_shape().height() - 1 ? ",\n  " : " ]\n");
    }
}

template <typename T>
void print_tensor_data(const nn::Tensor<T, 3>& tensor) {
    for (nn::Dim i = 0; i < tensor.get_shape().depth(); ++i) {
        std::cout << "( depth: " << i << " )" << std::endl;
        nn::Idx i_idx = tensor.get_shape().depth_index(i);
        std::cout << "[ ";
        for (nn::Dim j = 0; j < tensor.get_shape().height(); ++j) {
            nn::Idx j_idx = tensor.get_shape().height_index(i_idx, j);
            std::cout << "[ ";
            for (nn::Dim k = 0; k < tensor.get_shape().width(); ++k) {
                print_value( tensor[tensor.get_shape().width_index(j_idx, k)] ); std::cout << (k != tensor.get_shape().width() - 1 ? "," : "");
            }
            std::cout << " ]" << (j != tensor.get_shape().height() - 1 ? ",\n  " : " ]\n");
        }
    }
}

template <typename T>
void print_tensor_data(const nn::Tensor<T, 4>& tensor) {
    for (nn::Dim i = 0; i < tensor.get_shape().channels(); ++i) {
        nn::Idx i_idx = tensor.get_shape().channel_index(i);
        for (nn::Dim j = 0; j < tensor.get_shape().depth(); ++j) {
            std::cout << "( channel: " << i << " depth: " << j << " )" << std::endl;
            nn::Idx j_idx = tensor.get_shape().depth_index(i_idx, j);
            std::cout << "[ ";
            for (nn::Dim k = 0; k < tensor.get_shape().height(); ++k) {
                nn::Idx k_idx = tensor.get_shape().height_index(j_idx, k);
                std::cout << "[ ";
                for (nn::Dim l = 0; l < tensor.get_shape().width(); ++l) {
                    print_value( tensor[tensor.get_shape().width_index(k_idx, l)] ); std::cout << (l != tensor.get_shape().width() - 1 ? "," : "");
                }
                std::cout << " ]" << (k != tensor.get_shape().height() - 1 ? ",\n  " : " ]\n");
            }
        }
    }
}

template <typename T>
void print_tensor_data(const nn::Tensor<T, 5>& tensor) {
    for (nn::Dim i = 0; i < tensor.get_shape().batches(); ++i) {
        nn::Idx i_idx = tensor.get_shape().batch_index(i);
        for (nn::Dim j = 0; j < tensor.get_shape().channels(); ++j) {
            nn::Idx j_idx = tensor.get_shape().channel_index(i_idx, j);
            for (nn::Dim k = 0; k < tensor.get_shape().depth(); ++k) {
                std::cout << "( batch: " << i << " channel: " << j << " depth: " << k << " )" << std::endl;
                nn::Idx k_idx = tensor.get_shape().depth_index(j_idx, k);
                std::cout << "[ ";
                for (nn::Dim l = 0; l < tensor.get_shape().height(); ++l) {
                    nn::Idx l_idx = tensor.get_shape().height_index(k_idx, l);
                    std::cout << "[ ";
                    for (nn::Dim m = 0; m < tensor.get_shape().width(); ++m) {
                        print_value( tensor[tensor.get_shape().width_index(l_idx, m)] ); std::cout << (m != tensor.get_shape().width() - 1 ? "," : "");
                    }
                    std::cout << " ]" << (l != tensor.get_shape().height() - 1 ? ",\n  " : " ]\n");
                }
            }
        }
    }
}

template <typename T, nn::u8 N>
void print_tensor(const nn::Tensor<T, N>& tensor)
{
    std::cout << "====================" << std::endl;
    print_tensor_info( tensor );
    std::cout << "--------------------" << std::endl;
    print_tensor_data( tensor );
    std::cout << "====================" << std::endl;
}

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_TESTS_CORE_PRINT_HPP_
