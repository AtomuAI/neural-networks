// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main()
{
    std::vector<std::string> types =
    {
        //"int8_t",
        //"int16_t",
        //"int32_t",
        //"int64_t",
        "float"
        //"double",
        //"std::complex<int8_t>",
        //"std::complex<int16_t>",
        //"std::complex<int32_t>",
        //"std::complex<int64_t>",
        //"std::complex<float>",
        //"std::complex<double>"
    };

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/softmax_layer/softmax_layer.cpp");
    if (!file)
    {
        std::cerr << "Unable to create file" << std::endl;
        return 1;
    }

    // Write the common parts of the file
    file <<
R"(// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>
#include <functional>

//: Library Headers
#include <bewusstsein_util.hpp>

//: Project Headers
#include "c++/bewusstsein_neural_networks/source/core/counter/counter.hpp"
#include "c++/bewusstsein_neural_networks/source/core/step_size/step_size.hpp"
#include "c++/bewusstsein_neural_networks/source/core/beta/beta.hpp"
#include "c++/bewusstsein_neural_networks/source/core/epsilon/epsilon.hpp"
#include "c++/bewusstsein_neural_networks/source/core/training_mode/training_mode.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/softmax_layer/softmax_layer.hpp"

namespace nn
{
    //: Constructors
        /*
        SoftmaxLayer::SoftmaxLayer() :
            BaseLayer( LayerType::softmax_layer, "" ) {}
        */

        SoftmaxLayer::SoftmaxLayer() : BaseLayer( LayerType::softmax_layer )
        {}

    //: Destructors
        SoftmaxLayer::~SoftmaxLayer() {}

    //: Methods
        void SoftmaxLayer::save_model( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_model_to_file( file );
            file.close();
        }

        void SoftmaxLayer::save_state( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_state_to_file( file );
            file.close();
        }

        void SoftmaxLayer::load_model( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file( file );
            file.close();
        }

        void SoftmaxLayer::load_state( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_state_from_file( file );
            file.close();
        }

        void SoftmaxLayer::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );
        }

        void SoftmaxLayer::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );
        }

        void SoftmaxLayer::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );
        }

        void SoftmaxLayer::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );
        }

        template <typename U, typename V>
        void SoftmaxLayer::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            util::Shape input_shape = input_layer.get_shape();
            util::Size spacial_size = std::accumulate( input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>() );

            for ( util::Dim i = 0; i < input_shape.batches(); ++i )
            {
                U sum = 0;
                util::Index batch_index = i * spacial_size;
                for ( util::Index j = 0; j < spacial_size; ++j )
                {
                    util::Index index = batch_index + j;
                    U value = exp( input_layer[ index ] );
                    output_layer[ index ] = value;
                    sum += value;
                }
                for ( util::Index j = 0; j < spacial_size; ++j )
                {
                    util::Index index = batch_index + j;
                    if ( ( sum != INFINITY ) && ( sum != 0 ) && ( sum != -INFINITY ) )
                    {
                        output_layer[ index ] /= sum;
                    }
                    else if ( sum == INFINITY )
                    {
                        output_layer[ index ] = 0;
                    }
                    else
                    {
                        output_layer[ index ] = std::numeric_limits<U>::max();
                    }
                }
            }
        }

        template <typename U, typename V>
        void SoftmaxLayer::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const
        {
            util::Shape input_shape = input_layer.get_shape();
            util::Size spacial_size = std::accumulate( input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>() );

            for ( util::Dim i = 0; i < input_shape.batches(); ++i )
            {
                U sum = 0;
                util::Index batch_index = i * spacial_size;
                for ( util::Index j = 0; j < spacial_size; ++j )
                {
                    util::Index index = batch_index + j;
                    U delta = ( output_layer.get_delta( index ) > 0 ) ? 1 : 0;
                    input_layer.get_delta( index ) = output_layer[ index ] * ( delta - input_layer[ index ] );
                }
            }
        }
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& u : types)
    {
        for (const auto& v : types)
        {
            std::string code =
R"(
    template void SoftmaxLayer::inference( const NodeLayer<{u}>& input_layer, NodeLayer<{v}>& output_layer ) const;)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);
            while ((pos = code.find("{v}")) != std::string::npos)
                code.replace(pos, 3, v);

            file << code;
        }
    }

        file <<
R"(
)";

    for (const auto& u : types)
    {
        for (const auto& v : types)
        {
            std::string code =
R"(
    template void SoftmaxLayer::backpropagation( NodeLayer<{u}>& input_layer, const NodeLayer<{v}>& output_layer ) const;)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);
            while ((pos = code.find("{v}")) != std::string::npos)
                code.replace(pos, 3, v);

            file << code;
        }
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}