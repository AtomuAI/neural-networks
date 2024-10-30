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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/dropout_layer/dropout_layer.cpp");
    if (!file)
    {
        std::cerr << "Unable to create file" << std::endl;
        return 1;
    }

    // Write the common parts of the file
    file <<
R"(// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdlib>

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
#include "c++/bewusstsein_neural_networks/source/layers/dropout_layer/dropout_layer.hpp"

namespace nn
{
    //: Constructors
        /*
        DropoutLayer::DropoutLayer() :
            BaseLayer( LayerType::dropout_layer, "" ), rate( 0 ) {}
        */

        DropoutLayer::DropoutLayer( const util::Shape& shape, const double dropout_rate ):
            BaseLayer( LayerType::dropout_layer ), mask( util::Shape( shape, 4 ), 1 ), rate( dropout_rate )
        {
            if( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Dropoutlayer can have at least 1 and at most 4 dimensions" );
            }
        }


    //: Destructors
        DropoutLayer::~DropoutLayer() {}

    //: Methods
        util::Shape DropoutLayer::get_shape() const
        {
            return this->mask.get_shape();
        }

        int DropoutLayer::get_size() const
        {
            return this->mask.get_size();
        }

        const util::Tensor<uint8_t>& DropoutLayer::get_mask() const
        {
            return this->mask;
        }

        double DropoutLayer::get_rate() const
        {
            return this->rate;
        }

        void DropoutLayer::info() const
        {
            std::cout << "DropoutLayer.Info:" << std::endl << "{" << std::endl;
            std::cout << "Mask:" << std::endl;
            this->mask.info();
            std::cout << "}" << std::endl;
        }

        void DropoutLayer::print() const
        {
            std::cout << "DropoutLayer.Print:" << std::endl << "{" << std::endl;
            std::cout << "Mask:" << std::endl;
            this->mask.print();
            std::cout << "}" << std::endl;
        }

        void DropoutLayer::reshape( const util::Shape& shape )
        {
            this->mask.reshape( shape );
        }

        void DropoutLayer::resize( const util::Shape& shape )
        {
            this->mask.resize( shape );
        }

        void DropoutLayer::set_dropout( double rate )
        {
            this->rate = rate;
        }

        void DropoutLayer::set_dropout( util::Index index, uint8_t value )
        {
            this->mask[ index ] = ( value ) ? 1 : 0;
        }

        void DropoutLayer::set_dropout( std::vector<uint8_t>& data )
        {
            if ( data.size() != mask.get_size() )
            {
                throw std::invalid_argument( "Init data must be the same size as the mask" );
            }

            mask.set(data);
        }

        void DropoutLayer::save_model( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_model_to_file( file );
            file.close();
        }

        void DropoutLayer::save_state( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_state_to_file( file );
            file.close();
        }

        void DropoutLayer::load_model( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file( file );
            file.close();
        }

        void DropoutLayer::load_state( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_state_from_file( file );
            file.close();
        }

        void DropoutLayer::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Rate
            file.write( reinterpret_cast<const char*>( &this->rate ), sizeof( double ) );

            // Save Dropout Tensor
            this->mask.save( file );
        }

        void DropoutLayer::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Rate
            file.write( reinterpret_cast<const char*>( &this->rate ), sizeof( double ) );

            // Save Dropout Tensor
            this->mask.save( file );
        }

        void DropoutLayer::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Rate
            file.read( reinterpret_cast<char*>( &this->rate ), sizeof( double ) );

            // Load Dropout Tensor
            this->mask.load( file );
        }

        void DropoutLayer::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Rate
            file.read( reinterpret_cast<char*>( &this->rate ), sizeof( double ) );

            // Load Dropout Tensor
            this->mask.load( file );
        }

        template <typename U>
        void DropoutLayer::inference( NodeLayer<U>& layer )
        {
            srand( clock() );

            util::Size mask_size = mask.get_size();

            unsigned int seed = time( 0 );
            for ( int index = 0; index < mask_size; index++ )
            {
                mask[ index ] = ( ( (rand_r( &seed ) / static_cast<float>( RAND_MAX ) ) > this->rate ) ? true : false );
            }

            util::Shape layer_shape = layer.get_shape();
            util::Size spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 1, 1, std::multiplies<int>() );

            for ( util::Dim i = 0; i < layer_shape.batches(); i++ )
            {
                util::Index batch_index = i * spacial_size;
                for ( util::Index j = 0; j < spacial_size; j++ )
                {
                    util::Index index = batch_index + j;

                    if ( !mask[ j ] )
                    {
                        layer[ index ] = 0;
                    }
                }
            }
        }

        template <typename U>
        void DropoutLayer::backpropagation( NodeLayer<U>& layer ) const
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 1, 1, std::multiplies<int>() );

            for ( util::Dim i = 0; i < layer_shape.batches(); i++ )
            {
                util::Index batch_index = i * spacial_size;
                for ( util::Index j = 0; j < spacial_size; j++ )
                {
                    util::Index index = batch_index + j;

                    if ( !mask[ j ] )
                    {
                        layer.get_delta( index ) = 0;
                    }
                }
            }
        }
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& u : types)
    {
        std::string code =
R"(
    template void DropoutLayer::inference( NodeLayer<{u}>& layer );)";
        // Replace placeholders in the code with actual types
        size_t pos;
        while ((pos = code.find("{u}")) != std::string::npos)
            code.replace(pos, 3, u);

        file << code;
    }

    file <<
R"(
)";

    for (const auto& u : types)
    {
        std::string code =
R"(
    template void DropoutLayer::backpropagation( NodeLayer<{u}>& layer ) const;)";
        // Replace placeholders in the code with actual types
        size_t pos;
        while ((pos = code.find("{u}")) != std::string::npos)
            code.replace(pos, 3, u);

        file << code;
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}