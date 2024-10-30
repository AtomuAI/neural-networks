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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer_specializations.hpp");
    if (!file)
    {
        std::cerr << "Unable to create file" << std::endl;
        return 1;
    }

    // Write the common parts of the file
    file <<
R"(// Copyright 2024 Shane W. Mulcahy

#ifndef CPP_BEWUSSTSEIN_NEURAL_NETWORKS_SOURCE_LAYERS_NODE_LAYER_NODE_LAYER_SPECIALIZATIONS_HPP_
#define CPP_BEWUSSTSEIN_NEURAL_NETWORKS_SOURCE_LAYERS_NODE_LAYER_NODE_LAYER_SPECIALIZATIONS_HPP_

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>

//: Library Headers
#include <bewusstsein_util.hpp>

//: Project Headers
#include "c++/bewusstsein_neural_networks/source/core/counter/counter.hpp"
#include "c++/bewusstsein_neural_networks/source/core/step_size/step_size.hpp"
#include "c++/bewusstsein_neural_networks/source/core/beta/beta.hpp"
#include "c++/bewusstsein_neural_networks/source/core/epsilon/epsilon.hpp"
#include "c++/bewusstsein_neural_networks/source/core/training_mode/training_mode.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        NodeLayer<T>::NodeLayer() : BaseLayer( LayerType::node_layer, "" ) {}

        template <typename T>
        NodeLayer<T>::NodeLayer( const std::string& name, const util::Shape& shape ) :
            BaseLayer( LayerType::node_layer, name ), nodes( util::Shape( shape, 5 ) )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw ( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }
        }

        template <typename T>
        NodeLayer<T>::NodeLayer( const std::string& name, const util::Shape& shape, const T scalar ) :
            BaseLayer( LayerType::node_layer, name ), nodes( util::Shape( shape, 5 ), scalar )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }
        }

        template <typename T>
        NodeLayer<T>::NodeLayer( const std::string& name, const util::Shape& shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::node_layer, name ), nodes( util::Shape( shape, 5 ), data )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }
        }

        template <typename T>
        NodeLayer<T>::NodeLayer( const std::string& name, const util::Shape& shape, TrainingMode training_mode ) :
            BaseLayer( LayerType::node_layer, name ), nodes( util::Shape( shape, 5 ) )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }
        */

        template <typename T>
        NodeLayer<T>::NodeLayer( const util::Shape& shape, const T scalar, TrainingMode training_mode ) :
            BaseLayer( LayerType::node_layer ), nodes( util::Shape( shape, 5 ), scalar )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        NodeLayer<T>::NodeLayer( const util::Shape& shape, const std::vector<T>& data, TrainingMode training_mode ) :
            BaseLayer( LayerType::node_layer ), nodes( util::Shape( shape, 5 ), data )
        {
            if ( ( shape.size() > 5 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Nodelayer can have at least 1 and at most 5 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }


    //: Destructors
        template <typename T>
        NodeLayer<T>::~NodeLayer() {}


    //: Operators
        template <typename T>
        T NodeLayer<T>::operator[]( const util::Indices& indices ) const
        {
            return nodes[ indices ];
        }

        template <typename T>
        T NodeLayer<T>::operator[]( const util::Index index ) const
        {
            return nodes[ index ];
        }

        template <typename T>
        T& NodeLayer<T>::operator[]( const util::Indices& indices )
        {
            return nodes[ indices ];
        }

        template <typename T>
        T& NodeLayer<T>::operator[]( const util::Index index )
        {
            return nodes[ index ];
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator+( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + other.nodes;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator-( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - other.nodes;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator*( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * other.nodes;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator/( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / other.nodes;
            return result;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator+=( const NodeLayer<T>& other )
        {
            this->nodes += other.nodes;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator-=( const NodeLayer<T>& other )
        {
            this->nodes -= other.nodes;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator*=( const NodeLayer<T>& other )
        {
            this->nodes *= other.nodes;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator/=( const NodeLayer<T>& other )
        {
            this->nodes /= other.nodes;
            return *this;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator+( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + scalar;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator-( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - scalar;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator*( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * scalar;
            return result;
        }

        template <typename T>
        NodeLayer<T> NodeLayer<T>::operator/( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / scalar;
            return result;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator+=( const T scalar )
        {
            this->nodes += scalar;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator-=( const T scalar )
        {
            this->nodes -= scalar;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator*=( const T scalar )
        {
            this->nodes *= scalar;
            return *this;
        }

        template <typename T>
        NodeLayer<T>& NodeLayer<T>::operator/=( const T scalar )
        {
            this->nodes /= scalar;
            return *this;
        }

    //: Methods
        template <typename T>
        void NodeLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape shape = this->nodes.get_shape();
                    this->delta.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape shape = this->nodes.get_shape();
                    this->delta.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape shape = this->nodes.get_shape();
                    this->delta.resize( shape );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Invalid training mode selection" );
                }
            }
        }

        template <typename T>
        util::Shape NodeLayer<T>::get_shape() const
        {
            return nodes.get_shape();
        }

        template <typename T>
        util::Size NodeLayer<T>::get_size() const
        {
            return nodes.get_size();
        }

        template <typename T>
        const util::Tensor<T>& NodeLayer<T>::get_nodes() const
        {
            return this->nodes;
        }

        template <typename T>
        const util::Tensor<T>& NodeLayer<T>::get_delta() const
        {
            return this->delta;
        }

        template <typename T>
        void NodeLayer<T>::info() const
        {
            std::cout << "NodeLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Nodes:" << std::endl;
            nodes.info();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void NodeLayer<T>::print() const
        {
            std::cout << "NodeLayer<T>.Print:" << std::endl << "{" << std::endl;
            std::cout << "Nodes:" << std::endl;
            nodes.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void NodeLayer<T>::print_delta() const
        {
            std::cout << "NodeLayer<T>.PrintDelta:" << std::endl << "{" << std::endl;
            std::cout << "Delta:" << std::endl;
            delta.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void NodeLayer<T>::reshape( const util::Shape& new_shape )
        {
            nodes.reshape( new_shape );
        }

        template <typename T>
        void NodeLayer<T>::resize( const util::Shape& new_shape )
        {
            nodes.resize( new_shape );
        }

        template <typename T>
        void NodeLayer<T>::fill( const T value )
        {
            nodes.fill( value );
        }

        template <typename T>
        void NodeLayer<T>::zero()
        {
            nodes.zero();
        }

        template <typename T>
        void NodeLayer<T>::randomize( const T min, const T max )
        {
            nodes.randomize( min, max );
        }

        template <typename T>
        T NodeLayer<T>::get_delta( const util::Indices& indices ) const
        {
            return delta[ indices ];
        }

        template <typename T>
        T NodeLayer<T>::get_delta( const util::Dim index ) const
        {
            return delta[ index ];
        }

        template <typename T>
        T& NodeLayer<T>::get_delta( const util::Indices& indices )
        {
            return delta[ indices ];
        }

        template <typename T>
        T& NodeLayer<T>::get_delta( const util::Dim index )
        {
            return delta[ index ];
        }

        template <typename T>
        void NodeLayer<T>::fill_delta( const T value )
        {
            delta.fill( value );
        }

        template <typename T>
        void NodeLayer<T>::zero_delta()
        {
            delta.zero();
        }

        template <typename T>
        void NodeLayer<T>::randomize_delta( const T min, const T max )
        {
            delta.randomize( min, max );
        }

        template <typename T>
        void NodeLayer<T>::load_images( const std::vector<std::string>& filenames, const util::ImageType type )
        {
            nodes.load_images( filenames, type );
        }

        template <typename T>
        void NodeLayer<T>::save_images( const std::vector<std::string>& filenames ) const
        {
            nodes.save_images( filenames );
        }

        template <typename T>
        void NodeLayer<T>::save_model( const std::string file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_model_to_file( file );
            file.close();
        }

        template <typename T>
        void NodeLayer<T>::save_state( const std::string file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_state_to_file( file );
            file.close();
        }

        template <typename T>
        void NodeLayer<T>::load_model( const std::string file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file( file );
            file.close();
        }

        template <typename T>
        void NodeLayer<T>::load_state( const std::string file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_state_from_file( file );
            file.close();
        }

        template <typename T>
        void NodeLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Nodes Tensor
            this->nodes.save( file );
        }

        template <typename T>
        void NodeLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Nodes Tensor
            this->nodes.save( file );

            // Save Delta Tensor
            this->delta.save( file );
        }

        template <typename T>
        void NodeLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Nodes Tensor
            this->nodes.load( file );
        }

        template <typename T>
        void NodeLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Nodes Tensor
            this->nodes.load( file );

            // Load Delta Tensor
            this->delta.load( file );
        }
        /*
        template <typename T>
        void NodeLayer<T>::create_window( const uint width, const uint height ) const 
        {
            cv::namedWindow( this->name, cv::WINDOW_NORMAL );
            cv::resizeWindow( this->name, width, height );
        }

        template <typename T>
        void NodeLayer<T>::destroy_window() const 
        {
            cv::destroyWindow( this->name );
        }

        template <typename T>
        void NodeLayer<T>::show( const uint batch_index ) const 
        {
            nodes.show( this->name, batch_index );
        }

        template <typename T>
        void NodeLayer<T>::show_batch() const 
        {
            nodes.show_batch( this->name );
        }

        template <typename T>
        void NodeLayer<T>::show_video( const uint batch_index, const uint channel ) const 
        {
            nodes.show_video( this->name, batch_index, channel );
        }

        template <typename T>
        void NodeLayer<T>::show_batch_video() const 
        {
            nodes.show_batch_video( this->name );
        }

        template <typename T>
        void NodeLayer<T>::record_frame( cv::VideoWriter& video, const uint batch_index, const uint channel ) const 
        {
            nodes.record_frame( video, batch_index, channel );
        }
        */
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& t : types)
    {
        std::string code =
R"(
    template class NodeLayer<{t}>;)";
        // Replace placeholders in the code with actual types
        size_t pos;
        while ((pos = code.find("{t}")) != std::string::npos)
            code.replace(pos, 3, t);

        file << code;
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}