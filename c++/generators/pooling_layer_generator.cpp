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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/pooling_layer/pooling_layer.cpp");
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
#include "c++/bewusstsein_neural_networks/source/layers/pooling_layer/pooling_layer.hpp"

namespace nn
{
    //: Constructors
        /*
        PoolingLayer::PoolingLayer() :
            BaseLayer( LayerType::pooling_layer, "" ), type( PoolingType::max ) {}
        */

        PoolingLayer::PoolingLayer( const PoolingType type, const util::Shape& pool, const util::Shape& stride, const util::Shape& dilation ) :
            BaseLayer( LayerType::pooling_layer ), type( type ), pool( pool ), stride( stride ), dilation( dilation )
        {}

    //: Destructors
        PoolingLayer::~PoolingLayer() {}

    //: Methods
        PoolingType PoolingLayer::get_pooling_type() const
        {
            return this->type;
        }

        const util::Shape& PoolingLayer::get_shape() const
        {
            return this->pool;
        }

        const util::Shape& PoolingLayer::get_stride() const
        {
            return this->stride;
        }

        const util::Shape& PoolingLayer::get_dilation() const
        {
            return this->dilation;
        }

        void PoolingLayer::info() const
        {
            util::Size shape_size = this->pool.size();
            util::Size stride_size = this->stride.size();
            util::Size dilation_size = this->dilation.size();

            std::cout << "PoolingLayer.Info:" << std::endl << "{" << std::endl;

            std::cout << "Pool: [";
            this->pool.print();

            std::cout << "Stride: [";
            this->stride.print();

            std::cout << "Dilation: [";
            this->dilation.print();

            std::cout << "Pooling Type: ";
            switch( this->type )
            {
                case PoolingType::max:
                {
                    std::cout << "max" << std::endl;
                    break;
                }

                case PoolingType::average:
                {
                    std::cout << "average" << std::endl;
                    break;
                }

                default:
                {
                    throw std::invalid_argument( "Pooling Type is invalid/uninitialized" );
                    break;
                }
            }

            std::cout << "}" << std::endl;
        }

        void PoolingLayer::save_model( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_model_to_file( file );
            file.close();
        }

        void PoolingLayer::save_state( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_state_to_file( file );
            file.close();
        }

        void PoolingLayer::load_model( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file( file );
            file.close();
        }

        void PoolingLayer::load_state( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_state_from_file( file );
            file.close();
        }

        void PoolingLayer::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Pooling Type
            file.write( reinterpret_cast<const char*>( &this->type ), sizeof( PoolingType ) );

            // Save Shape
            this->pool.save( file );

            // Save Stride
            this->stride.save( file );

            // Save Dilation
            this->dilation.save( file );
        }

        void PoolingLayer::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Pooling Type
            file.write( reinterpret_cast<const char*>( &this->type ), sizeof( PoolingType ) );

            // Save Shape
            this->pool.save( file );

            // Save Stride
            this->stride.save( file );

            // Save Dilation
            this->dilation.save( file );
        }

        void PoolingLayer::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Padding Type
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( PoolingType ) );

            // Load Shape
            this->pool.load( file );

            // Load Stride
            this->stride.load( file );

            // Load Dilation
            this->dilation.load( file );
        }

        void PoolingLayer::load_state_from_file( std::ifstream& file)
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Padding Type
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( PoolingType ) );

            // Load Shape
            this->pool.load( file );

            // Load Stride
            this->stride.load( file );

            // Load Dilation
            this->dilation.load( file );
        }

        template <typename U, typename V>
        void PoolingLayer::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            switch( this->type )
            {
                case PoolingType::max:
                {
                    inference_maxpooled( input_layer, output_layer );
                    break;
                }

                case PoolingType::average:
                {
                    inference_avgpooled( input_layer, output_layer );
                    break;
                }

                default:
                {
                    throw std::invalid_argument( "Pooling Type is invalid/uninitialized" );
                    break;
                }
            }
        }

        template <typename U, typename V>
        void PoolingLayer::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const
        {
            switch( this->type )
            {
                case PoolingType::max:
                {
                    backpropagation_maxpooled( input_layer, output_layer );
                    break;
                }

                case PoolingType::average:
                {
                    backpropagation_avgpooled( input_layer, output_layer );
                    break;
                }

                default:
                {
                    throw std::invalid_argument( "Pooling Type is invalid/uninitialized" );
                    break;
                }
            }
        }

        template <typename U, typename V>
        void PoolingLayer::inference_maxpooled( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape output_shape = output_layer.get_shape();
            const util::Shape input_shape = input_layer.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );

            const util::Dim3 pool_size( this->pool.width(), this->pool.height(), this->pool.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx       = out_b_dim * output_size.c;
                util::Index in_b_idx        = out_b_dim * input_size.c;
                for ( util::Dim out_c_dim   = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_n_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx       = ( out_b_idx + out_c_dim ) * output_size.z;
                    for ( util::Dim out_z_dim   = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound          = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx       = ( out_c_idx + out_z_dim ) * output_size.y;
                        for ( util::Dim out_y_dim   = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound          = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx       = ( out_z_idx + out_y_dim ) * output_size.z;
                            for ( util::Dim out_x_dim   = 0; out_x_dim < output_size.z; out_x_dim++ )
                            {
                                //bool out_x_bound  = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.z );
                                int out_index       = out_y_idx + out_x_dim;

                                float max_value = std::numeric_limits<float>::lowest();

                                util::Dim in_b_dim          = out_b_dim;
                                util::Dim in_c_dim          = out_c_dim % input_size.c;
                                bool in_c_bound             = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx        = ( in_b_idx + in_c_dim ) * input_size.z;
                                for ( util::Dim pool_z_dim  = 0; pool_z_dim < pool_size.z; pool_z_dim++ )
                                {
                                    util::Index in_z_dim        = out_z_dim * stride_size.z + pool_z_dim * dilation_size.z;
                                    bool in_z_bound             = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx        = ( in_c_idx + in_z_dim ) * input_size.y;
                                    for ( util::Dim pool_y_dim  = 0; pool_y_dim < pool_size.y; pool_y_dim++ )
                                    {
                                        util::Index in_y_dim        = out_y_dim * stride_size.y + pool_y_dim * dilation_size.y;
                                        bool in_y_bound             = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx        = ( in_z_idx + in_y_dim ) * input_size.x;
                                        for ( util::Dim pool_x_dim  = 0; pool_x_dim < pool_size.x; pool_x_dim++ )
                                        {
                                            util::Index in_x_dim    = out_x_dim * stride_size.x + pool_x_dim * dilation_size.x;
                                            bool in_bound           = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index    = in_y_idx + in_x_dim;

                                            float current_value = input_layer[ in_index ];
                                            if ( current_value > max_value )
                                            {
                                                max_value = current_value;
                                            }
                                        }
                                    }
                                }
                                output_layer[ out_index ] = max_value;
                            }
                        }
                    }
                }
            }
        }

        template <typename U, typename V>
        void PoolingLayer::inference_avgpooled( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape output_shape = output_layer.get_shape();
            const util::Shape input_shape = input_layer.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );

            const util::Dim3 pool_size( this->pool.width(), this->pool.height(), this->pool.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            float pooling_scalar = 1.0f / ( pool_size.x * pool_size.y * pool_size.z );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx       = out_b_dim * output_size.c;
                util::Index in_b_idx        = out_b_dim * input_size.c;
                for ( util::Dim out_c_dim   = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_n_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx       = ( out_b_idx + out_c_dim ) * output_size.z;
                    for ( util::Dim out_z_dim   = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound          = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx       = ( out_c_idx + out_z_dim ) * output_size.y;
                        for ( util::Dim out_y_dim   = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound          = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx       = ( out_z_idx + out_y_dim ) * output_size.z;
                            for ( util::Dim out_x_dim   = 0; out_x_dim < output_size.z; out_x_dim++ )
                            {
                                //bool out_x_bound      = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.z );
                                util::Index out_index   = out_y_idx + out_x_dim;

                                float sum_value = 0.0f;

                                util::Dim in_b_dim          = out_b_dim;
                                util::Dim in_c_dim          = out_c_dim % input_size.c;
                                bool in_c_bound             = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx        = ( in_b_idx + in_c_dim ) * input_size.z;
                                for ( util::Dim pool_z_dim  = 0; pool_z_dim < pool_size.z; pool_z_dim++ )
                                {
                                    util::Index in_z_dim        = out_z_dim * stride_size.z + pool_z_dim * dilation_size.z;
                                    bool in_z_bound             = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx        = ( in_c_idx + in_z_dim ) * input_size.y;
                                    for ( util::Dim pool_y_dim  = 0; pool_y_dim < pool_size.y; pool_y_dim++ )
                                    {
                                        util::Index in_y_dim        = out_y_dim * stride_size.y + pool_y_dim * dilation_size.y;
                                        bool in_y_bound             = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx        = ( in_z_idx + in_y_dim ) * input_size.x;
                                        for ( util::Dim pool_x_dim  = 0; pool_x_dim < pool_size.x; pool_x_dim++ )
                                        {
                                            util::Index in_x_dim    = out_x_dim * stride_size.x + pool_x_dim * dilation_size.x;
                                            bool in_bound           = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index    = in_y_idx + in_x_dim;

                                            float current_value     = input_layer[ in_index ];
                                            sum_value               += current_value;
                                        }
                                    }
                                }
                                output_layer[ out_index ] = sum_value * pooling_scalar;
                            }
                        }
                    }
                }
            }
        }

        template <typename U, typename V>
        void PoolingLayer::backpropagation_maxpooled( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const
        {
            const util::Shape output_shape = output_layer.get_shape();
            const util::Shape input_shape = input_layer.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );

            const util::Dim3 pool_size( this->pool.width(), this->pool.height(), this->pool.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx       = out_b_dim * output_size.c;
                util::Index in_b_idx        = out_b_dim * input_size.c;
                for ( util::Dim out_c_dim   = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_n_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx       = ( out_b_idx + out_c_dim ) * output_size.z;
                    for ( util::Dim out_z_dim   = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound          = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx       = ( out_c_idx + out_z_dim ) * output_size.y;
                        for ( util::Dim out_y_dim   = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound          = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx       = ( out_z_idx + out_y_dim ) * output_size.z;
                            for ( util::Dim out_x_dim   = 0; out_x_dim < output_size.z; out_x_dim++ )
                            {
                                //bool out_x_bound          = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.z );
                                util::Index out_index       = out_y_idx + out_x_dim;

                                util::Index max_index       = 0;
                                float max_value             = input_layer[max_index];

                                util::Dim in_b_dim          = out_b_dim;
                                util::Dim in_c_dim          = out_c_dim % input_size.c;
                                bool in_c_bound             = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx        = ( in_b_idx + in_c_dim ) * input_size.z;
                                for ( util::Dim pool_z_dim  = 0; pool_z_dim < pool_size.z; pool_z_dim++ )
                                {
                                    util::Index in_z_dim        = out_z_dim * stride_size.z + pool_z_dim * dilation_size.z;
                                    bool in_z_bound             = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx        = ( in_c_idx + in_z_dim ) * input_size.y;
                                    for ( util::Dim pool_y_dim  = 0; pool_y_dim < pool_size.y; pool_y_dim++ )
                                    {
                                        util::Index in_y_dim        = out_y_dim * stride_size.y + pool_y_dim * dilation_size.y;
                                        bool in_y_bound             = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx        = ( in_z_idx + in_y_dim ) * input_size.x;
                                        for ( util::Dim pool_x_dim  = 0; pool_x_dim < pool_size.x; pool_x_dim++ )
                                        {
                                            util::Index in_x_dim    = out_x_dim * stride_size.x + pool_x_dim * dilation_size.x;
                                            bool in_bound           = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index    = in_y_idx + in_x_dim;

                                            float current_value = input_layer[ in_index ];

                                            if ( current_value > max_value )
                                            {
                                                max_index                           = in_index;
                                                max_value                           = current_value;
                                                input_layer.get_delta( in_index )   = 0;
                                            }
                                        }
                                    }
                                }
                                float out_delta = output_layer.get_delta( out_index );
                                input_layer.get_delta( max_index ) = out_delta;
                            }
                        }
                    }
                }
            }
        }

        template <typename U, typename V>
        void PoolingLayer::backpropagation_avgpooled( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const
        {
            const util::Shape output_shape = output_layer.get_shape();
            const util::Shape input_shape = input_layer.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );

            const util::Dim3 pool_size ( this->pool.width(), this->pool.height(), this->pool.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            float pooling_scalar = 1.0f / ( pool_size.x * pool_size.y * pool_size.z );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ ) // Output Batch Loop
            {
                //bool out_c_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx       = out_b_dim * output_size.c;
                util::Index in_b_idx        = out_b_dim * input_size.c;
                for ( util::Dim out_c_dim   = 0; out_c_dim < output_size.c; out_c_dim++ ) // Output Channel Loop
                {
                    //bool out_n_bound          = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx       = ( out_b_idx + out_c_dim ) * output_size.z;
                    for ( util::Dim out_z_dim   = 0; out_z_dim < output_size.z; out_z_dim++ ) // Output Z-Dim Loop
                    {
                        //bool out_z_bound          = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx       = ( out_c_idx + out_z_dim ) * output_size.y;
                        for ( util::Dim out_y_dim   = 0; out_y_dim < output_size.y; out_y_dim++ ) // Output Y-Dim Loop
                        {
                            //bool out_y_bound          = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx       = ( out_z_idx + out_y_dim ) * output_size.z;
                            for ( util::Dim out_x_dim   = 0; out_x_dim < output_size.z; out_x_dim++ ) // Output X-Dim Loop
                            {
                                //bool out_x_bound      = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.z );
                                util::Index out_index   = out_y_idx + out_x_dim;
                                float out_delta         = output_layer.get_delta( out_index );

                                util::Dim in_b_dim          = out_b_dim;
                                util::Dim in_c_dim          = out_c_dim % input_size.c;
                                bool in_c_bound             = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx        = ( in_b_idx + in_c_dim ) * input_size.z;
                                for ( util::Dim pool_z_dim  = 0; pool_z_dim < pool_size.z; pool_z_dim++ ) // Pool Z-Dim Loop
                                {
                                    util::Index in_z_dim        = out_z_dim * stride_size.z + pool_z_dim * dilation_size.z;
                                    bool in_z_bound             = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx        = (in_c_idx + in_z_dim) * input_size.y;
                                    for ( util::Dim pool_y_dim  = 0; pool_y_dim < pool_size.y; pool_y_dim++ ) // Pool Y-Dim Loop
                                    {
                                        util::Index in_y_dim        = out_y_dim * stride_size.y + pool_y_dim * dilation_size.y;
                                        bool in_y_bound             = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx        = ( in_z_idx + in_y_dim ) * input_size.x;
                                        for ( util::Dim pool_x_dim  = 0; pool_x_dim < pool_size.x; pool_x_dim++ ) // Pool X-Dim Loop
                                        {
                                            util::Index in_x_dim                = out_x_dim * stride_size.x + pool_x_dim * dilation_size.x;
                                            bool in_bound                       = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index                = in_y_idx + in_x_dim;
                                            input_layer.get_delta( in_index )   = out_delta * pooling_scalar;
                                        }
                                    }
                                }
                            }
                        }
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
        for (const auto& v : types)
        {
            std::string code =
R"(
    template void PoolingLayer::inference( const NodeLayer<{u}>& input_layer, NodeLayer<{v}>& output_layer ) const;)";
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
    template void PoolingLayer::backpropagation( NodeLayer<{u}>& input_layer, const NodeLayer<{v}>& output_layer ) const;)";
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