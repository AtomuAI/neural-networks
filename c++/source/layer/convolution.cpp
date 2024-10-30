// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/math.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/core/initialization_type.hpp"
#include "bewusstsein_neural_networks/c++/include/core/distribution_type.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/convolution.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& stride, const Shape<3>& dilation, const T scalar ) requires ( ( C == ConvolutionType::down_sample ) && ( S != PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, scalar ), stride( stride ), dilation( dilation )
        {
            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = Shape<3>( 0, 0, 0 );
                    this->inv_padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }
                case PaddingSize::same:
                {
                    this->padding = Shape<3>( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }
                case PaddingSize::full:
                {
                    this->padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = Shape<3>( 0, 0, 0 );
                    break;
                }
                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& stride, const Shape<3>& dilation, const std::vector<T> data ) requires ( ( C == ConvolutionType::down_sample ) && ( S != PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, data ), stride( stride ), dilation( dilation )
        {
            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = Shape<3>( 0, 0, 0 );
                    this->inv_padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }
                case PaddingSize::same:
                {
                    this->padding = Shape<3>( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }
                case PaddingSize::full:
                {
                    this->padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = Shape<3>( 0, 0, 0 );
                    break;
                }
                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& padding, const Shape<3>& stride, const Shape<3>& dilation, const T scalar ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, scalar ), padding( padding ), stride( stride ), dilation( dilation ) {}

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& padding, const Shape<3>& stride, const Shape<3>& dilation, const std::vector<T> data ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, data ), padding( padding ), stride( stride ), dilation( dilation ) {}

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& input_dilation, const Shape<3>& stride, const Shape<3>& dilation, const T scalar ) requires ( ( C == ConvolutionType::up_sample ) && ( S != PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, scalar ), stride( stride ), dilation( dilation ), input_dilation( input_dilation )
        {
            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = Shape<3>( 0, 0, 0 );
                    this->inv_padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }
                case PaddingSize::same:
                {
                    this->padding = Shape<3>( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }
                case PaddingSize::full:
                {
                    this->padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = Shape<3>( 0, 0, 0 );
                    break;
                }
                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& input_dilation, const Shape<3>& stride, const Shape<3>& dilation, const std::vector<T> data ) requires ( ( C == ConvolutionType::up_sample ) && ( S != PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, data ), stride( stride ), dilation( dilation ), input_dilation( input_dilation )
        {
            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = Shape<3>( 0, 0, 0 );
                    this->inv_padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }
                case PaddingSize::same:
                {
                    this->padding = Shape<3>( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }
                case PaddingSize::full:
                {
                    this->padding = Shape<3>( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = Shape<3>( 0, 0, 0 );
                    break;
                }
                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& input_dilation, const Shape<3>& padding, const Shape<3>& stride, const Shape<3>& dilation, const T scalar ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, scalar ), padding( padding ), stride( stride ), dilation( dilation ), input_dilation( input_dilation ) {}

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::ConvolutionLayer( const Shape<4>& filter_shape, const Shape<3>& input_dilation, const Shape<3>& padding, const Shape<3>& stride, const Shape<3>& dilation, const std::vector<T> data ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::custom ) ) :
            BaseLayer( LayerType::convolution_layer ), filter( filter_shape, data ), padding( padding ), stride( stride ), dilation( dilation ), input_dilation( input_dilation ) {}

    //: Destructors
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        ConvolutionLayer<T, C, P, S>::~ConvolutionLayer() {}

    //: Methods
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Shape<5> ConvolutionLayer<T, C, P, S>::calculate_output_shape( const Shape<5>& input_shape ) const requires ( C == ConvolutionType::down_sample )
        {
            Shape<5> shape
            (
                ( ( input_shape.width() + ( 2 * this->padding.width() ) - ( this->dilation.width() * ( this->filter.get_shape().width() - 1 ) ) - 1 ) / this->stride.width() ) + 1,
                ( ( input_shape.height() + ( 2 * this->padding.height() ) - ( this->dilation.height() * ( this->filter.get_shape().height() - 1 ) ) - 1 ) / this->stride.height() ) + 1,
                ( ( input_shape.depth() + ( 2 * this->padding.depth() ) - ( this->dilation.depth() * ( this->filter.get_shape().depth() - 1 ) ) - 1 ) / this->stride.depth() ) + 1,
                this->filter.get_shape().channels(),
                input_shape.batches()
            );

            return shape;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Shape<5> ConvolutionLayer<T, C, P, S>::calculate_output_shape( const Shape<5>& input_shape ) const requires ( C == ConvolutionType::up_sample )
        {
            Shape<5> shape
            (
                ( ( ( this->input_dilation.width() * input_shape.width() ) + ( 2 * this->padding.width() ) - ( this->dilation.width() * ( this->filter.get_shape().width() - 1 ) ) - 1 ) / this->stride.width() ) + 1,
                ( ( ( this->input_dilation.height() * input_shape.height() ) + ( 2 * this->padding.height() ) - ( this->dilation.height() * ( this->filter.get_shape().height() - 1 ) ) - 1 ) / this->stride.height() ) + 1,
                ( ( ( this->input_dilation.depth() * input_shape.depth() ) + ( 2 * this->padding.depth() ) - ( this->dilation.depth() * ( this->filter.get_shape().depth() - 1 ) ) - 1 ) / this->stride.depth() ) + 1,
                this->filter.get_shape().channels(),
                input_shape.batches()
            );

            return shape;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<4>& ConvolutionLayer<T, C, P, S>::get_shape() const
        {
            return this->filter.get_shape();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Size ConvolutionLayer<T, C, P, S>::get_size() const
        {
            return this->filter.get_size();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::fill_filter( const T value )
        {
            this->filter.fill( value );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::fill_jacobian( const T value )
        {
            this->jacobian.fill( value );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::fill_momentum( const T value )
        {
            this->momentum.fill( value );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::fill_velocity( const T value )
        {
            this->velocity.fill( value );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::zero_filter()
        {
            this->filter.zero();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::zero_jacobian()
        {
            this->jacobian.zero();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::zero_momentum()
        {
            this->momentum.zero();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::zero_velocity()
        {
            this->velocity.zero();
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::randomize_filter( const T min, const T max )
        {
            this->filter.randomize( min, max );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::randomize_jacobian( const T min, const T max )
        {
            this->jacobian.randomize( min, max );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::randomize_momentum( const T min, const T max )
        {
            this->momentum.randomize( min, max );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::randomize_velocity( const T min, const T max )
        {
            this->velocity.randomize( min, max );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<3>& ConvolutionLayer<T, C, P, S>::get_stride() const
        {
            return this->stride;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<3>& ConvolutionLayer<T, C, P, S>::get_padding() const
        {
            return this->padding;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<3>& ConvolutionLayer<T, C, P, S>::get_dilation() const
        {
            return this->dilation;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<3>& ConvolutionLayer<T, C, P, S>::get_input_dilation() const
        {
            return this->input_dilation;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Shape<3>& ConvolutionLayer<T, C, P, S>::get_inverse_padding() const
        {
            return this->inv_padding;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Tensor<T, 4>& ConvolutionLayer<T, C, P, S>::get_filter() const
        {
            return this->filter;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Tensor<T, 4>& ConvolutionLayer<T, C, P, S>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Tensor<T, 4>& ConvolutionLayer<T, C, P, S>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        const Tensor<T, 4>& ConvolutionLayer<T, C, P, S>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::reshape( const Shape<4>& shape )
        {
            this->BaseLayer::reshape( shape, this->filter, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::resize( const Shape<4>& shape )
        {
            this->BaseLayer::resize( shape, this->filter, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::set_training_mode( const TrainingMode training_mode )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( this->filter, this->jacobian, this->momentum, this->velocity );
        }

        /*
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        void ConvolutionLayer<T, C, P, S>::initialize( const Shape<5>& input_layer_shape, const Shape<5>& output_layer_shape, const InitializationType type, const DistributionType distribution )
        {
            Size input_layer_size = input_layer_shape.distance( 0, input_layer_shape.volume() - 1 );
            Size output_layer_size = output_layer_shape.distance( 0, output_layer_shape.volume() - 1 );
            T variance = 0;
            T std_dev = 0;

            switch ( type )
            {
                case InitializationType::xavier_glorot: { variance = 2.0 / ( input_layer_size + output_layer_size ); std_dev = nn::sqrt( variance ); break; }
                case InitializationType::he: { variance = 2.0 / ( input_layer_size ); std_dev = nn::sqrt( variance ); break; }
                case InitializationType::lecun: { variance = 1.0 / ( input_layer_size ); std_dev = nn::sqrt( variance ); break; }
                default: { throw std::invalid_argument( "Initialization Type is invalid/uninitialized" ); break; }
            }

            switch ( distribution )
            {
                case DistributionType::normal: { this->filter.fill_normal_distribution( 0, std_dev ); break; }
                case DistributionType::poisson: { filter.fill_poisson_distribution(); break; }
                case DistributionType::binomial: { filter.fill_binomial_distribution(); break; }
                case DistributionType::exponential: { filter.fill_exponential_distribution(); break; }
                case DistributionType::uniform: { filter.fill_uniform_distribution(); break; }
                case DistributionType::bernoulli: { filter.fill_bernoulli_distribution(); break; }
                case DistributionType::beta: { filter.fill_beta_distribution(); break; }
                case DistributionType::weibull: { filter.fill_weibull_distribution(); break; }
                case DistributionType::gamma: { filter.fill_gamma_distribution(); break; }
                case DistributionType::chi_squared: { filter.fill_chi_squared_distribution(); break; }
                case DistributionType::log_normal: { filter.fill_log_normal_distribution(); break; }
                case DistributionType::f: { filter.fill_f_distribution(); break; }
                case DistributionType::discrete_uniform: { filter.fill_discrete_uniform_distribution(); break; }
                default: { throw std::invalid_argument( "Distribution Type is invalid/uninitialized" ); break; }
            };
        }
        */

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        Error ConvolutionLayer<T, C, P, S>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<5>& input_shape = input_layer.get_shape();

            if ( output_shape != this->calculate_output_shape( input_shape ) ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim out_b_dim = 0; out_b_dim < output_shape.batches(); ++out_b_dim )
            {
                Idx out_b_idx = output_shape.batch_index( out_b_dim );
                Idx in_b_idx = input_shape.batch_index( out_b_dim );

                for ( Dim out_c_dim = 0; out_c_dim < output_shape.channels(); ++out_c_dim )
                {
                    Idx out_c_idx = output_shape.channel_index( out_b_idx, out_c_dim );

                    for ( Dim out_z_dim = 0; out_z_dim < output_shape.depth(); ++out_z_dim )
                    {
                        Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                        for ( Dim out_y_dim = 0; out_y_dim < output_shape.height(); ++out_y_dim )
                        {
                            Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                            for ( Dim out_x_dim = 0; out_x_dim < output_shape.width(); ++out_x_dim )
                            {
                                Idx output_index = output_shape.width_index( out_y_idx, out_x_dim );

                                output_layer.get_node( output_index ) = this->filter_inference<U, V>( input_layer, in_b_idx, out_c_dim, out_z_dim, out_y_dim, out_x_dim );
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        Error ConvolutionLayer<T, C, P, S>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            if ( output_shape != this->calculate_output_shape( input_shape ) ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim out_b_dim = 0; out_b_dim < output_shape.batches(); ++out_b_dim )
            {
                Idx out_b_idx = output_shape.batch_index( out_b_dim );;
                Idx in_b_idx = input_shape.batch_index( out_b_dim );

                for( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
                {
                    Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

                    for( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
                    {
                        Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                        for( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                        {
                            Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                            for( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                            {
                                Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                                this->get_jacobian( filter_index ) += this->filter_backpropagation<U, V>( input_layer, output_layer, in_b_idx, out_b_idx, filter_c_dim, filter_z_dim, filter_y_dim, filter_x_dim );
                            }
                        }
                    }
                }
                for ( Dim in_c_dim = 0; in_c_dim < input_shape.channels(); ++in_c_dim )
                {
                    Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

                    for ( Dim in_z_dim = 0; in_z_dim < input_shape.depth(); ++in_z_dim )
                    {
                        Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                        for ( Dim in_y_dim = 0; in_y_dim < input_shape.height(); ++in_y_dim )
                        {
                            Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                            for ( Dim in_x_dim = 0; in_x_dim < input_shape.width(); ++in_x_dim )
                            {
                                Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                                input_layer.get_delta( in_index ) += this->layer_backpropagation<U, V>( output_layer, out_b_idx, in_c_dim, in_z_dim, in_y_dim, in_x_dim );
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Error ConvolutionLayer<T, C, P, S>::gradient_decent_normal( const Dim batch_size, const StepSize step_size )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->filter, this->jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Error ConvolutionLayer<T, C, P, S>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->filter, this->jacobian, this->momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        Error ConvolutionLayer<T, C, P, S>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->filter, this->jacobian, this->momentum, this->velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }

    //: Inline Methods
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline Dim ConvolutionLayer<T, C, P, S>::stride_pad_dilation_dim( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim pad_dim_size, const Dim dilation_dim_size ) const
        {
            return in_dim * stride_dim_size - pad_dim_size + out_dim * dilation_dim_size;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline Dim ConvolutionLayer<T, C, P, S>::stride_dilation_dim( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim dilation_dim_size ) const
        {
            return in_dim * stride_dim_size + out_dim * dilation_dim_size;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline Dim ConvolutionLayer<T, C, P, S>::circular_dim( const Dim dim, const Dim dim_size ) const
        {
            return ( ( dim + dim_size ) % dim_size );
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::valid ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->stride_dilation_dim( out_z_dim, filter_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->stride_dilation_dim( out_y_dim, filter_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->stride_dilation_dim( out_x_dim, filter_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->stride_pad_dilation_dim( out_z_dim, filter_z_dim, stride_shape.depth(), padding_shape.depth(), dilation_shape.depth() );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->stride_pad_dilation_dim( out_y_dim, filter_y_dim, stride_shape.height(), padding_shape.height(), dilation_shape.height() );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->stride_pad_dilation_dim( out_x_dim, filter_x_dim, stride_shape.width(), padding_shape.width(), dilation_shape.width() );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->circular_dim
                (
                    this->stride_pad_dilation_dim( out_z_dim, filter_z_dim, stride_shape.depth(), padding_shape.depth(), dilation_shape.depth() ),
                    input_shape.depth()
                );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->circular_dim
                    (
                        this->stride_pad_dilation_dim( out_y_dim, filter_y_dim, stride_shape.height(), padding_shape.height(), dilation_shape.height() ),
                        input_shape.height()
                    );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->circular_dim
                        (
                            this->stride_pad_dilation_dim( out_x_dim, filter_x_dim, stride_shape.width(), padding_shape.width(), dilation_shape.width() ),
                            input_shape.width()
                        );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::valid ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->stride_dilation_dim( ( out_z_dim / input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->stride_dilation_dim( ( out_y_dim / input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->stride_dilation_dim( ( out_x_dim / input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->stride_pad_dilation_dim( ( out_z_dim / input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), padding_shape.depth(), dilation_shape.depth() );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->stride_pad_dilation_dim( ( out_y_dim / input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), padding_shape.height(), dilation_shape.height() );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->stride_pad_dilation_dim( ( out_x_dim / input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), padding_shape.width(), dilation_shape.width() );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline V ConvolutionLayer<T, C, P, S>::filter_inference( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& padding_shape = this->get_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            V sum( 0 );

            Dim filter_c_dim = out_c_dim;
            Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );

            Dim in_c_dim = out_c_dim % input_shape.channels();

            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim filter_z_dim = 0; filter_z_dim < filter_shape.depth(); ++filter_z_dim )
            {
                //bool filter_z_bound = filter_shape.within_depth( filter_z_dim ) && filter_c_bound;
                Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                Dim     in_z_dim    = this->circular_dim
                (
                    this->stride_pad_dilation_dim( ( out_z_dim / input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), padding_shape.depth(), dilation_shape.depth() ),
                    input_shape.depth()
                );
                bool    in_z_bound  = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx     in_z_idx    = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim filter_y_dim = 0; filter_y_dim < filter_shape.height(); ++filter_y_dim )
                {
                    //bool filter_y_bound = filter_shape.within_height( filter_y_dim ) && filter_z_bound;
                    Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                    Dim     in_y_dim    = this->circular_dim
                    (
                        this->stride_pad_dilation_dim( ( out_y_dim / input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), padding_shape.height(), dilation_shape.height() ),
                        input_shape.height()
                    );
                    bool    in_y_bound  = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx     in_y_idx    = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim filter_x_dim = 0; filter_x_dim < filter_shape.width(); ++filter_x_dim )
                    {
                        //bool filter_x_bound = filter_shape.within_width( filter_x_dim ) && filter_y_bound;
                        Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                        Dim     in_x_dim    = this->circular_dim
                        (
                            this->stride_pad_dilation_dim( ( out_x_dim / input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), padding_shape.width(), dilation_shape.width() ),
                            input_shape.width()
                        );
                        bool    in_x_bound  = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx     in_index    = input_shape.width_index( in_y_idx, in_x_dim );

                        sum += this->get_filter( filter_index ) * ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) );
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline T ConvolutionLayer<T, C, P, S>::filter_backpropagation( const NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer, const Idx in_b_idx, const Idx out_b_idx, const Dim filter_c_dim, const Dim filter_z_dim, const Dim filter_y_dim, const Dim filter_x_dim )
        {
            const Shape<5>& input_shape = input_layer.get_shape();
            const Shape<5>& output_shape = output_layer.get_shape();

            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );
            Dim in_c_dim = filter_c_dim % input_shape.channels();

            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            T curr_jacob( 0 );

            for ( Dim out_z_dim = 0; out_z_dim < output_shape.depth(); ++out_z_dim )
            {
                //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_shape.depth() );
                Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                Dim in_z_dim = this->stride_dilation_dim( filter_z_dim, out_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool in_z_bound = input_shape.within_depth( in_z_dim );
                Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim out_y_dim = 0; out_y_dim < output_shape.height(); ++out_y_dim )
                {
                    //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_shape.height() );
                    Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                    Dim in_y_dim = this->stride_dilation_dim( filter_y_dim, out_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool in_y_bound = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim out_x_dim = 0; out_x_dim < output_shape.width(); ++out_x_dim )
                    {
                        //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_shape.width() );
                        Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                        Dim in_x_dim = this->stride_dilation_dim( filter_x_dim, out_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool in_x_bound = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                        curr_jacob += ( in_x_bound ? input_layer.get_node( in_index ) : U( 0 ) ) * output_layer.get_delta( out_index );
                    }
                }
            }

            return curr_jacob;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::valid ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->stride_dilation_dim( in_z_dim, filter_z_dim, stride_shape.depth(), dilation_shape.depth() );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->stride_dilation_dim( in_y_dim, filter_y_dim, stride_shape.height(), dilation_shape.height() );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->stride_dilation_dim( in_x_dim, filter_x_dim, stride_shape.width(), dilation_shape.width() );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim )requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& inv_padding_shape = this->get_inverse_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->stride_pad_dilation_dim( in_z_dim, filter_z_dim, stride_shape.depth(), inv_padding_shape.depth(), dilation_shape.depth() );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->stride_pad_dilation_dim( in_y_dim, filter_y_dim, stride_shape.height(), inv_padding_shape.height(), dilation_shape.height() );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->stride_pad_dilation_dim( in_x_dim, filter_x_dim, stride_shape.width(), inv_padding_shape.width(), dilation_shape.width() );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& inv_padding_shape = this->get_inverse_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->circular_dim
                    (
                        this->stride_pad_dilation_dim( in_z_dim, filter_z_dim, stride_shape.depth(), inv_padding_shape.depth(), dilation_shape.depth() ),
                        output_shape.depth()
                    );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->circular_dim
                        (
                            this->stride_pad_dilation_dim( in_y_dim, filter_y_dim, stride_shape.height(), inv_padding_shape.height(), dilation_shape.height() ),
                            output_shape.height()
                        );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->circular_dim
                            (
                                this->stride_pad_dilation_dim( in_x_dim, filter_x_dim, stride_shape.width(), inv_padding_shape.width(), dilation_shape.width() ),
                                output_shape.width()
                            );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::valid ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->stride_dilation_dim( ( in_z_dim * input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), dilation_shape.depth() );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->stride_dilation_dim( ( in_y_dim * input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), dilation_shape.height() );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->stride_dilation_dim( ( in_x_dim * input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), dilation_shape.width() );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& inv_padding_shape = this->get_inverse_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->stride_pad_dilation_dim( ( in_z_dim * input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), inv_padding_shape.depth(), dilation_shape.depth() );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->stride_pad_dilation_dim( ( in_y_dim * input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), inv_padding_shape.height(), dilation_shape.height() );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->stride_pad_dilation_dim( ( in_x_dim * input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), inv_padding_shape.width(), dilation_shape.width() );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        template <typename U, typename V>
        inline U ConvolutionLayer<T, C, P, S>::layer_backpropagation( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) )
        {
            const Shape<5>& output_shape = output_layer.get_shape();
            const Shape<4>& filter_shape = this->filter.get_shape();

            const Shape<3>& inv_padding_shape = this->get_inverse_padding();
            const Shape<3>& stride_shape = this->get_stride();
            const Shape<3>& dilation_shape = this->get_dilation();
            const Shape<3>& input_dilation_shape = this->get_input_dilation();

            U sum( 0 );

            for ( Dim filter_c_dim = 0; filter_c_dim < filter_shape.channels(); ++filter_c_dim )
            {
                Idx filter_c_idx = filter_shape.channel_index( filter_c_dim );
                Idx out_c_idx = output_shape.channel_index( out_b_idx, filter_c_dim );

                for( Dim filter_z_dim = filter_shape.depth() - 1; filter_z_dim >= 0; --filter_z_dim )
                {
                    Idx filter_z_idx = filter_shape.depth_index( filter_c_idx, filter_z_dim );

                    Dim out_z_dim = this->circular_dim
                    (
                        this->stride_pad_dilation_dim( ( in_z_dim * input_dilation_shape.depth() ), filter_z_dim, stride_shape.depth(), inv_padding_shape.depth(), dilation_shape.depth() ),
                        output_shape.depth()
                    );
                    bool out_z_bound = output_shape.within_depth( out_z_dim );
                    Idx out_z_idx = output_shape.depth_index( out_c_idx, out_z_dim );

                    for( Dim filter_y_dim = filter_shape.height() - 1; filter_y_dim >= 0; --filter_y_dim )
                    {
                        Idx filter_y_idx = filter_shape.height_index( filter_z_idx, filter_y_dim );

                        Dim out_y_dim = this->circular_dim
                        (
                            this->stride_pad_dilation_dim( ( in_y_dim * input_dilation_shape.height() ), filter_y_dim, stride_shape.height(), inv_padding_shape.height(), dilation_shape.height() ),
                            output_shape.height()
                        );
                        bool out_y_bound = output_shape.within_height( out_y_dim ) && out_z_bound;
                        Idx out_y_idx = output_shape.height_index( out_z_idx, out_y_dim );

                        for( Dim filter_x_dim = filter_shape.width() - 1; filter_x_dim >= 0; --filter_x_dim )
                        {
                            Idx filter_index = filter_shape.width_index( filter_y_idx, filter_x_dim );

                            Dim out_x_dim = this->circular_dim
                            (
                                this->stride_pad_dilation_dim( ( in_x_dim * input_dilation_shape.width() ), filter_x_dim, stride_shape.width(), inv_padding_shape.width(), dilation_shape.width() ),
                                output_shape.width()
                            );
                            bool out_bound = output_shape.within_width( out_x_dim ) && out_y_bound;
                            Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                            sum += this->get_filter( filter_index ) * ( out_bound ? output_layer.get_delta( out_index ) : V( 0 ) );
                        }
                    }
                }
            }

            return sum;
        }
} // namespace nn

//: Specializations
namespace nn
{
    //#define ClassMacro( convolution_type, padding_type, padding_size, type )\
        template class ConvolutionLayer<type, convolution_type, padding_type, padding_size>;

    #define FunctionMacroB( type_a, convolution_type, padding_type, padding_size, type_b, type_c )\
        template Error ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>::inference( const NodeLayer<type_b>& input_layer, NodeLayer<type_c>& output_layer ) const;\
        template Error ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>::backpropagation( NodeLayer<type_b>& input_layer, const NodeLayer<type_c>& output_layer );

    #define FunctionMacro( type_a, convolution_type, padding_type, padding_size, type_b )\
        FunctionMacroB( type_a, convolution_type, padding_type, padding_size, type_b, i32 )\
        FunctionMacroB( type_a, convolution_type, padding_type, padding_size, type_b, i64 )\
        FunctionMacroB( type_a, convolution_type, padding_type, padding_size, type_b, f32 )\
        FunctionMacroB( type_a, convolution_type, padding_type, padding_size, type_b, f64 )

    #define ClassMacroE( type_a, convolution_type, padding_type, padding_size )\
        template class ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>;\
        FunctionMacro( type_a, convolution_type, padding_type, padding_size, i32 )\
        FunctionMacro( type_a, convolution_type, padding_type, padding_size, i64 )\
        FunctionMacro( type_a, convolution_type, padding_type, padding_size, f32 )\
        FunctionMacro( type_a, convolution_type, padding_type, padding_size, f64 )

    #define ClassMacroD( type_a, convolution_type, padding_type )\
        ClassMacroE( type_a, convolution_type, padding_type, PaddingSize::valid )\
        ClassMacroE( type_a, convolution_type, padding_type, PaddingSize::same )\
        ClassMacroE( type_a, convolution_type, padding_type, PaddingSize::full )\
        ClassMacroE( type_a, convolution_type, padding_type, PaddingSize::custom )

    #define ClassMacroC( type_a, convolution_type )\
        ClassMacroD( type_a, convolution_type, PaddingType::zero )\
        ClassMacroD( type_a, convolution_type, PaddingType::circular )

    #define ClassMacroB( type_a )\
        ClassMacroC( type_a, ConvolutionType::down_sample )\
        ClassMacroC( type_a, ConvolutionType::up_sample )

    #define ClassMacro\
        ClassMacroB( i32 )\
        ClassMacroB( i64 )\
        ClassMacroB( f32 )\
        ClassMacroB( f64 )

    ClassMacro

    #undef FunctionMacroB
    #undef FunctionMacro
    #undef ClassMacroE
    #undef ClassMacroD
    #undef ClassMacroC
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn

/*
#define CLASS( type_a, convolution_type, padding_type, padding_size )\
        template class ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>;\

    #define FUNCTION( type_a, convolution_type, padding_type, padding_size, type_b, type_c )\
        template Error ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>::inference( const NodeLayer<type_b>& input_layer, NodeLayer<type_c>& output_layer ) const;\
        template Error ConvolutionLayer<type_a, convolution_type, padding_type, padding_size>::backpropagation( NodeLayer<type_b>& input_layer, const NodeLayer<type_c>& output_layer );

    CLASS( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
        CLASS( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    CLASS( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
        CLASS( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    CLASS( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
        CLASS( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    CLASS( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
        CLASS( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    CLASS( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
        CLASS( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )

    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, bool, bool )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, bool, i32 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, bool, i64 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, bool, f32 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, bool, f64 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, i32, bool )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, i32, i32 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, i32, i64 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, i32, f32 )
        FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid, i32, f64 )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
    FUNCTION( bool, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
    FUNCTION( i32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
    FUNCTION( i64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
    FUNCTION( f32, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::valid )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::same )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::full )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::zero, PaddingSize::custom )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::valid )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::same )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::full )
    FUNCTION( f64, ConvolutionType::down_sample, PaddingType::circular, PaddingSize::custom )
*/