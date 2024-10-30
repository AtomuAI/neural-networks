// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>

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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/convolution_layer/convolution_layer.cpp");
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

//: Library Headers
#include <bewusstsein_util.hpp>

//: Project Headers
#include "c++/bewusstsein_neural_networks/source/core/counter/counter.hpp"
#include "c++/bewusstsein_neural_networks/source/core/step_size/step_size.hpp"
#include "c++/bewusstsein_neural_networks/source/core/beta/beta.hpp"
#include "c++/bewusstsein_neural_networks/source/core/epsilon/epsilon.hpp"
#include "c++/bewusstsein_neural_networks/source/core/training_mode/training_mode.hpp"
#include "c++/bewusstsein_neural_networks/source/core/initialization_type/initialization_type.hpp"
#include "c++/bewusstsein_neural_networks/source/core/distribution_type/distribution_type.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: This Header
# include "c++/bewusstsein_neural_networks/source/layers/convolution_layer/convolution_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer() : BaseLayer( LayerType::convolution_layer, "" ), {}

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ) ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( this->padding_type )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape[ 0 ] - 1 ) * dilation.width() ) / 2, ( ( filter_shape[ 1 ] - 1 ) * dilation.height() ) / 2, ( ( filter_shape[ 2 ] - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = {( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth()};
                    this->inv_padding = {0, 0, 0};
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation, const T scalar ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ), scalar ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( this->padding_type )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape[ 0 ] - 1 ) * dilation.width() ) / 2, ( ( filter_shape[ 1 ] - 1 ) * dilation.height() ) / 2, ( ( filter_shape[ 2 ] - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = util::Shape( ( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth() );
                    this->inv_padding = util::Shape( 0, 0, 0 );
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation, const std::vector<T> data ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ), data ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( padding_type )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape[ 0 ] - 1 ) * dilation.width() ) / 2, ( ( filter_shape[ 1 ] - 1 ) * dilation.height() ) / 2, ( ( filter_shape[ 2 ] - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = util::Shape( ( filter_shape[ 0 ] - 1 ) * dilation.width(), ( filter_shape[ 1 ] - 1 ) * dilation.height(), ( filter_shape[ 2 ] - 1 ) * dilation.depth() );
                    this->inv_padding = util::Shape( 0, 0, 0 );
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ) ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation, const T scalar ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ), scalar ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation, const std::vector<T> data ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ), data ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation, TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ) ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = {( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth()};
                    this->inv_padding = {0, 0, 0};
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }

            this->allocate_training_memory( training_mode );
        }
        */

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation, const T scalar, const TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer ), filter( util::Shape( filter_shape, 4 ), scalar ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( this->padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = util::Shape( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = util::Shape( 0, 0, 0 );
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const PaddingSize padding_size, const util::Shape& input_dilation, const util::Shape& stride, const util::Shape& dilation, const std::vector<T> data, const TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer ), filter( util::Shape( filter_shape, 4 ), data ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( padding_size ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            switch ( padding_size )
            {
                case PaddingSize::valid:
                {
                    this->padding = util::Shape( 0, 0, 0 );
                    this->inv_padding = util::Shape( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    break;
                }

                case PaddingSize::same:
                {
                    this->padding = util::Shape( ( ( filter_shape.width() - 1 ) * dilation.width() ) / 2, ( ( filter_shape.height() - 1 ) * dilation.height() ) / 2, ( ( filter_shape.depth() - 1 ) * dilation.depth() ) / 2 );
                    this->inv_padding = this->padding;
                    break;
                }

                case PaddingSize::full:
                {
                    this->padding = util::Shape( ( filter_shape.width() - 1 ) * dilation.width(), ( filter_shape.height() - 1 ) * dilation.height(), ( filter_shape.depth() - 1 ) * dilation.depth() );
                    this->inv_padding = util::Shape( 0, 0, 0 );
                    break;
                }

                default: { throw std::invalid_argument( "Padding Type is invalid/uninitialized" ); break; }
            }

            this->allocate_training_memory( training_mode );
        }

        /*
        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const std::string& name, const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation, TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer, name ), filter( util::Shape( filter_shape, 4 ) ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            this->allocate_training_memory( training_mode );
        }
        */

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation, const T scalar, const TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer ), filter( util::Shape( filter_shape, 4 ), scalar ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        ConvolutionLayer<T>::ConvolutionLayer( const util::Shape& filter_shape, const ConvolutionType convolution_type, const PaddingType padding_type, const util::Shape& input_dilation, const util::Shape& padding, const util::Shape& stride, const util::Shape& dilation, const std::vector<T> data, const TrainingMode training_mode ) :
            BaseLayer( LayerType::convolution_layer ), filter( util::Shape( filter_shape, 4 ), data ), padding( padding ), stride( stride ), dilation( dilation ), conv_type( convolution_type ), padding_type( padding_type ), padding_size( PaddingSize::custom ), input_dilation( input_dilation )
        {
            if( ( filter_shape.size() > 4 ) || ( filter_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Convlayer filter can have at least 1 and at most 4 dimensions." );
            }

            this->allocate_training_memory( training_mode );
        }

    //: Destructors
        template <typename T>
        ConvolutionLayer<T>::~ConvolutionLayer() {}

    //: Methods
        template <typename T>
        const util::Tensor<T>& ConvolutionLayer<T>::get_filter() const
        {
            return this->filter;
        }

        template <typename T>
        const util::Tensor<T>& ConvolutionLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const util::Tensor<T>& ConvolutionLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const util::Tensor<T>& ConvolutionLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        ConvolutionType ConvolutionLayer<T>::get_convolution_type() const
        {
            return this->conv_type;
        }

        template <typename T>
        const util::Shape& ConvolutionLayer<T>::get_stride() const
        {
            return this->stride;
        }

        template <typename T>
        const util::Shape& ConvolutionLayer<T>::get_padding() const
        {
            return this->padding;
        }

        template <typename T>
        const util::Shape& ConvolutionLayer<T>::get_dilation() const
        {
            return this->dilation;
        }

        template <typename T>
        const util::Shape& ConvolutionLayer<T>::get_input_dilation() const
        {
            return this->input_dilation;
        }

        template <typename T>
        const util::Shape& ConvolutionLayer<T>::get_inverse_padding() const
        {
            return this->inv_padding;
        }

        template <typename T>
        PaddingType ConvolutionLayer<T>::get_padding_type() const
        {
            return this->padding_type;
        }

        template <typename T>
        PaddingSize ConvolutionLayer<T>::get_padding_size() const
        {
            return this->padding_size;
        }

        template <typename T>
        void ConvolutionLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape shape = this->filter.get_shape();
                    this->jacobian.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape shape = this->filter.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape shape = this->filter.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    this->velocity.resize( shape );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Invalid training mode selection" );
                }
            }
        }

        template <typename T>
        void ConvolutionLayer<T>::initialize( const util::Shape& input_layer_shape, const util::Shape& output_layer_shape, const InitializationType type, const DistributionType distribution )
        {
            util::Size input_layer_size = input_layer_shape.distance( 0, input_layer_shape.size() - 1 );
            util::Size output_layer_size = output_layer_shape.distance( 0, output_layer_shape.size() - 1 );
            T variance = 0;
            T std_dev = 0;
            switch ( type )
            {
                case InitializationType::xavier_glorot: { variance = 2.0 / ( input_layer_size + output_layer_size ); std_dev = sqrt( variance ); break; }
                case InitializationType::he: { variance = 2.0 / ( input_layer_size ); std_dev = sqrt( variance ); break; }
                case InitializationType::lecun: { variance = 1.0 / ( input_layer_size ); std_dev = sqrt( variance ); break; }
                default: { throw std::invalid_argument( "Initialization Type is invalid/uninitialized" ); break; }
            }

            switch ( distribution )
            {
                case DistributionType::normal: { this->filter.fill_normal_distribution( 0, std_dev ); break; }
                /*
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
                */
                default: { throw std::invalid_argument( "Distribution Type is invalid/uninitialized" ); break; }
            };
        }

        template <typename T>
        void ConvolutionLayer<T>::randomize( const T min, const T max )
        {
            filter.randomize( min, max );
        }

        template <typename T>
        void ConvolutionLayer<T>::print() const
        {
            std::cout << "ConvolutionLayer<T>.Print:" << std::endl << "{" << std::endl;
            std::cout << "Filter:" << std::endl;
            filter.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ConvolutionLayer<T>::print_jacobian() const
        {
            std::cout << "ConvolutionLayer<T>.PrintJacobian:" << std::endl << "{" << std::endl;
            std::cout << "Jacobian:" << std::endl;
            jacobian.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ConvolutionLayer<T>::info() const
        {
            util::Size padding_size = this->padding.size();
            util::Size stride_size = this->stride.size();
            util::Size dilation_size = this->dilation.size();

            std::cout << "ActivationLayer<T>.Info:" << std::endl << "{" << std::endl;

            std::cout << "Padding: [ ";
            this->padding.print();

            std::cout << "Stride: [ ";
            this->stride.print();

            std::cout << "Dilation: [ ";
            this->dilation.print();

            std::cout << "Padding Type: ";
            switch( this->padding_type )
            {
                case PaddingSize::valid:
                {
                    std::cout << "valid" << std::endl;
                    break;
                }

                case PaddingSize::same:
                {
                    std::cout << "same" << std::endl;
                    break;
                }

                case PaddingSize::full:
                {
                    std::cout << "full" << std::endl;
                    break;
                }

                case PaddingSize::custom:
                {
                    std::cout << "full" << std::endl;
                    break;
                }

                default:
                {
                    throw std::invalid_argument( "Padding Size is invalid/uninitialized" );
                    break;
                }
            }

            std::cout << "Padding Value: ";
            switch( this->padding_value )
            {
                case PaddingType::zero:
                {
                    std::cout << "zero" << std::endl;
                    break;
                }

                case PaddingType::circular:
                {
                    std::cout << "circular" << std::endl;
                    break;
                }

                default:
                {
                    throw std::invalid_argument( "Padding Type is invalid/uninitialized" );
                    break;
                }
            }

            filter.info();

            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ConvolutionLayer<T>::save_model( const std::string& file_name ) const
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
        void ConvolutionLayer<T>::save_state( const std::string& file_name ) const
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
        void ConvolutionLayer<T>::load_model( const std::string& file_name )
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
        void ConvolutionLayer<T>::load_state( const std::string& file_name )
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
        void ConvolutionLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Padding Type
            file.write( reinterpret_cast<const char*>( &this->padding_type ), sizeof( PaddingType ) );

            // Save Padding Value
            file.write( reinterpret_cast<const char*>( &this->padding_size ), sizeof( PaddingSize ) );

            // Save Padding
            this->padding.save( file );

            // Save Stride
            this->stride.save( file );

            // Save Dilation
            this->dilation.save( file );

            // Save Filter Tensor
            this->filter.save( file );
        }

        template <typename T>
        void ConvolutionLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Padding Type
            file.write( reinterpret_cast<const char*>( &this->padding_type ), sizeof( PaddingType ) );

            // Save Padding size
            file.write( reinterpret_cast<const char*>( &this->padding_size ), sizeof( PaddingSize ) );

            // Save Padding
            this->padding.save( file );

            // Save Stride
            this->stride.save( file );

            // Save Dilation
            this->dilation.save( file );

            // Save Filter Tensor
            this->filter.save( file );

            // Save Jacobian Tensor
            this->jacobian.save( file );

            // Save momentum Tensor
            this->momentum.save( file );

            // Save velocity Tensor
            this->velocity.save( file );

            // Save Time Step
            file.write( reinterpret_cast<const char*>( &this->tim_step ), sizeof( int ) );
        }

        template <typename T>
        void ConvolutionLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Padding Type
            file.read( reinterpret_cast<char*>( &this->padding_type ), sizeof( PaddingType ) );

            // Load Padding Value
            file.read( reinterpret_cast<char*>( &this->padding_size ), sizeof( PaddingSize ) );

            // Load Padding
            this->padding.load( file );

            // Load Stride
            this->stride.load( file );

            // Load Dilation
            this->dilation.load( file );

            // Load Filter Tensor
            this->filter.load( file );
        }

        template <typename T>
        void ConvolutionLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Padding Type
            file.read( reinterpret_cast<char*>( &this->padding_type ), sizeof( PaddingType ) );

            // Load Padding Value
            file.read( reinterpret_cast<char*>( &this->padding_size ), sizeof( PaddingSize ) );

            // Load Padding
            this->padding.load( file );

            // Load Stride
            this->stride.load( file );

            // Load Dilation
            this->dilation.load( file );

            // Load Filter Tensor
            this->filter.load( file );

            // Load Jacobian Tensor
            this->jacobian.load( file );

            // Load momentum Tensor
            this->momentum.load( file );

            // Load velocity Tensor
            this->velocity.load( file );

            // Load Time Step
            file.read( reinterpret_cast<char*>( &this->time_step ), sizeof( int ) );
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            if ( this->conv_type == ConvolutionType::normal )
            {
                if ( this->padding_size == PaddingSize::valid )
                {
                    this->inference_nonpadded( input_layer, output_layer );
                }

                else if ( !( this->padding_size == PaddingSize::valid ) )
                {
                    if ( this->padding_type == PaddingType::zero )
                    {
                        this->inference_zeropadded( input_layer, output_layer );
                    }

                    else if ( this->padding_type == PaddingType::circular )
                    {
                        this->inference_circularpadded( input_layer, output_layer );
                    }

                    else
                    {
                        throw std::invalid_argument( "Padding Type is invalid/uninitialized" );
                    }
                }

                else
                {
                    throw std::invalid_argument( "Padding Size is invalid/uninitialized" );
                }
            }
            else
            {
                if ( this->padding_size == PaddingSize::valid )
                {
                    this->inference_transposed_nonpadded( input_layer, output_layer );
                }

                else if ( !( this->padding_size == PaddingSize::valid ) )
                {
                    if ( this->padding_type == PaddingType::zero )
                    {
                        this->inference_transposed_zeropadded( input_layer, output_layer );
                    }

                    else if ( this->padding_type == PaddingType::circular )
                    {
                        this->inference_transposed_circularpadded( input_layer, output_layer );
                    }

                    else
                    {
                        throw std::invalid_argument( "Padding Type is invalid/uninitialized" );
                    }
                }

                else
                {
                    throw std::invalid_argument( "Padding Size is invalid/uninitialized" );
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            if ( this->padding_size == PaddingSize::full )
            {
                this->backpropagation_nonpadded( input_layer, output_layer );
            }

            else if ( this->padding_size == PaddingSize::same )
            {
                if ( this->padding_type == PaddingType::zero )
                {
                    this->backpropagation_zeropadded( input_layer, output_layer );
                }

                else if ( this->padding_type == PaddingType::circular )
                {
                    this->backpropagation_circularpadded( input_layer, output_layer );
                }

                else
                {
                    throw std::invalid_argument( "Padding Type is invalid/uninitialized" );
                }
            }

            else if ( this->padding_size == PaddingSize::valid )
            {
                this->backpropagation_zeropadded( input_layer, output_layer );
            }

            else
            {
                throw std::invalid_argument( "Padding Size is invalid/uninitialized" );
            }
        }

        template <typename T>
        void ConvolutionLayer<T>::gradient_decent( const util::Dim batch_size, const StepSize step_size )
        {
            util::Size filter_size = this->filter.get_size();

            for ( util::Index index = 0; index < filter_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                this->filter[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->filter[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void ConvolutionLayer<T>::gradient_decent_momentum( const util::Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            util::Size filter_size = this->filter.get_size();

            for ( util::Index index = 0; index < filter_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;
                T momentum_value = this->momentum[ index ];

                this->momentum[ index ] = gradient;

                this->filter[ index ] += step_size * ( ( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->filter[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void ConvolutionLayer<T>::gradient_decent_adam( const util::Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            uint32_t step = this->time_step.get_count();
            Beta beta1_mp = 1 - pow( beta1, step );
            Beta beta2_mp = 1 - pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            util::Size filter_size = this->filter.get_size();

            for ( util::Index index = 0; index < filter_size; index++ )
            {
                // Gradient
                T gradient = this->jacobian[ index ] / batch_size;

                // Update momentum
                T momentum_value = this->momentum[ index ];
                momentum_value = ( beta1 * momentum_value ) + ( beta1_m * gradient );
                this->momentum[ index ] = momentum_value;

                // Update velocity
                T velocity_value = this->velocity[ index ];
                velocity_value = ( beta2 * velocity_value ) + ( beta2_m * ( gradient * gradient ) );
                this->velocity[ index ] = velocity_value;

                // Compute bias-corrected momentum and velocity
                T momentum_hat = momentum_value / beta1_mp;
                T velocity_hat = velocity_value / beta2_mp;

                // Update weight
                this->filter[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );


                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->filter[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_zeropadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;

                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                V sum = 0;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = out_z_dim * stride_size.z - padding_size.z + filter_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = out_y_dim * stride_size.y - padding_size.y + filter_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = out_x_dim * stride_size.x - padding_size.x + filter_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                        }
                                    }
                                }

                                output_layer[ out_index ] = sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_circularpadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;

                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                V sum = 0;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = out_z_dim * stride_size.z - padding_size.z + filter_z_dim * dilation_size.z;
                                    in_z_dim = ( ( in_z_dim + input_size.z ) % input_size.z );
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = out_y_dim * stride_size.y - padding_size.y + filter_y_dim * dilation_size.y;
                                        in_y_dim = ( ( in_y_dim + input_size.y ) % input_size.y );
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = out_x_dim * stride_size.x - padding_size.x + filter_x_dim * dilation_size.x;
                                            in_x_dim = ( ( in_x_dim + input_size.x ) % input_size.x );
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                        }
                                    }
                                }

                                output_layer[ out_index ] = sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_nonpadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;

                //std::cout << "[ " << "Batch:" << ( int )out_b_dim << " ]" << std::endl;

                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    //std::cout << "\t" << "[ " << "Output-C:" << ( int )out_c_dim << " ]" << std::endl;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        //std::cout << "\t\t" << "[ " << "Output-Z:" << ( int )out_z_dim << " ]" << std::endl;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            //std::cout << "\t\t\t" << "[ " << "Output-Y:" << ( int )out_y_dim << " ]" << std::endl;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                V sum = 0;

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = out_z_dim * stride_size.z + filter_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound;
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    //std::cout << "\t\t\t\t\t\t" << "[ " << "Filter-Z:" << ( int )filter_z_dim << " : " << "Input-Z:" << ( int )in_z_dim << " ]" << std::endl;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = out_y_dim * stride_size.y + filter_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        //std::cout << "\t\t\t\t\t\t\t" << "[ " << "Filter-Y:" << ( int )filter_y_dim << " : " << "Input-Y:" << ( int )in_y_dim << " ]" << std::endl;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = out_x_dim * stride_size.x + filter_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            //std::cout << "\t\t\t\t\t\t\t\t" << "[ " << "Filter-X:" << ( int )filter_x_dim << " : " << "Input-X:" << ( int )in_x_dim << " : " << "Filter-Index:" << ( int )filter_index << " : " << "Input-Index:" << ( int )in_index << " ]" << std::endl;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                        }
                                    }
                                }

                                output_layer[ out_index ] = sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::backpropagation_zeropadded( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            input_layer.zero_delta();

            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 inv_padding_size( this->inv_padding.width(), this->inv_padding.height(), this->inv_padding.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim b = 0; b < output_size.b; b++ )
            {
                util::Index out_b_idx = b * output_size.c;
                util::Index in_b_idx = b * input_size.c;

                //std::cout << "[ " << "Batch:" << b << " ]" << std::endl;

                for( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                {
                    //bool filter_c_bound = ( filter_c_dim >= 0 ) && ( filter_c_dim < filter_size.c );
                    util::Index filter_c_idx = filter_c_dim * filter_size.z;

                    //std::cout << "\t" << "[ " << "Filter-C:" << ( int )filter_c_dim << " ]" << std::endl;

                    for( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                    {
                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                        //std::cout << "\t\t" << "[ " << "Filter-Z:" << ( int )filter_z_dim << " ]" << std::endl;

                        for( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                        {
                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                            //std::cout << "\t\t\t" << "[ " << "Filter-Y:" << ( int )filter_y_dim << " ]" << std::endl;

                            for( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                            {
                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                util::Index filter_index = filter_y_idx + filter_x_dim;
                                util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;
                                util::Dim in_c_dim = filter_c_dim % input_size.c;

                                util::Index in_c_idx = ( b + in_c_dim ) * input_size.z;

                                T curr_jacob = 0;

                                for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                                {
                                    //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                    util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                    util::Dim in_z_dim = filter_z_dim * stride_size.z + out_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    //std::cout << "\t\t\t\t\t\t" << "[ " << "Output-Z:" << ( int )out_z_dim << " : " << "Input-Z:" << ( int )in_z_dim << " ]" << std::endl;

                                    for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                                    {
                                        //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                                        util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                        util::Dim in_y_dim = filter_y_dim * stride_size.y + out_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        //std::cout << "\t\t\t\t\t\t\t" << "[ " << "Output-Y:" << ( int )out_y_dim << " : " << "Input-Y:" << ( int )in_y_dim << " ]" << std::endl;

                                        for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                                        {
                                            //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                            util::Index out_index = out_y_idx + out_x_dim;

                                            util::Dim in_x_dim = filter_x_dim * stride_size.x + out_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            //std::cout << "\t\t\t\t\t\t\t\t" << "[ " << "Output-X:" << ( int )out_x_dim << " : " << "Input-X:" << ( int )in_x_dim << " : " << "Output-Index:" << ( int )out_index << " : " << "Input-Index:" << ( int )in_index << " ]" << std::endl;

                                            curr_jacob += ( in_bound ? input_layer[ in_index ] : 0 ) * output_layer.get_delta( out_index );
                                            //std::cout << "\t\t\t\t\t\t\t\t\t" << curr_jacob << " += " << ( in_bound ? input_layer[ in_index ] : 0 ) << " * " << output_layer.get_delta( out_index ) << std::endl;
                                        }
                                    }
                                }
                                jacobian[ filter_index ] += curr_jacob;
                            }
                        }
                    }
                }

                for ( util::Dim in_c_dim = 0; in_c_dim < input_size.c; in_c_dim++ )
                {
                    //bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                    util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                    //std::cout << "\t" << "[ " << "Input-C:" << ( int )in_c_dim << " ]" << std::endl;

                    for ( util::Dim in_z_dim = 0; in_z_dim < input_size.z; in_z_dim++ )
                    {
                        //bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                        util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                        //std::cout << "\t\t" << "[ " << "Input-Z:" << ( int )in_z_dim << " ]" << std::endl;

                        for ( util::Dim in_y_dim = 0; in_y_dim < input_size.y; in_y_dim++ )
                        {
                            //bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y );
                            util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                            //std::cout << "\t\t\t" << "[ " << "Input-Y:" << ( int )in_y_dim << " ]" << std::endl;

                            for ( util::Dim in_x_dim = 0; in_x_dim < input_size.x; in_x_dim++ )
                            {
                                //bool in_x_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x );
                                util::Index in_index = in_y_idx + in_x_dim;

                                //std::cout << "\t\t\t\t" << "[ " << "Input-X:" << ( int )in_x_dim << " : " << "Input-Index:" << ( int )in_index << " ]" << std::endl;

                                U sum = 0;

                                for ( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                                {
                                    util::Index filter_c_idx = filter_c_dim * filter_size.z;
                                    util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;

                                    //std::cout << "\t\t\t\t\t" << "[ " << "Filter-C:" << ( int )filter_c_dim << " : " << "Output-C:" << ( int )out_c_dim << " ]" << std::endl;

                                    for( util::Dim filter_z_dim = filter_size.z - 1; filter_z_dim >= 0; filter_z_dim-- )
                                    {
                                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                        util::Dim out_z_dim = in_z_dim * stride_size.z - inv_padding_size.z + filter_z_dim * dilation_size.z;
                                        bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                        //std::cout << "\t\t\t\t\t\t" << "[ " << "Filter-Z:" << ( int )filter_z_dim << " : " << "Output-Z:" << ( int )out_z_dim << " ]" << std::endl;

                                        for( util::Dim filter_y_dim = filter_size.y - 1; filter_y_dim >= 0; filter_y_dim-- )
                                        {
                                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                            util::Dim out_y_dim = in_y_dim * stride_size.y - inv_padding_size.y + filter_y_dim * dilation_size.y;
                                            bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y ) && out_z_bound;
                                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                            //std::cout << "\t\t\t\t\t\t\t" << "[ " << "Filter-Y:" << ( int )filter_y_dim << " : " << "Output-Y:" << ( int )out_y_dim << " ]" << std::endl;

                                            for( util::Dim filter_x_dim = filter_size.x - 1; filter_x_dim >= 0; filter_x_dim-- )
                                            {
                                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                                util::Index filter_index = filter_y_idx + filter_x_dim;

                                                util::Dim out_x_dim = in_x_dim * stride_size.x - inv_padding_size.x + filter_x_dim * dilation_size.x;
                                                bool out_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x ) && out_y_bound;
                                                util::Index out_index = out_y_idx + out_x_dim;

                                                //std::cout << "\t\t\t\t\t\t\t\t" << "[ " << "Filter-X:" << ( int )filter_x_dim << " : " << "Output-X:" << ( int )out_x_dim << " : " << "Filter-Index:" << ( int )filter_index << " : " << "Output-Index:" << ( int )out_index << " ]" << std::endl;

                                                sum += filter[ filter_index ] * ( out_bound ? output_layer.get_delta( out_index ) : 0 );
                                                //std::cout << "\t\t\t\t\t\t\t\t\t" << sum << " += " << ( out_bound ? output_layer.get_delta( out_index ) : 0 )  << " * " << filter[ filter_index ] << std::endl;
                                            }
                                        }
                                    }
                                }
                                input_layer.get_delta( in_index ) += sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::backpropagation_circularpadded( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            input_layer.zero_delta();

            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 inv_padding_size( this->inv_padding.width(), this->inv_padding.height(), this->inv_padding.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim b = 0; b < output_size.b; b++ )
            {
                util::Index out_b_idx = b * output_size.c;
                util::Index in_b_idx = b * input_size.c;

                for( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                {
                    //bool filter_c_bound = ( filter_c_dim >= 0 ) && ( filter_c_dim < filter_size.c );
                    util::Index filter_c_idx = filter_c_dim * filter_size.z;

                    for( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                    {
                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                        for( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                        {
                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                            for( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                            {
                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                util::Index filter_index = filter_y_idx + filter_x_dim;
                                util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;
                                util::Dim in_c_dim = filter_c_dim % input_size.c;

                                util::Index in_c_idx = ( b + in_c_dim ) * input_size.z;

                                T curr_jacob = 0;

                                for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                                {
                                    //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                    util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                    util::Dim in_z_dim = filter_z_dim * stride_size.z + out_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                                    {
                                        //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                                        util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                        util::Dim in_y_dim = filter_y_dim * stride_size.y + out_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                                        {
                                            //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                            util::Index out_index = out_y_idx + out_x_dim;

                                            util::Dim in_x_dim = filter_x_dim * stride_size.x + out_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            curr_jacob += ( in_bound ? input_layer[ in_index ] : 0 ) * output_layer.get_delta( out_index );
                                        }
                                    }
                                }
                                jacobian[ filter_index ] += curr_jacob;
                            }
                        }
                    }
                }

                for ( util::Dim in_c_dim = 0; in_c_dim < input_size.c; in_c_dim++ )
                {
                    //bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                    util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                    for ( util::Dim in_z_dim = 0; in_z_dim < input_size.z; in_z_dim++ )
                    {
                        //bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                        util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                        for ( util::Dim in_y_dim = 0; in_y_dim < input_size.y; in_y_dim++ )
                        {
                            //bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y );
                            util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                            for ( util::Dim in_x_dim = 0; in_x_dim < input_size.x; in_x_dim++ )
                            {
                                //bool in_x_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x );
                                util::Index in_index = in_y_idx + in_x_dim;

                                U sum = 0;

                                for ( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                                {
                                    util::Index filter_c_idx = filter_c_dim * filter_size.z;
                                    util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;

                                    for( util::Dim filter_z_dim = filter_size.z - 1; filter_z_dim >= 0; filter_z_dim-- )
                                    {
                                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                        util::Dim out_z_dim = in_z_dim * stride_size.z - inv_padding_size.z + filter_z_dim * dilation_size.z;
                                        out_z_dim = ( ( out_z_dim + output_size.z ) % input_size.z );
                                        bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                        for( util::Dim filter_y_dim = filter_size.y - 1; filter_y_dim >= 0; filter_y_dim-- )
                                        {
                                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                            util::Dim out_y_dim = in_y_dim * stride_size.y - inv_padding_size.y + filter_y_dim * dilation_size.y;
                                            out_y_dim = ( ( out_y_dim + output_size.y ) % input_size.y );
                                            bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y ) && out_z_bound;
                                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                            for( util::Dim filter_x_dim = filter_size.x - 1; filter_x_dim >= 0; filter_x_dim-- )
                                            {
                                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                                util::Index filter_index = filter_y_idx + filter_x_dim;

                                                util::Dim out_x_dim = in_x_dim * stride_size.x - inv_padding_size.x + filter_x_dim * dilation_size.x;
                                                out_x_dim = ( ( out_x_dim + output_size.x ) % input_size.x );
                                                bool out_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x ) && out_y_bound;
                                                util::Index out_index = out_y_idx + out_x_dim;

                                                sum += filter[ filter_index ] * ( out_bound ? output_layer.get_delta( out_index ) : 0 );
                                            }
                                        }
                                    }
                                }
                                input_layer.get_delta( in_index ) += sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::backpropagation_nonpadded( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            input_layer.zero_delta();

            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 inv_padding_size( this->inv_padding.width(), this->inv_padding.height(), this->inv_padding.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim b = 0; b < output_size.b; b++ )
            {
                util::Index out_b_idx = b * output_size.c;
                util::Index in_b_idx = b * input_size.c;

                for( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                {
                    //bool filter_c_bound = ( filter_c_dim >= 0 ) && ( filter_c_dim < filter_size.c );
                    util::Index filter_c_idx = filter_c_dim * filter_size.z;

                    for( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                    {
                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                        for( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                        {
                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                            for( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                            {
                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                util::Index filter_index = filter_y_idx + filter_x_dim;
                                util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;
                                util::Dim in_c_dim = filter_c_dim % input_size.c;

                                util::Index in_c_idx = ( b + in_c_dim ) * input_size.z;

                                T curr_jacob = 0;

                                for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                                {
                                    //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                    util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                    util::Dim in_z_dim = filter_z_dim * stride_size.z + out_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                                    {
                                        //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                                        util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                        util::Dim in_y_dim = filter_y_dim * stride_size.y + out_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound;
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                                        {
                                            //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                            util::Index out_index = out_y_idx + out_x_dim;

                                            util::Dim in_x_dim = filter_x_dim * stride_size.x + out_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound;
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            curr_jacob += ( in_bound ? input_layer[ in_index ] : 0 ) * output_layer.get_delta( out_index );
                                        }
                                    }
                                }
                                jacobian[ filter_index ] += curr_jacob;
                            }
                        }
                    }
                }

                for ( util::Dim in_c_dim = 0; in_c_dim < input_size.c; in_c_dim++ )
                {
                    //bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                    util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                    for ( util::Dim in_z_dim = 0; in_z_dim < input_size.z; in_z_dim++ )
                    {
                        //bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z );
                        util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                        for ( util::Dim in_y_dim = 0; in_y_dim < input_size.y; in_y_dim++ )
                        {
                            //bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y );
                            util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                            for ( util::Dim in_x_dim = 0; in_x_dim < input_size.x; in_x_dim++ )
                            {
                                //bool in_x_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x );
                                util::Index in_index = in_y_idx + in_x_dim;

                                U sum = 0;

                                for ( util::Dim filter_c_dim = 0; filter_c_dim < filter_size.c; filter_c_dim++ )
                                {
                                    util::Index filter_c_idx = filter_c_dim * filter_size.z;
                                    util::Index out_c_idx = ( out_b_idx + filter_c_dim ) * output_size.z;

                                    for( util::Dim filter_z_dim = filter_size.z - 1; filter_z_dim >= 0; filter_z_dim-- )
                                    {
                                        //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                        util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                        util::Dim out_z_dim = in_z_dim * stride_size.z + filter_z_dim * dilation_size.z;
                                        bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                                        for( util::Dim filter_y_dim = filter_size.y - 1; filter_y_dim >= 0; filter_y_dim-- )
                                        {
                                            //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                            util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                            util::Dim out_y_dim = in_y_dim * stride_size.y + filter_y_dim * dilation_size.y;
                                            bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y ) && out_z_bound;
                                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                                            for( util::Dim filter_x_dim = filter_size.x - 1; filter_x_dim >= 0; filter_x_dim-- )
                                            {
                                                //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                                util::Index filter_index = filter_y_idx + filter_x_dim;

                                                util::Dim out_x_dim = in_x_dim * stride_size.x + filter_x_dim * dilation_size.x;
                                                bool out_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x ) && out_y_bound;
                                                util::Index out_index = out_y_idx + out_x_dim;

                                                sum += filter[ filter_index ] * ( out_bound ? output_layer.get_delta( out_index ) : 0 );
                                            }
                                        }
                                    }
                                }
                                input_layer.get_delta( in_index ) += sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_transposed_zeropadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 input_dilation_size( this->input_dilation.width(), this->input_dilation.height(), this->input_dilation.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );


            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;


                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                V sum = 0;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = ( out_z_dim/input_dilation_size.z ) * stride_size.z - padding_size.z + filter_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound && ( ( ( out_z_dim + filter_z_dim ) % input_dilation_size.z ) == 0 );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = ( out_y_dim/input_dilation_size.y ) * stride_size.y - padding_size.y + filter_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound && ( ( ( out_y_dim + filter_y_dim ) % input_dilation_size.y ) == 0 );
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = ( out_x_dim/input_dilation_size.x ) * stride_size.x - padding_size.x + filter_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound && ( ( ( out_x_dim + filter_x_dim ) % input_dilation_size.x ) == 0 );
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                            //std::cout << sum << " += " << ( in_bound ? input_layer[ in_index ] : 0 ) << " * " << filter[ filter_index ] << std::endl;
                                        }
                                    }
                                }
                                output_layer[ out_index ] = sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_transposed_circularpadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 input_dilation_size( this->input_dilation.width(), this->input_dilation.height(), this->input_dilation.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;

                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                V sum = 0;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = ( out_z_dim/input_dilation_size.z ) * stride_size.z - padding_size.z + filter_z_dim * dilation_size.z;
                                    in_z_dim = ( ( in_z_dim + input_size.z ) % input_size.z );
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound && ( ( ( out_z_dim + filter_z_dim ) % input_dilation_size.z ) == 0 );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = ( out_y_dim/input_dilation_size.y ) * stride_size.y - padding_size.y + filter_y_dim * dilation_size.y;
                                        in_y_dim = ( ( in_y_dim + input_size.y ) % input_size.y );
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound && ( ( ( out_y_dim + filter_y_dim ) % input_dilation_size.y ) == 0 );
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = ( out_x_dim/input_dilation_size.x ) * stride_size.x - padding_size.x + filter_x_dim * dilation_size.x;
                                            in_x_dim = ( ( in_x_dim + input_size.x ) % input_size.x );
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound && ( ( ( out_x_dim + filter_x_dim ) % input_dilation_size.x ) == 0 );
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                            //std::cout << sum << " += " << ( in_bound ? input_layer[ in_index ] : 0 ) << " * " << filter[ filter_index ] << std::endl;
                                        }
                                    }
                                }
                                output_layer[ out_index ] = sum;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void ConvolutionLayer<T>::inference_transposed_nonpadded( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            const util::Shape& output_shape = output_layer.get_shape();
            const util::Shape& input_shape = input_layer.get_shape();
            const util::Shape& filter_shape = filter.get_shape();

            const util::Dim5 input_size( input_shape.width(), input_shape.height(), input_shape.depth(), input_shape.channels(), input_shape.batches() );
            const util::Dim5 output_size( output_shape.width(), output_shape.height(), output_shape.depth(), output_shape.channels(), output_shape.batches() );
            const util::Dim5 filter_size( filter_shape.width(), filter_shape.height(), filter_shape.depth(), filter_shape.channels(), 1 );

            const util::Dim3 input_dilation_size( this->input_dilation.width(), this->input_dilation.height(), this->input_dilation.depth() );
            const util::Dim3 padding_size( this->padding.width(), this->padding.height(), this->padding.depth() );
            const util::Dim3 stride_size( this->stride.width(), this->stride.height(), this->stride.depth() );
            const util::Dim3 dilation_size( this->dilation.width(), this->dilation.height(), this->dilation.depth() );

            for ( util::Dim out_b_dim = 0; out_b_dim < output_size.b; out_b_dim++ )
            {
                //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                util::Index out_b_idx = out_b_dim * output_size.c;
                util::Index in_b_idx = out_b_dim * input_size.c;

                //std::cout << "[ " << "Batch:" << ( int )out_b_dim << " ]" << std::endl;

                for ( util::Dim out_c_dim = 0; out_c_dim < output_size.c; out_c_dim++ )
                {
                    //bool out_c_bound = ( out_c_dim >= 0 ) && ( out_c_dim < output_size.c );
                    util::Index out_c_idx = ( out_b_idx + out_c_dim ) * output_size.z;

                    //std::cout << "\t" << "[ " << "Output-C:" << ( int )out_c_dim << " ]" << std::endl;

                    for ( util::Dim out_z_dim = 0; out_z_dim < output_size.z; out_z_dim++ )
                    {
                        //bool out_z_bound = ( out_z_dim >= 0 ) && ( out_z_dim < output_size.z );
                        util::Index out_z_idx = ( out_c_idx + out_z_dim ) * output_size.y;

                        //std::cout << "\t\t" << "[ " << "Output-Z:" << ( int )out_z_dim << " ]" << std::endl;

                        for ( util::Dim out_y_dim = 0; out_y_dim < output_size.y; out_y_dim++ )
                        {
                            //bool out_y_bound = ( out_y_dim >= 0 ) && ( out_y_dim < output_size.y );
                            util::Index out_y_idx = ( out_z_idx + out_y_dim ) * output_size.x;

                            //std::cout << "\t\t\t" << "[ " << "Output-Y:" << ( int )out_y_dim << " ]" << std::endl;

                            for ( util::Dim out_x_dim = 0; out_x_dim < output_size.x; out_x_dim++ )
                            {
                                //bool out_x_bound = ( out_x_dim >= 0 ) && ( out_x_dim < output_size.x );
                                util::Index out_index = out_y_idx + out_x_dim;

                                util::Dim filter_c_dim = out_c_dim;
                                util::Index filter_c_idx = filter_c_dim * filter_size.z;

                                util::Dim in_b_dim = out_b_dim;
                                util::Dim in_c_dim = out_c_dim % input_size.c;

                                V sum = 0;

                                bool in_c_bound = ( in_c_dim >= 0 ) && ( in_c_dim < input_size.c );
                                util::Index in_c_idx = ( in_b_idx + in_c_dim ) * input_size.z;

                                //std::cout << "\t\t\t\t\t" << "[ " << "Filter-C:" << ( int )filter_c_dim << " : " << "Input-C:" << ( int )in_c_dim << " ]" << std::endl;

                                //output_layer.print();

                                for ( util::Dim filter_z_dim = 0; filter_z_dim < filter_size.z; filter_z_dim++ )
                                {
                                    //bool filter_z_bound = ( filter_z_dim >= 0 ) && ( filter_z_dim < filter_size.z );
                                    util::Index filter_z_idx = ( filter_c_idx + filter_z_dim ) * filter_size.y;

                                    util::Dim in_z_dim = ( out_z_dim/input_dilation_size.z ) * stride_size.z + filter_z_dim * dilation_size.z;
                                    bool in_z_bound = ( in_z_dim >= 0 ) && ( in_z_dim < input_size.z ) && in_c_bound && ( ( ( out_z_dim + filter_z_dim ) % input_dilation_size.z ) == 0 );
                                    util::Index in_z_idx = ( in_c_idx + in_z_dim ) * input_size.y;

                                    //std::cout << "\t\t\t\t\t\t" << "[ " << "Filter-Z:" << ( int )filter_z_dim << " : " << "Input-Z:" << ( int )in_z_dim << " ]" << std::endl;

                                    for ( util::Dim filter_y_dim = 0; filter_y_dim < filter_size.y; filter_y_dim++ )
                                    {
                                        //bool filter_y_bound = ( filter_y_dim >= 0 ) && ( filter_y_dim < filter_size.y );
                                        util::Index filter_y_idx = ( filter_z_idx + filter_y_dim ) * filter_size.x;

                                        util::Dim in_y_dim = ( out_y_dim/input_dilation_size.y ) * stride_size.y + filter_y_dim * dilation_size.y;
                                        bool in_y_bound = ( in_y_dim >= 0 ) && ( in_y_dim < input_size.y ) && in_z_bound && ( ( ( out_y_dim + filter_y_dim ) % input_dilation_size.y ) == 0 );
                                        util::Index in_y_idx = ( in_z_idx + in_y_dim ) * input_size.x;

                                        //std::cout << "\t\t\t\t\t\t\t" << "[ " << "Filter-Y:" << ( int )filter_y_dim << " : " << "Input-Y:" << ( int )in_y_dim << " ]" << std::endl;

                                        for ( util::Dim filter_x_dim = 0; filter_x_dim < filter_size.x; filter_x_dim++ )
                                        {
                                            //bool filter_x_bound = ( filter_x_dim >= 0 ) && ( filter_x_dim < filter_size.x );
                                            util::Index filter_index = filter_y_idx + filter_x_dim;

                                            util::Dim in_x_dim = ( out_x_dim/input_dilation_size.x ) * stride_size.x + filter_x_dim * dilation_size.x;
                                            bool in_bound = ( in_x_dim >= 0 ) && ( in_x_dim < input_size.x ) && in_y_bound && ( ( ( out_x_dim + filter_x_dim ) % input_dilation_size.x ) == 0 );
                                            util::Index in_index = in_y_idx + in_x_dim;

                                            //std::cout << "\t\t\t\t\t\t\t\t" << "[ " << "Filter-X:" << ( int )filter_x_dim << " : " << "Input-X:" << ( int )in_x_dim << " : " << "Filter-Index:" << ( int )filter_index << " : " << "Input-Index:" << ( int )in_index << " ]" << std::endl;

                                            sum += filter[ filter_index ] * ( in_bound ? input_layer[ in_index ] : 0 );
                                            //std::cout << "\t\t\t\t\t\t\t\t\t" << sum << " += " << input_layer[ in_index ] << " * " << filter[ filter_index ] << std::endl;
                                        }
                                    }
                                }
                                output_layer[ out_index ] = sum;
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

    for (const auto& t : types)
    {
        std::string code =
R"(
    template class ConvolutionLayer<{t}>;)";
        // Replace placeholders in the code with actual types
        size_t pos;
        while ((pos = code.find("{t}")) != std::string::npos)
            code.replace(pos, 3, t);

        file << code;
    }

    file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            for (const auto& v : types)
            {
                std::string code =
R"(
    template void ConvolutionLayer<{t}>::inference( const NodeLayer<{u}>& input_layer, NodeLayer<{v}>& output_layer ) const;)";
                // Replace placeholders in the code with actual types
                size_t pos;
                while ((pos = code.find("{t}")) != std::string::npos)
                    code.replace(pos, 3, t);
                while ((pos = code.find("{u}")) != std::string::npos)
                    code.replace(pos, 3, u);
                while ((pos = code.find("{v}")) != std::string::npos)
                    code.replace(pos, 3, v);

                file << code;
            }
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            for (const auto& v : types)
            {
                std::string code =
R"(
    template void ConvolutionLayer<{t}>::backpropagation( NodeLayer<{u}>& input_layer, const NodeLayer<{v}>& output_layer );)";
                // Replace placeholders in the code with actual types
                size_t pos;
                while ((pos = code.find("{t}")) != std::string::npos)
                    code.replace(pos, 3, t);
                while ((pos = code.find("{u}")) != std::string::npos)
                    code.replace(pos, 3, u);
                while ((pos = code.find("{v}")) != std::string::npos)
                    code.replace(pos, 3, v);

                file << code;
            }
        }
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}