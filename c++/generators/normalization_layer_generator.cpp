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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/normalization_layer/normalization_layer.cpp");
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
#include "c++/bewusstsein_neural_networks/source/layers/normalization_layer/normalization_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        NormalizationLayer<T>::NormalizationLayer() : BaseLayer( LayerType::normalization_layer, "" ) {}

        template <typename T>
        NormalizationLayer<T>::NormalizationLayer( const std::string& name, const NormalizationType type, const util::Shape& layer_shape ):
            BaseLayer( LayerType::normalization_layer, name ), norm_type( type )
        {
            switch ( type )
            {
                case NormalizationType::batch_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ] ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape[ 3 ] ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape[ 3 ] ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ] ) );
                    break;
                }
                case NormalizationType::layer_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, 1, layer_shape[ 4 ] ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, 1, layer_shape[ 4 ] ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, 1, layer_shape[ 4 ] ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, 1, layer_shape[ 4 ] ) );
                    break;
                }
                case NormalizationType::instance_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ], layer_shape[ 4 ] ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape[ 3 ], layer_shape[ 4 ] ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape[ 3 ], layer_shape[ 4 ] ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ], layer_shape[ 4 ] ) );
                    break;
                }
                case NormalizationType::group_norm:
                {
                    util::Dim group_size = 1;
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ] / group_size, layer_shape[ 4 ] ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape[ 3 ] / group_size, layer_shape[ 4 ] ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape[ 3 ] / group_size, layer_shape[ 4 ] ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape[ 3 ] / group_size, layer_shape[ 4 ] ) );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
            }
        }
        */

        template <typename T>
        NormalizationLayer<T>::NormalizationLayer( const NormalizationType type, const util::Shape& layer_shape, const TrainingMode training_mode ):
            BaseLayer( LayerType::normalization_layer ), norm_type( type )
        {
            switch ( type )
            {
                case NormalizationType::batch_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape.channels() ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape.channels() ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape.channels() ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape.channels() ) );
                    break;
                }
                case NormalizationType::layer_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, 1, layer_shape.batches() ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, 1, layer_shape.batches() ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, 1, layer_shape.batches() ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, 1, layer_shape.batches() ) );
                    break;
                }
                case NormalizationType::instance_norm:
                {
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape.channels(), layer_shape.batches() ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape.channels(), layer_shape.batches() ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape.channels(), layer_shape.batches() ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape.channels(), layer_shape.batches() ) );
                    break;
                }
                case NormalizationType::group_norm:
                {
                    util::Dim group_size = 1;
                    this->mean.resize(      util::Shape( 1, 1, 1, layer_shape.channels() / group_size, layer_shape.batches() ) );
                    this->variance.resize(  util::Shape( 1, 1, 1, layer_shape.channels() / group_size, layer_shape.batches() ) );
                    this->gamma.resize(     util::Shape( 1, 1, 1, layer_shape.channels() / group_size, layer_shape.batches() ), 1 );
                    this->beta.resize(      util::Shape( 1, 1, 1, layer_shape.channels() / group_size, layer_shape.batches() ) );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
            }

            this->allocate_training_memory( training_mode );
        }

    //: Destructors
        template <typename T>
        NormalizationLayer<T>::~NormalizationLayer() {}

    //: Methods
        template <typename T>
        NormalizationType NormalizationLayer<T>::get_type() const
        {
            return this->norm_type;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_mean() const
        {
            return this->mean;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_variance() const
        {
            return this->variance;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_beta() const
        {
            return this->beta;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_gamma() const
        {
            return this->gamma;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_beta_jacobian() const
        {
            return this->beta_jacobian;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_gamma_jacobian() const
        {
            return this->gamma_jacobian;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_beta_momentum() const
        {
            return this->beta_momentum;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_gamma_momentum() const
        {
            return this->gamma_momentum;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_beta_velocity() const
        {
            return this->beta_velocity;
        }

        template <typename T>
        const util::Tensor<T>& NormalizationLayer<T>::get_gamma_velocity() const
        {
            return this->gamma_velocity;
        }

        template <typename T>
        void NormalizationLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape beta_shape = this->beta.get_shape();
                    util::Shape gamma_shape = this->gamma.get_shape();
                    this->beta_jacobian.resize( beta_shape );
                    this->gamma_jacobian.resize( gamma_shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape beta_shape = this->beta.get_shape();
                    util::Shape gamma_shape = this->gamma.get_shape();
                    this->beta_jacobian.resize( beta_shape );
                    this->gamma_jacobian.resize( gamma_shape );
                    this->beta_momentum.resize( beta_shape );
                    this->gamma_momentum.resize( gamma_shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape beta_shape = this->beta.get_shape();
                    util::Shape gamma_shape = this->gamma.get_shape();
                    this->beta_jacobian.resize( beta_shape );
                    this->gamma_jacobian.resize( gamma_shape );
                    this->beta_momentum.resize( beta_shape );
                    this->gamma_momentum.resize( gamma_shape );
                    this->beta_velocity.resize( beta_shape );
                    this->gamma_velocity.resize( gamma_shape );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Invalid training mode selection" );
                }
            }
        }

        template <typename T>
        void NormalizationLayer<T>::print() const
        {
            std::cout << "NormalizationLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Mean:" << std::endl;
            this->mean.print();
            std::cout << "Variance:" << std::endl;
            this->variance.print();
            std::cout << "Gamma:" << std::endl;
            this->gamma.print();
            std::cout << "Beta:" << std::endl;
            beta.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void NormalizationLayer<T>::print_jacobian() const
        {
            std::cout << "NormalizationLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Gamma.jacobian" << std::endl;
            this->gamma_jacobian.print();
            std::cout << "Beta.jacobian" << std::endl;
            this->beta_jacobian.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void NormalizationLayer<T>::save_model( const std::string& file_name ) const
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
        void NormalizationLayer<T>::save_state( const std::string& file_name ) const
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
        void NormalizationLayer<T>::load_model( const std::string& file_name )
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
        void NormalizationLayer<T>::load_state( const std::string& file_name )
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
        void NormalizationLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            file.write( reinterpret_cast<const char*>( &this->norm_type ), sizeof( NormalizationType ) );

            this->mean.save( file );
            this->variance.save( file );
            this->gamma.save( file );
            this->beta.save( file );
        }

        template <typename T>
        void NormalizationLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            file.write( reinterpret_cast<const char*>( &this->norm_type ), sizeof( NormalizationType ) );

            this->mean.save( file );
            this->variance.save( file );
            this->gamma.save( file );
            this->gamma_jacobian.save( file );
            this->gamma_momentum.save( file );
            this->gamma_velocity.save( file );
            this->beta.save( file );
            this->beta_jacobian.save( file );
            this->beta_momentum.save( file );
            this->beta_velocity.save( file );
        }

        template <typename T>
        void NormalizationLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            file.read( reinterpret_cast<char*>( this->norm_type ), sizeof( NormalizationType ) );

            this->mean.load( file );
            this->variance.load( file );
            this->gamma.load( file );
            this->beta.load( file );
        }

        template <typename T>
        void NormalizationLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            file.read( reinterpret_cast<char*>( this->norm_type ), sizeof( NormalizationType ) );

            this->mean.load( file );
            this->variance.load( file );
            this->gamma.load( file );
            this->gamma_jacobian.load( file );
            this->gamma_momentum.load( file );
            this->gamma_velocity.load( file );
            this->beta.load( file );
            this->beta_jacobian.load( file );
            this->beta_momentum.load( file );
            this->beta_velocity.load( file );
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::stat_analysis( const NodeLayer<U>& layer )
        {
            switch ( this->norm_type )
            {
                case NormalizationType::batch_norm:
                {
                    this->stat_analysis_batch( layer );
                    break;
                }
                case NormalizationType::layer_norm:
                {
                    this->stat_analysis_layer( layer );
                    break;
                }
                case NormalizationType::instance_norm:
                {
                    this->stat_analysis_instance( layer );
                    break;
                }
                case NormalizationType::group_norm:
                {
                    this->stat_analysis_group( layer );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::inference( NodeLayer<U>& layer ) const
        {
            switch ( this->norm_type )
            {
                case NormalizationType::batch_norm:
                {
                    this->inference_batch( layer );
                    break;
                }
                case NormalizationType::layer_norm:
                {
                    this->inference_layer( layer );
                    break;
                }
                case NormalizationType::instance_norm:
                {
                    this->inference_instance( layer );
                    break;
                }
                case NormalizationType::group_norm:
                {
                    this->inference_group( layer );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::backpropagation( NodeLayer<U>& layer )
        {
            switch ( this->norm_type )
            {
                case NormalizationType::batch_norm:
                {
                    this->backpropagation_batch( layer );
                    break;
                }
                case NormalizationType::layer_norm:
                {
                    this->backpropagation_layer( layer );
                    break;
                }
                case NormalizationType::instance_norm:
                {
                    this->backpropagation_instance( layer );
                    break;
                }
                case NormalizationType::group_norm:
                {
                    this->backpropagation_group( layer );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Normalization Type is invalid/uninitialized" );
                    break;
                }
            }
        }

        template <typename T>
        void NormalizationLayer<T>::gradient_decent( const util::Dim batch_size, const StepSize step_size )
        {
            util::Size gamma_size = this->gamma.get_size();
            util::Size beta_size = this->beta.get_size();

            for ( util::Index index = 0; index < gamma_size; ++index )
            {
                T gradient = this->gamma_jacobian[ index ] / batch_size;

                this->gamma[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->gamma[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->gamma_jacobian.zero();

            for ( util::Index index = 0; index < beta_size; ++index )
            {
                T gradient = this->beta_jacobian[ index ] / batch_size;

                this->beta[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->beta[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->beta_jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void NormalizationLayer<T>::gradient_decent_momentum( const util::Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            util::Size gamma_size = this->gamma.get_size();
            util::Size beta_size = this->beta.get_size();

            for ( util::Index index = 0; index < gamma_size; ++index )
            {
                T gradient = this->gamma_jacobian[ index ] / batch_size;
                T momentum_value = this->gamma_momentum[ index ];

                this->gamma_momentum[ index ] = gradient;

                this->gamma[ index ] += step_size * ( ( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->gamma[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->gamma_jacobian.zero();

            for ( util::Index index = 0; index < beta_size; ++index )
            {
                T gradient = this->beta_jacobian[ index ] / batch_size;
                T momentum_value = this->beta_momentum[ index ];

                this->beta_momentum[ index ] = gradient;

                this->beta[ index ] += step_size * ( ( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->beta[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->beta_jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void NormalizationLayer<T>::gradient_decent_adam( const util::Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            uint32_t step = this->time_step.get_count();
            T beta1_mp = 1 - pow( beta1, step );
            T beta2_mp = 1 - pow( beta2, step );
            T beta1_m = 1 - beta1;
            T beta2_m = 1 - beta2;

            util::Size gamma_size = this->gamma.get_size();
            util::Size beta_size = this->beta.get_size();

            for ( util::Index index = 0; index < gamma_size; ++index )
            {
                T gradient = this->gamma_jacobian[ index ] / batch_size;

                // Update momentum
                T momentum_value = this->gamma_momentum[ index ];
                momentum_value = ( beta1 * momentum_value ) + ( beta1_m * gradient );
                this->gamma_momentum[ index ] = momentum_value;

                // Update velocity
                T velocity_value = this->gamma_velocity[ index ];
                velocity_value = ( beta2 * velocity_value ) + ( beta2_m * ( gradient * gradient ) );
                this->gamma_velocity[ index ] = velocity_value;

                // Compute bias-corrected momentum and velocity
                T momentum_hat = momentum_value / beta1_mp;
                T velocity_hat = velocity_value / beta2_mp;

                // Update weight
                this->gamma[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->gamma[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->gamma_jacobian.zero();

            for ( util::Index index = 0; index < beta_size; ++index )
            {
                T gradient = this->beta_jacobian[ index ] / batch_size;

                // Update momentum
                T momentum_value = this->beta_momentum[ index ];
                momentum_value = ( beta1 * momentum_value ) + ( beta1_m * gradient );
                this->beta_momentum[ index ] = momentum_value;

                // Update velocity
                T velocity_value = this->beta_velocity[ index ];
                velocity_value = ( beta2 * velocity_value ) + ( beta2_m * ( gradient * gradient ) );
                this->beta_velocity[ index ] = velocity_value;

                // Compute bias-corrected momentum and velocity
                T momentum_hat = momentum_value / beta1_mp;
                T velocity_hat = velocity_value / beta2_mp;

                // Update weight
                this->beta[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->beta[ index ] ) ) {throw std::invalid_argument( "Value is NaN" );}
                #endif
            }

            this->beta_jacobian.zero();

            this->time_step.tick();
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::stat_analysis_batch( const NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            util::Size layer_norm_size = layer_spacial_size * layer_b_size;

            for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
            {
                T mean = 0;
                for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    util::Index batch_idx = batch * layer_c_size;
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        mean += layer[ index ];
                    }
                }

                T variance = 0;
                for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    util::Index batch_idx = batch * layer_c_size;
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T diff = layer[ index ] - mean;
                        variance += diff * diff;
                    }
                }

                uint32_t step = this->time_step.get_count();
                float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_norm_size ) ) / step;
                float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_norm_size ) ) / step;

                this->mean[ channel ]       = this->mean[ channel ]     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                this->variance[ channel ]   = this->variance[ channel ] * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::stat_analysis_layer( const NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            util::Size layer_norm_size = layer_spacial_size * layer_c_size;

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = 0;
                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        mean += layer[ index ];
                    }
                }

                T variance = 0;
                for ( util::Dim channel = 0; channel < layer_shape[ 3 ]; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T diff = layer[ index ] - mean;
                        variance += diff * diff;
                    }
                }

                uint32_t step = this->time_step.get_count();
                float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_norm_size ) ) / step;
                float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_norm_size ) ) / step;

                this->mean[ batch ]       = this->mean[ batch ]     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                this->variance[ batch ]   = this->variance[ batch ] * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::stat_analysis_instance( const NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    T mean = 0;
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        mean += layer[ index ];
                    }

                    T variance = 0;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T diff = layer[ index ] - mean;
                        variance += diff * diff;
                    }

                    uint32_t step = this->time_step.get_count();
                    float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                    float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_spacial_size ) ) / step;
                    float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_spacial_size ) ) / step;

                    this->mean[ channel_idx ]       = this->mean[ channel_idx ]     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                    this->variance[ channel_idx ]   = this->variance[ channel_idx ] * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::stat_analysis_group( const NodeLayer<U>& layer )
        {
            // TO DO
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::inference_batch( NodeLayer<U>& layer ) const
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
            {
                /*--------------------------------------------------*/
                //std::cout << "[ " << "Channel:" << ( int )channel << " ]" << std::endl;
                /*--------------------------------------------------*/

                T mean = this->mean[ channel ];
                T variance = this->variance[ channel ];
                T inv_standard_deviation = 1.0 / sqrt( variance + 1e-7 );
                T beta = this->beta[ channel ];
                T gamma = this->gamma[ channel ];

                for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    /*--------------------------------------------------*/
                    //std::cout << "\t" << "[ " << "Batch:" << ( int )batch << " ]" << std::endl;
                    /*--------------------------------------------------*/

                    util::Index batch_idx = batch * layer_c_size;
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        /*--------------------------------------------------*/
                        //std::cout << "\t\t" << "[ " << "Spacial:" << ( int )spacial << " : " << "Spacial-Index:" << ( int )index << " ]" << std::endl;
                        /*--------------------------------------------------*/

                        T normalized = ( layer[ index ] - mean ) * inv_standard_deviation;
                        layer[ index ] = ( normalized * gamma ) + beta;

                        /*--------------------------------------------------*/
                        //std::cout << "\t\t\t" << "-" << normalized << " = ( " << layer[ index ] << " - " << mean << " ) * " << inv_sqrt_variance << std::endl;
                        //std::cout << "\t\t\t" << "-" << layer[ index ] << " = ( " << normalized << " * " << gamma << " ) + " << beta << std::endl;
                        /*--------------------------------------------------*/
                    }
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::inference_layer( NodeLayer<U>& layer ) const
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = this->mean[ batch ];
                T variance = this->variance[ batch ];
                T inv_standard_deviation = 1.0 / sqrt( variance + 1e-7 );
                T beta = this->beta[ batch ];
                T gamma = this->gamma[ batch ];

                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T normalized = ( layer[ index ] - mean ) * inv_standard_deviation;
                        layer[ index ] = ( normalized * gamma ) + beta;
                    }
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::inference_instance( NodeLayer<U>& layer ) const
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;

                    T mean = this->mean[ channel_idx ];
                    T variance = this->variance[ channel_idx ];
                    T inv_standard_deviation = 1.0 / sqrt( variance + 1e-7 );
                    T beta = this->beta[ channel_idx ];
                    T gamma = this->gamma[ channel_idx ];

                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T normalized = ( layer[ index ] - mean ) * inv_standard_deviation;
                        layer[ index ] = ( normalized * gamma ) + beta;
                    }
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::inference_group( NodeLayer<U>& layer ) const
        {
            // TODO( Shenmarukai ):
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::backpropagation_batch( NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            util::Size layer_norm_size = layer_spacial_size * layer_b_size;

            for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
            {
                T mean = this->mean[ channel ];
                T variance = this->variance[ channel ];
                T cube_root_variance = pow( variance + 1e-7, -1.5 );
                T inv_sqrt_variance = 1.0 / sqrt( variance + 1e-7 );
                T beta = this->beta[ channel ];
                T gamma = this->gamma[ channel ];

                T beta_gradient = 0;
                T gamma_gradient = 0;

                for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    util::Index batch_idx = batch * layer_c_size;
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T delta_node = layer.get_delta( index );

                        T node_minus_mean = layer[ index ] - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = -0.5 * dnormalized * normalized * cube_root_variance;
                        T dmean = -dnormalized * inv_sqrt_variance - 2.0 * dvariance * node_minus_mean / layer_norm_size;

                        // Calculate gradient for beta
                        beta_gradient += delta_node;

                        // Calculate gradient for gamma
                        gamma_gradient += delta_node * normalized;

                        // Backpropagate through scaling and offset
                        delta_node = dnormalized * inv_sqrt_variance + dvariance * 2.0 * node_minus_mean / layer_norm_size + dmean / layer_norm_size;
                        layer.get_delta( index ) = delta_node;
                    }
                }

                this->beta_jacobian[ channel ] += beta_gradient / layer_norm_size;
                this->gamma_jacobian[ channel ] += gamma_gradient / layer_norm_size;
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::backpropagation_layer( NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            util::Size layer_norm_size = layer_spacial_size * layer_c_size;

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = this->mean[ batch ];
                T variance = this->variance[ batch ];
                T cube_root_variance = pow( variance + 1e-7, -1.5 );
                T inv_sqrt_variance = 1.0 / sqrt( variance + 1e-7 );
                T beta = this->beta[ batch ];
                T gamma = this->gamma[ batch ];

                T beta_gradient = 0;
                T gamma_gradient = 0;

                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T delta_node = layer.get_delta( index );

                        T node_minus_mean = layer[ index ] - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = -0.5 * dnormalized * normalized * cube_root_variance;
                        T dmean = -dnormalized * inv_sqrt_variance - 2.0 * dvariance * node_minus_mean / layer_norm_size;

                        // Calculate gradient for beta
                        beta_gradient += delta_node;

                        // Calculate gradient for gamma
                        gamma_gradient += delta_node * normalized;

                        // Backpropagate through scaling and offset
                        delta_node = dnormalized * inv_sqrt_variance + dvariance * 2.0 * node_minus_mean / layer_norm_size + dmean / layer_norm_size;
                        layer.get_delta( index ) = delta_node;
                    }
                }

                this->beta_jacobian[ batch ] += beta_gradient / layer_norm_size;
                this->gamma_jacobian[ batch ] += gamma_gradient / layer_norm_size;
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::backpropagation_instance( NodeLayer<U>& layer )
        {
            util::Shape layer_shape = layer.get_shape();
            util::Size layer_spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 2, 1, std::multiplies<int>() );

            util::Dim layer_b_size = layer_shape.batches();
            util::Dim layer_c_size = layer_shape.channels();

            for ( util::Dim batch = 0; batch < layer_b_size; batch++ )
            {
                util::Index batch_idx = batch * layer_c_size;
                for ( util::Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    util::Index channel_idx = ( batch_idx + channel ) * layer_spacial_size;

                    T mean = this->mean[ channel_idx ];
                    T variance = this->variance[ channel_idx ];
                    T cube_root_variance = pow( variance + 1e-7, -1.5 );
                    T inv_sqrt_variance = 1.0 / sqrt( variance + 1e-7 );
                    T beta = this->beta[ channel_idx ];
                    T gamma = this->gamma[ channel_idx ];

                    T beta_gradient = 0;
                    T gamma_gradient = 0;

                    for ( util::Index spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        util::Index index = channel_idx + spacial;

                        T delta_node = layer.get_delta( index );

                        T node_minus_mean = layer[ index ] - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = -0.5 * dnormalized * normalized * cube_root_variance;
                        T dmean = -dnormalized * inv_sqrt_variance - 2.0 * dvariance * node_minus_mean / layer_spacial_size;

                        // Calculate gradient for beta
                        beta_gradient += delta_node;

                        // Calculate gradient for gamma
                        gamma_gradient += delta_node * normalized;

                        // Backpropagate through scaling and offset
                        delta_node = dnormalized * inv_sqrt_variance + dvariance * 2.0 * node_minus_mean / layer_spacial_size + dmean / layer_spacial_size;
                        layer.get_delta( index ) = delta_node;
                    }

                    this->beta_jacobian[ channel_idx ] += beta_gradient / layer_spacial_size;
                    this->gamma_jacobian[ channel_idx ] += gamma_gradient / layer_spacial_size;
                }
            }
        }

        template <typename T>
        template <typename U>
        void NormalizationLayer<T>::backpropagation_group( NodeLayer<U>& layer )
        {
            // TODO( Shenmarukai ):
        }
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& t : types)
    {
        std::string code =
R"(
    template class NormalizationLayer<{t}>;)";
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
            std::string code =
R"(
    template void NormalizationLayer<{t}>::stat_analysis( const NodeLayer<{u}>& layer );)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{t}")) != std::string::npos)
                code.replace(pos, 3, t);
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);

            file << code;
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            std::string code =
R"(
    template void NormalizationLayer<{t}>::inference( NodeLayer<{u}>& layer ) const;)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{t}")) != std::string::npos)
                code.replace(pos, 3, t);
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);

            file << code;
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            std::string code =
R"(
    template void NormalizationLayer<{t}>::backpropagation( NodeLayer<{u}>& layer );)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{t}")) != std::string::npos)
                code.replace(pos, 3, t);
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);

            file << code;
        }
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}