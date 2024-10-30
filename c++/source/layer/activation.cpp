// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Macros
#include "bewusstsein_neural_networks/c++/macro/for.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/math.hpp"
#include "bewusstsein_neural_networks/c++/include/core/random.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"
#include "bewusstsein_neural_networks/c++/include/core/activation.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/activation.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer() requires ( is_type<T, none>::value && ( M == ParameterMode::none ) ) :
            BaseLayer( LayerType::activation_layer ) {}

        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer( const T parameter ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) ) :
            BaseLayer( LayerType::activation_layer ), ActivationParameters<T, A, M>( parameter ) {}

        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer( const u8 channels, const T parameter ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) ) :
            BaseLayer( LayerType::activation_layer ), ActivationParameters<T, A, M>( Shape<4>( 1, 1, 1, channels ), parameter ) {}

        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer( const u8 channels, const std::vector<T>& parameters ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) ) :
            BaseLayer( LayerType::activation_layer ), ActivationParameters<T, A, M>( Shape<4>( 1, 1, 1, channels ), parameters ) {}

        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer( const Shape<4>& shape, const T parameter ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) ) :
            BaseLayer( LayerType::activation_layer ), ActivationParameters<T, A, M>( shape, parameter ) {}

        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::ActivationLayer( const Shape<4>& shape, const std::vector<T>& parameters ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) ) :
            BaseLayer( LayerType::activation_layer ), ActivationParameters<T, A, M>( shape, parameters ) {}

    //: Destructors
        template <typename T, ActivationType A, ParameterMode M>
        ActivationLayer<T, A, M>::~ActivationLayer() {}

    //: Methods
        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::set_training_mode( const TrainingMode training_mode ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( this->parameters, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ActivationType A, ParameterMode M>
        const Shape<4>& ActivationLayer<T, A, M>::get_shape() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters.get_shape();
        }

        template <typename T, ActivationType A, ParameterMode M>
        Size ActivationLayer<T, A, M>::get_size() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters.get_size();
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::set_parameter( const T value ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->parameter = value;
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::set_jacobian( const T value ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->jacobian = value;
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::set_momentum( const T value ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->momentum = value;
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::set_velocity( const T value ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->velocity = value;
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::fill_parameters( const T value ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->parameters.fill( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::fill_jacobian( const T value ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->jacobian.fill( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::fill_momentum( const T value ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->momentum.fill( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::fill_velocity( const T value ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->velocity.fill( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_parameter() requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->parameter = T( 0 );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_jacobian() requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->jacobian = T( 0 );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_momentum() requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->momentum = T( 0 );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_velocity() requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->velocity = T( 0 );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_parameters() requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->parameters.zero();
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_jacobian() requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->jacobian.zero();
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_momentum() requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->momentum.zero();
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::zero_velocity() requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->velocity.zero();
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_parameter( const T min, const T max ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->parameter = random_value( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_jacobian( const T min, const T max ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->jacobian = random_value( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_momentum( const T min, const T max ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->momentum = random_value( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_velocity( const T min, const T max ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            this->velocity = random_value( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_parameters( const T min, const T max ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->parameters.randomize( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_jacobian( const T min, const T max ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->jacobian.randomize( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_momentum( const T min, const T max ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->momentum.randomize( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::randomize_velocity( const T min, const T max ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            this->velocity.randomize( min, max );
        }

        template <typename T, ActivationType A, ParameterMode M>
        T ActivationLayer<T, A, M>::get_parameter() const requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            return this->parameter;
        }

        template <typename T, ActivationType A, ParameterMode M>
        T ActivationLayer<T, A, M>::get_jacobian() const requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            return this->jacobian;
        }

        template <typename T, ActivationType A, ParameterMode M>
        T ActivationLayer<T, A, M>::get_momentum() const requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            return this->momentum;
        }

        template <typename T, ActivationType A, ParameterMode M>
        T ActivationLayer<T, A, M>::get_velocity() const requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            return this->velocity;
        }

        template <typename T, ActivationType A, ParameterMode M>
        const Tensor<T, 4>& ActivationLayer<T, A, M>::get_parameters() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters;
        }

        template <typename T, ActivationType A, ParameterMode M>
        const Tensor<T, 4>& ActivationLayer<T, A, M>::get_jacobian() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->jacobian;
        }

        template <typename T, ActivationType A, ParameterMode M>
        const Tensor<T, 4>& ActivationLayer<T, A, M>::get_momentum() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->momentum;
        }

        template <typename T, ActivationType A, ParameterMode M>
        const Tensor<T, 4>& ActivationLayer<T, A, M>::get_velocity() const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->velocity;
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::reshape( const u8 channels ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) )
        {
            this->BaseLayer::reshape( Shape<4>( 1, 1, 1, channels ), this->parameters, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::reshape( const Shape<4>& shape ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) )
        {
            this->BaseLayer::reshape( shape, this->parameters, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::resize( const u8 channels ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) )
        {
            this->BaseLayer::resize( Shape<4>( 1, 1, 1, channels ), this->parameters, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ActivationType A, ParameterMode M>
        void ActivationLayer<T, A, M>::resize( const Shape<4>& shape ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) )
        {
            this->BaseLayer::resize( shape, this->parameters, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::inference( NodeLayer<U>& layer ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) )
        {
            Size layer_size = layer.get_shape().volume();

            for ( Idx i = 0; i < layer_size; ++i )
            {
                layer.get_node( i ) = this->activation( layer.get_node( i ) );
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::inference( NodeLayer<U>& layer ) const requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            Size layer_size = layer.get_shape().volume();

            for ( Idx i = 0; i < layer_size; ++i )
            {
                layer.get_node( i ) = this->activation( this->parameter, layer.get_node( i ) );
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::inference( NodeLayer<U>& layer ) const requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) )
        {
            const Shape<5>& layer_shape       = layer.get_shape();
            const Shape<4>& parameters_shape  = this->parameters.get_shape();

            Size layer_spacial_size     = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != parameters_shape.channels() ) { return Error::MISMATCHED_CHANNELS; }

            for ( Dim b_dim = 0; b_dim < layer_shape.batches(); ++b_dim )
            {
                Idx b_idx = b_dim * layer_shape.channels();
                for ( Dim c_dim = 0; c_dim < layer_shape.channels(); ++c_dim )
                {
                    Idx c_idx = ( b_idx + c_dim ) * layer_spacial_size;

                    const U& parameter = this->get_parameter( c_dim );

                    for ( Idx spacial_idx = 0; spacial_idx < layer_spacial_size; ++spacial_idx )
                    {
                        Idx index = c_idx + spacial_idx;

                        layer.get_node( index ) = this->activation( parameter, layer.get_node( index ) );
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::inference( NodeLayer<U>& layer ) const requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) )
        {
            const Shape<5>& layer_shape       = layer.get_shape();
            const Shape<4>& parameters_shape  = this->parameters.get_shape();

            Size spacial_size = parameters_shape.distance( 0, 4 );

            if ( layer_shape.distance( 0, 4 ) != spacial_size ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim batch_dim = 0; batch_dim < layer_shape.batches(); ++batch_dim )
            {
                Idx batch_idx = batch_dim * spacial_size;
                for ( Idx spacial_idx = 0; spacial_idx < spacial_size; ++spacial_idx )
                {
                    Idx layer_idx = batch_idx + spacial_idx;
                    layer.get_node( layer_idx ) = this->activation( this->get_parameter( spacial_idx ), layer.get_node( layer_idx ) );
                }
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::backpropagation( NodeLayer<U>& layer ) requires ( is_type<T, none>::value && ( M == ParameterMode::none ) )
        {
            Size size = layer.get_size();

            for ( Idx i = 0; i < size; ++i )
            {
                layer.get_delta( i ) = this->d_activation( layer.get_node( i ) ) * layer.get_delta( i );
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::backpropagation( NodeLayer<U>& layer ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) )
        {
            Size size = layer.get_size();

            T sum( 0 );
            for ( Idx i = 0; i < size; ++i )
            {
                T gradient = this->d_activation( this->parameter, layer.get_node( i ) ) * layer.get_delta( i );
                sum += gradient;
                layer.get_delta( i ) = gradient;
            }
            this->jacobian += ( sum / size );

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::backpropagation( NodeLayer<U>& layer ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) )
        {
            const Shape<5>& layer_shape       = layer.get_shape();
            const Shape<4>& parameters_shape  = this->parameters.get_shape();

            Size layer_spacial_size     = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != parameters_shape.channels() ) { return Error::MISMATCHED_CHANNELS; }

            for ( Dim b_dim = 0; b_dim < layer_shape.batches(); ++b_dim )
            {
                Idx b_idx = b_dim * layer_shape.channels();
                for ( Dim c_dim = 0; c_dim < layer_shape.channels(); ++c_dim )
                {
                    Idx c_idx = ( b_idx + c_dim ) * layer_spacial_size;

                    const U& parameter = this->get_parameter( c_dim );

                    U gradient( 0 );

                    for ( Idx spacial_idx = 0; spacial_idx < layer_spacial_size; ++spacial_idx )
                    {
                        Idx index = c_idx + spacial_idx;

                        T gradient = this->d_activation( parameter, layer.get_node( index ) ) * layer.get_delta( index );
                        gradient += gradient;
                        layer.get_delta( index ) = gradient;
                    }

                    this->get_jacobian( c_dim ) += ( gradient / layer_spacial_size );
                }
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        Error ActivationLayer<T, A, M>::backpropagation( NodeLayer<U>& layer ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) )
        {
            const Shape<5>& layer_shape       = layer.get_shape();
            const Shape<4>& parameters_shape  = this->parameters.get_shape();

            Size spacial_size = parameters_shape.distance( 0, 4 );

            if ( layer_shape.distance( 0, 4 ) != spacial_size ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim batch = 0; batch < layer_shape.batches(); batch++ )
            {
                Idx batch_index = batch * spacial_size;
                for ( Idx spacial_idx = 0; spacial_idx < spacial_size; ++spacial_idx )
                {
                    Idx index = batch_index + spacial_idx;

                    T gradient = this->d_activation( this->get_parameter( spacial_idx ), layer.get_node( index ) ) * layer.get_delta( index );
                    this->get_jacobian( index ) += gradient;
                    layer.get_delta( index ) = gradient;
                }
            }

            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_normal( const Dim batch_size, const StepSize step_size ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->parameter, this->jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_normal( const Dim batch_size, const StepSize step_size ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->parameters, this->jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->parameter, this->jacobian, this->momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->parameters, this->jacobian, this->momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->parameter, this->jacobian, this->momentum, this->velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        Error ActivationLayer<T, A, M>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->parameters, this->jacobian, this->momentum, this->velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::sigmoid ) )
        {
            return sigmoid( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::tanh ) )
        {
            return htan( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::relu ) )
        {
            return relu( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::swish ) )
        {
            return swish( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::leaky_relu ) )
        {
            return leaky_relu( parameter, value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::elu ) )
        {
            return elu( parameter, value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::eswish ) )
        {
            return eswish( parameter, value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::sigmoid ) )
        {
            return d_sigmoid( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::tanh ) )
        {
            return d_htan( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::relu ) )
        {
            return d_relu( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const U value ) const requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::swish ) )
        {
            return d_swish( value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::leaky_relu ) )
        {
            return d_leaky_relu( parameter, value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::elu ) )
        {
            return d_elu( parameter, value );
        }

        template <typename T, ActivationType A, ParameterMode M>
        template <typename U>
        inline U ActivationLayer<T, A, M>::d_activation( const T parameter, const U value ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::eswish ) )
        {
            return d_eswish( parameter, value );
        }

} // namespace nn

//: Specializations
namespace nn
{
    #define CLASS( type_a, activation_type, parameter_mode )\
        template class ActivationLayer<type_a, activation_type, parameter_mode>;\

    #define FUNCTION( type_a, activation_type, parameter_mode, type_b )\
        template Error ActivationLayer<type_a, activation_type, parameter_mode>::inference( NodeLayer<type_b>& layer ) const;\
        template Error ActivationLayer<type_a, activation_type, parameter_mode>::backpropagation( NodeLayer<type_b>& layer );

    CLASS( none, ActivationType::sigmoid, ParameterMode::none )
    CLASS( none, ActivationType::tanh, ParameterMode::none )
    CLASS( none, ActivationType::relu, ParameterMode::none )
    CLASS( none, ActivationType::swish, ParameterMode::none )
    CLASS( i32, ActivationType::leaky_relu, ParameterMode::layer_wise )
    CLASS( i64, ActivationType::leaky_relu, ParameterMode::layer_wise )
    CLASS( f32, ActivationType::leaky_relu, ParameterMode::layer_wise )
    CLASS( f64, ActivationType::leaky_relu, ParameterMode::layer_wise )
    CLASS( i32, ActivationType::leaky_relu, ParameterMode::channel_wise )
    CLASS( i64, ActivationType::leaky_relu, ParameterMode::channel_wise )
    CLASS( f32, ActivationType::leaky_relu, ParameterMode::channel_wise )
    CLASS( f64, ActivationType::leaky_relu, ParameterMode::channel_wise )
    CLASS( i32, ActivationType::leaky_relu, ParameterMode::element_wise )
    CLASS( i64, ActivationType::leaky_relu, ParameterMode::element_wise )
    CLASS( f32, ActivationType::leaky_relu, ParameterMode::element_wise )
    CLASS( f64, ActivationType::leaky_relu, ParameterMode::element_wise )
    CLASS( f32, ActivationType::elu, ParameterMode::layer_wise )
    CLASS( f64, ActivationType::elu, ParameterMode::layer_wise )
    CLASS( f32, ActivationType::elu, ParameterMode::channel_wise )
    CLASS( f64, ActivationType::elu, ParameterMode::channel_wise )
    CLASS( f32, ActivationType::elu, ParameterMode::element_wise )
    CLASS( f64, ActivationType::elu, ParameterMode::element_wise )

    FUNCTION( none, ActivationType::sigmoid, ParameterMode::none, f32 )
    FUNCTION( none, ActivationType::sigmoid, ParameterMode::none, f64 )
    FUNCTION( none, ActivationType::tanh, ParameterMode::none, f32 )
    FUNCTION( none, ActivationType::tanh, ParameterMode::none, f64 )
    FUNCTION( none, ActivationType::relu, ParameterMode::none, f32 )
    FUNCTION( none, ActivationType::relu, ParameterMode::none, f64 )
    FUNCTION( none, ActivationType::swish, ParameterMode::none, f32 )
    FUNCTION( none, ActivationType::swish, ParameterMode::none, f64 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::layer_wise, i32 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::layer_wise, i64 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::layer_wise, i32 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::layer_wise, i64 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::layer_wise, f32 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::layer_wise, f64 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::layer_wise, f32 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::layer_wise, f64 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::channel_wise, i32 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::channel_wise, i64 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::channel_wise, i32 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::channel_wise, i64 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::channel_wise, f32 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::channel_wise, f64 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::channel_wise, f32 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::channel_wise, f64 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::element_wise, i32 )
    FUNCTION( i32, ActivationType::leaky_relu, ParameterMode::element_wise, i64 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::element_wise, i32 )
    FUNCTION( i64, ActivationType::leaky_relu, ParameterMode::element_wise, i64 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::element_wise, f32 )
    FUNCTION( f32, ActivationType::leaky_relu, ParameterMode::element_wise, f64 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::element_wise, f32 )
    FUNCTION( f64, ActivationType::leaky_relu, ParameterMode::element_wise, f64 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::layer_wise, f32 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::layer_wise, f64 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::layer_wise, f32 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::layer_wise, f64 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::channel_wise, f32 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::channel_wise, f64 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::channel_wise, f32 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::channel_wise, f64 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::element_wise, f32 )
    FUNCTION( f32, ActivationType::elu, ParameterMode::element_wise, f64 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::element_wise, f32 )
    FUNCTION( f64, ActivationType::elu, ParameterMode::element_wise, f64 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::layer_wise, f32 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::layer_wise, f64 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::layer_wise, f32 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::layer_wise, f64 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::channel_wise, f32 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::channel_wise, f64 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::channel_wise, f32 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::channel_wise, f64 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::element_wise, f32 )
    FUNCTION( f32, ActivationType::eswish, ParameterMode::element_wise, f64 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::element_wise, f32 )
    FUNCTION( f64, ActivationType::eswish, ParameterMode::element_wise, f64 )

    #undef CLASS
    #undef FUNCTION
} // namespace nn

/*
//: Specializations
namespace nn
{
    #define CLASS( type, activation_type, parameter_mode ) ActivationLayer<type, activation_type, parameter_mode>

    #define CLASS_SIGMOID                               CLASS( none,    ActivationType::sigmoid,       ParameterMode::none )
    #define CLASS_TANH                                  CLASS( none,    ActivationType::tanh,          ParameterMode::none )
    #define CLASS_RELU                                  CLASS( none,    ActivationType::relu,          ParameterMode::none )
    #define CLASS_SWISH                                 CLASS( none,    ActivationType::swish,         ParameterMode::none )
    #define CLASS_LEAKY_RELU( parameter_mode, type )    CLASS( type,    ActivationType::leaky_relu,    parameter_mode )
    #define CLASS_ELU( parameter_mode, type )           CLASS( type,    ActivationType::elu,           parameter_mode )
    #define CLASS_ESWISH( parameter_mode, type )        CLASS( type,    ActivationType::eswish,        parameter_mode )

    #define INSTANTIATE_CLASS_SIGMOID                               template class CLASS_SIGMOID;
    #define INSTANTIATE_CLASS_TANH                                  template class CLASS_TANH;
    #define INSTANTIATE_CLASS_RELU                                  template class CLASS_RELU;
    #define INSTANTIATE_CLASS_SWISH                                 template class CLASS_SWISH;
    #define INSTANTIATE_CLASS_LEAKY_RELU( parameter_mode, type )    template class CLASS_LEAKY_RELU(    parameter_mode, type );
    #define INSTANTIATE_CLASS_ELU( parameter_mode, type )           template class CLASS_ELU(           parameter_mode, type );
    #define INSTANTIATE_CLASS_ESWISH( parameter_mode, type )        template class CLASS_ESWISH(        parameter_mode, type );

    #define COMPILE_CLASS_SIGMOID                       INSTANTIATE_CLASS_SIGMOID
    #define COMPILE_CLASS_TANH                          INSTANTIATE_CLASS_TANH
    #define COMPILE_CLASS_RELU                          INSTANTIATE_CLASS_RELU
    #define COMPILE_CLASS_SWISH                         INSTANTIATE_CLASS_SWISH
    #define COMPILE_CLASS_LEAKY_RELU    FOR_3_FOR_4(    INSTANTIATE_CLASS_LEAKY_RELU,  ParameterMode::layer_wise, ParameterMode::channel_wise, ParameterMode::element_wise, i32, i64, f32, f64 )
    #define COMPILE_CLASS_ELU           FOR_3_FOR_2(    INSTANTIATE_CLASS_ELU,         ParameterMode::layer_wise, ParameterMode::channel_wise, ParameterMode::element_wise,           f32, f64 )
    #define COMPILE_CLASS_ESWISH        FOR_3_FOR_2(    INSTANTIATE_CLASS_ESWISH,      ParameterMode::layer_wise, ParameterMode::channel_wise, ParameterMode::element_wise,           f32, f64 )

    #define FUNCTION_INFERENCE( class_name, type )           Error class_name::inference                                ( NodeLayer<type>& layer ) const
    #define FUNCTION_BACKPROPAGATION( class_name, type )     Error class_name::backpropagation                          ( NodeLayer<type>& layer )
    #define FUNCTION_GRADIENT_DECENT_NORMAL( class_name )    Error class_name::gradient_decent<TrainingMode::normal>    ( const Dim batch_size, const StepSize step_size )
    #define FUNCTION_GRADIENT_DECENT_MOMENTUM( class_name )  Error class_name::gradient_decent<TrainingMode::momentum>  ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
    #define FUNCTION_GRADIENT_DECENT_ADAM( class_name )      Error class_name::gradient_decent<TrainingMode::adam>      ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )

    #define INSTANTIATE_FUNCTION_INFERENCE( class_name, type )           template FUNCTION_INFERENCE( class_name, type );
    #define INSTANTIATE_FUNCTION_BACKPROPAGATION( class_name, type )     template FUNCTION_BACKPROPAGATION( class_name, type );
    #define INSTANTIATE_FUNCTION_GRADIENT_DECENT_NORMAL( class_name )    template FUNCTION_GRADIENT_DECENT_NORMAL( class_name );
    #define INSTANTIATE_FUNCTION_GRADIENT_DECENT_MOMENTUM( class_name )  template FUNCTION_GRADIENT_DECENT_MOMENTUM( class_name );
    #define INSTANTIATE_FUNCTION_GRADIENT_DECENT_ADAM( class_name )      template FUNCTION_GRADIENT_DECENT_ADAM( class_name );

    #define INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION( class_name, type )\
        INSTANTIATE_FUNCTION_INFERENCE( class_name, type )\
        INSTANTIATE_FUNCTION_BACKPROPAGATION( class_name, type )\

    #define INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL( class_name, type )\
        INSTANTIATE_FUNCTION_INFERENCE( class_name, type )\
        INSTANTIATE_FUNCTION_BACKPROPAGATION( class_name, type )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_NORMAL( class_name )

    #define INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM( class_name, type )\
        INSTANTIATE_FUNCTION_INFERENCE( class_name, type )\
        INSTANTIATE_FUNCTION_BACKPROPAGATION( class_name, type )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_NORMAL( class_name )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_MOMENTUM( class_name )

    #define INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM( class_name, type )\
        INSTANTIATE_FUNCTION_INFERENCE( class_name, type )\
        INSTANTIATE_FUNCTION_BACKPROPAGATION( class_name, type )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_NORMAL( class_name )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_MOMENTUM( class_name )\
        INSTANTIATE_FUNCTION_GRADIENT_DECENT_ADAM( class_name )

    #define COMPILE_FUNCTION_SIGMOID\
        FOR_1_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION, \
            CLASS_SIGMOID, \
            f32, f64 )
    #define COMPILE_FUNCTION_TANH\
        FOR_1_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION, \
            CLASS_TANH, \
            f32, f64 )
    #define COMPILE_FUNCTION_RELU\
        FOR_1_FOR_4( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION, \
            CLASS_RELU, \
            i32, i64, f32, f64 )
    #define COMPILE_FUNCTION_SWISH\
        FOR_1_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION, \
            CLASS_SWISH, \
            f32, f64 )
    #define COMPILE_FUNCTION_LEAKY_RELU\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM, \
            CLASS_LEAKY_RELU( ParameterMode::layer_wise, i32 ), \
            CLASS_LEAKY_RELU( ParameterMode::layer_wise, i64 ), \
            i32, i64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_LEAKY_RELU( ParameterMode::layer_wise, f32 ), \
            CLASS_LEAKY_RELU( ParameterMode::layer_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM, \
            CLASS_LEAKY_RELU( ParameterMode::channel_wise, i32 ), \
            CLASS_LEAKY_RELU( ParameterMode::channel_wise, i64 ), \
            i32, i64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_LEAKY_RELU( ParameterMode::channel_wise, f32 ), \
            CLASS_LEAKY_RELU( ParameterMode::channel_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM, \
            CLASS_LEAKY_RELU( ParameterMode::element_wise, i32 ), \
            CLASS_LEAKY_RELU( ParameterMode::element_wise, i64 ), \
            i32, i64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_LEAKY_RELU( ParameterMode::element_wise, f32 ), \
            CLASS_LEAKY_RELU( ParameterMode::element_wise, f64 ), \
            f32, f64 )
    #define COMPILE_FUNCTION_ELU\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ELU( ParameterMode::layer_wise, f32 ), \
            CLASS_ELU( ParameterMode::layer_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ELU( ParameterMode::channel_wise, f32 ), \
            CLASS_ELU( ParameterMode::channel_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ELU( ParameterMode::element_wise, f32 ), \
            CLASS_ELU( ParameterMode::element_wise, f64 ), \
            f32, f64 )
    #define COMPILE_FUNCTION_ESWISH\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ESWISH( ParameterMode::layer_wise, f32 ), \
            CLASS_ESWISH( ParameterMode::layer_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ESWISH( ParameterMode::channel_wise, f32 ), \
            CLASS_ESWISH( ParameterMode::channel_wise, f64 ), \
            f32, f64 )\
        FOR_2_FOR_2( INSTANTIATE_FUNCTIONS_INFERENCE_BACKPROPAGATION_GRADIENT_DECENT_NORMAL_MOMENTUM_ADAM, \
            CLASS_ESWISH( ParameterMode::element_wise, f32 ), \
            CLASS_ESWISH( ParameterMode::element_wise, f64 ), \
            f32, f64 )

    COMPILE_CLASS_SIGMOID       COMPILE_FUNCTION_SIGMOID
    COMPILE_CLASS_TANH          COMPILE_FUNCTION_TANH
    COMPILE_CLASS_RELU          COMPILE_FUNCTION_RELU
    COMPILE_CLASS_SWISH         COMPILE_FUNCTION_SWISH
    COMPILE_CLASS_LEAKY_RELU    COMPILE_FUNCTION_LEAKY_RELU
    COMPILE_CLASS_ELU           COMPILE_FUNCTION_ELU
    COMPILE_CLASS_ESWISH        COMPILE_FUNCTION_ESWISH

} // namespace nn
*/

/*
//: Specializations
namespace nn
{
    #define FunctionMacro( activation_type, parameter_mode, type_a, type_b )\
        template Error ActivationLayer<type_a, activation_type, parameter_mode>::inference( NodeLayer<type_b>& layer ) const;\
        template Error ActivationLayer<type_a, activation_type, parameter_mode>::backpropagation( NodeLayer<type_b>& layer );\

    #define ClassMacroDInt( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::normal>( const Dim batch_size, const StepSize step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::momentum>( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::adam>( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );\
        FunctionMacro( activation_type, parameter_mode, type, i32 )\
        FunctionMacro( activation_type, parameter_mode, type, i64 )\

    #define ClassMacroDFloat( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::normal>( const Dim batch_size, const StepSize step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::momentum>( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::adam>( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );\
        FunctionMacro( activation_type, parameter_mode, type, f32 )\
        FunctionMacro( activation_type, parameter_mode, type, f64 )

        #define ClassMacroDNoneInt( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        FunctionMacro( activation_type, parameter_mode, type, i32 )\
        FunctionMacro( activation_type, parameter_mode, type, i64 )\

    #define ClassMacroDNoneFloat( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        FunctionMacro( activation_type, parameter_mode, type, f32 )\
        FunctionMacro( activation_type, parameter_mode, type, f64 )

    #define ClassMacroDNone( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        FunctionMacro( activation_type, parameter_mode, type, i32 )\
        FunctionMacro( activation_type, parameter_mode, type, i64 )\
        FunctionMacro( activation_type, parameter_mode, type, f32 )\
        FunctionMacro( activation_type, parameter_mode, type, f64 )

    #define ClassMacroD( activation_type, parameter_mode, type )\
        template class ActivationLayer<type, activation_type, parameter_mode>;\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::normal>( const Dim batch_size, const StepSize step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::momentum>( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );\
        template Error ActivationLayer<type, activation_type, parameter_mode>::gradient_decent<TrainingMode::adam>( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );\
        FunctionMacro( activation_type, parameter_mode, type, i32 )\
        FunctionMacro( activation_type, parameter_mode, type, i64 )\
        FunctionMacro( activation_type, parameter_mode, type, f32 )\
        FunctionMacro( activation_type, parameter_mode, type, f64 )

    #define ClassMacroCInt( activation_type, parameter_mode )\
        ClassMacroDInt( activation_type, parameter_mode, i32 )\
        ClassMacroDInt( activation_type, parameter_mode, i64 )\

    #define ClassMacroCFloat( activation_type, parameter_mode )\
        ClassMacroDFloat( activation_type, parameter_mode, f32 )\
        ClassMacroDFloat( activation_type, parameter_mode, f64 )

    #define ClassMacroC( activation_type, parameter_mode )\
        ClassMacroD( activation_type, parameter_mode, i32 )\
        ClassMacroD( activation_type, parameter_mode, i64 )\
        ClassMacroD( activation_type, parameter_mode, f32 )\
        ClassMacroD( activation_type, parameter_mode, f64 )

    #define ClassMacroBInt( activation_type )\
        ClassMacroCInt( activation_type, ParameterMode::layer_wise )\
        ClassMacroCInt( activation_type, ParameterMode::channel_wise )\
        ClassMacroCInt( activation_type, ParameterMode::element_wise )

    #define ClassMacroBFloat( activation_type )\
        ClassMacroCFloat( activation_type, ParameterMode::layer_wise )\
        ClassMacroCFloat( activation_type, ParameterMode::channel_wise )\
        ClassMacroCFloat( activation_type, ParameterMode::element_wise )

    #define ClassMacroB( activation_type )\
        ClassMacroC( activation_type, ParameterMode::layer_wise )\
        ClassMacroC( activation_type, ParameterMode::channel_wise )\
        ClassMacroC( activation_type, ParameterMode::element_wise )

    #define ClassMacro\
        ClassMacroB( ActivationType::leaky_relu )\
        ClassMacroBFloat( ActivationType::elu )\
        ClassMacroBFloat( ActivationType::eswish )\
        ClassMacroDNoneFloat( ActivationType::sigmoid, ParameterMode::none, none )\
        ClassMacroDNoneFloat( ActivationType::tanh, ParameterMode::none, none )\
        ClassMacroDNone( ActivationType::relu, ParameterMode::none, none )\
        ClassMacroDNoneFloat( ActivationType::swish, ParameterMode::none, none )

    ClassMacro

    #undef FunctionMacro
    #undef ClassMacroD
    #undef ClassMacroC
    #undef ClassMacroB
    #undef ClassMacroParameters
    #undef ClassMacroNoParameters
    #undef ClassMacro
} // namespace nn
*/
