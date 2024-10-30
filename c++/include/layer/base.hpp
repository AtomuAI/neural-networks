// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BASE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BASE_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
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

namespace nn
{
    enum class LayerType : u8
    {
        base_layer,
        node_layer,
        bias_layer,
        dense_layer,
        convolution_layer,
        pooling_layer,
        activation_layer,
        normalization_layer,
        mask_layer,
        dropout_layer,
        softmax_layer,
        cost_layer,
        rotation_layer,
        translation_layer,
        dilation_layer,
        num_layer_types
    };

    class BaseLayer
    {
        //: Members
        public:
            const LayerType  layer_type;
        protected:
            TrainingMode            training_mode;
            Counter<u64>            time_step;

        //: Constructors
        public:
            BaseLayer
            (
                const LayerType layer_type          = LayerType::base_layer,
                const TrainingMode training_mode    = TrainingMode::off
            );

        //: Destructors
        public:
            virtual ~BaseLayer();

        //: Methods
        public:
                                        LayerType           get_layer_type                      () const;
                                        const Counter<u64>& get_timer                           () const;
                                        u64                 get_time_step                       () const;
                                        void                tick_time_step                      ();
                                        void                reset_time_step                     ();
            template <typename T, u8 N> inline void         reshape                             ( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity );
            template <typename T, u8 N> inline void         reshape                             ( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta );
            template <typename T, u8 N> inline void         resize                              ( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity );
            template <typename T, u8 N> inline void         resize                              ( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta );
            template <typename T, u8 N> inline void         allocate_training_memory_off        ( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity  );
            template <typename T, u8 N> inline void         allocate_training_memory_normal     ( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity  );
            template <typename T, u8 N> inline void         allocate_training_memory_momentum   ( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity );
            template <typename T, u8 N> inline void         allocate_training_memory_adam       ( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity );
            template <typename T, u8 N> inline void         allocate_training_memory            ( const Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity );
            template <typename T, u8 N> inline void         allocate_training_memory            ( const Tensor<T, N>& nodes, Tensor<T, N>& delta );
            template <typename T>       inline void         gradient_decent_normal              ( T& parameter, T& jacobian, const Dim batch_size, const StepSize step_size );
            template <typename T, u8 N> inline void         gradient_decent_normal              ( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, const Dim batch_size, const StepSize step_size );
            template <typename T>       inline void         gradient_decent_momentum            ( T& parameter, T& jacobian, T& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size);
            template <typename T, u8 N> inline void         gradient_decent_momentum            ( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size);
            template <typename T>       inline void         gradient_decent_adam                ( T& parameter, T& jacobian, T& momentum, T& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );
            template <typename T, u8 N> inline void         gradient_decent_adam                ( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );
    };

    //: Inline Definitions
        template <typename T, u8 N>
        inline void BaseLayer::reshape( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
        {
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    parameters.reshape( shape );
                    break;
                }
                case TrainingMode::normal:
                {
                    parameters.reshape( shape );
                    jacobian.reshape( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    parameters.reshape( shape );
                    jacobian.reshape( shape );
                    momentum.reshape( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    parameters.reshape( shape );
                    jacobian.reshape( shape );
                    momentum.reshape( shape );
                    velocity.reshape( shape );
                    break;
                }
                default: { break; }
            }
        }

        template <typename T, u8 N>
        inline void BaseLayer::reshape( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta )
        {
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    nodes.reshape( shape );
                    break;
                }
                default:
                {
                    nodes.reshape( shape );
                    delta.reshape( shape );
                    break;
                }
            }
        }

        template <typename T, u8 N>
        inline void BaseLayer::resize( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
        {
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    parameters.resize( shape );
                    break;
                }
                case TrainingMode::normal:
                {
                    parameters.resize( shape );
                    jacobian.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    parameters.resize( shape );
                    jacobian.resize( shape );
                    momentum.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    parameters.resize( shape );
                    jacobian.resize( shape );
                    momentum.resize( shape );
                    velocity.resize( shape );
                    break;
                }
                default: { break; }
            }
        }

        template <typename T, u8 N>
        inline void BaseLayer::resize( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta )
        {
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    nodes.resize( shape );
                    break;
                }
                default:
                {
                    nodes.resize( shape );
                    delta.resize( shape );
                    break;
                }
            }
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory_off( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity  )
        {
            jacobian.resize( Shape<N>() );
            momentum.resize( Shape<N>() );
            velocity.resize( Shape<N>() );
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory_normal( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity  )
        {
            jacobian.resize( shape );
            momentum.resize( Shape<N>() );
            velocity.resize( Shape<N>() );
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory_momentum( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
        {
            jacobian.resize( shape );
            momentum.resize( shape );
            velocity.resize( Shape<N>() );
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory_adam( const Shape<N>& shape, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
        {
            jacobian.resize( shape );
            momentum.resize( shape );
            velocity.resize( shape );
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory( const Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
        {
            const Shape<N>& shape = parameters.get_shape();
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    this->allocate_training_memory_off( shape, jacobian, momentum, velocity  );
                    break;
                }
                case TrainingMode::normal:
                {
                    this->allocate_training_memory_normal( shape, jacobian, momentum, velocity );
                    break;
                }
                case TrainingMode::momentum:
                {
                    this->allocate_training_memory_momentum( shape, jacobian, momentum, velocity );
                    break;
                }
                case TrainingMode::adam:
                {
                    this->allocate_training_memory_adam( shape, jacobian, momentum, velocity );
                    break;
                }
                default: { break; }
            }
        }

        template <typename T, u8 N>
        inline void BaseLayer::allocate_training_memory( const Tensor<T, N>& nodes, Tensor<T, N>& delta )
        {
            const Shape<N>& shape = nodes.get_shape();
            switch ( this->training_mode )
            {
                case TrainingMode::off:
                {
                    delta.resize( Shape<N>() );
                    break;
                }
                default:
                {
                    delta.resize( shape );
                    break;
                }
            }
        }

        template <typename T>
        inline void BaseLayer::gradient_decent_normal( T& parameter, T& jacobian, const Dim batch_size, const StepSize step_size )
        {
            T gradient = jacobian / batch_size;

            parameter += step_size * gradient;

            jacobian = 0;
        }

        template <typename T, u8 N>
        inline void BaseLayer::gradient_decent_normal( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, const Dim batch_size, const StepSize step_size )
        {
            Size size = parameters.get_size();

            for ( Idx index = 0; index < size; ++index )
            {
                T gradient = jacobian[ index ] / batch_size;

                parameters[ index ] += step_size * gradient;
            }

            jacobian.zero();
        }

        template <typename T>
        inline void BaseLayer::gradient_decent_momentum( T& parameter, T& jacobian, T& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size)
        {
            T gradient = jacobian / batch_size;

            parameter += step_size * ( ( momentum_step_size * momentum ) + gradient );

            momentum = gradient;

            jacobian = 0;
        }

        template <typename T, u8 N>
        inline void BaseLayer::gradient_decent_momentum( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size)
        {
            Size size = parameters.get_size();

            for ( Idx index = 0; index < size; ++index )
            {
                T gradient = jacobian[ index ] / batch_size;

                parameters[ index ] += step_size * ( ( momentum_step_size * momentum[ index ] ) + gradient );

                momentum[ index ] = gradient;
            }

            jacobian.zero();
        }

        template <typename T>
        inline void BaseLayer::gradient_decent_adam( T& parameter, T& jacobian, T& momentum, T& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            u64 step = this->get_time_step();
            Beta beta1_mp = 1 - nn::pow( beta1, step );
            Beta beta2_mp = 1 - nn::pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            T gradient = jacobian / batch_size;

            momentum = ( beta1 * momentum ) + ( beta1_m * gradient );
            velocity = ( beta2 * velocity ) + ( beta2_m * ( gradient * gradient ) );

            T momentum_hat = momentum / beta1_mp;
            T velocity_hat = velocity / beta2_mp;

            parameter += step_size * ( momentum_hat / ( nn::sqrt( velocity_hat ) + epsilon ) );

            jacobian = 0;
        }

        template <typename T, u8 N>
        inline void BaseLayer::gradient_decent_adam( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            u64 step = this->get_time_step();
            Beta beta1_mp = 1 - nn::pow( beta1, step );
            Beta beta2_mp = 1 - nn::pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            Size size = parameters.get_size();

            for ( Idx index = 0; index < size; ++index )
            {
                T gradient = jacobian[ index ] / batch_size;

                momentum[ index ] = ( beta1 * momentum[ index ] ) + ( beta1_m * gradient );
                velocity[ index ] = ( beta2 * velocity[ index ] ) + ( beta2_m * ( gradient * gradient ) );

                T momentum_hat = momentum[ index ] / beta1_mp;
                T velocity_hat = momentum[ index ] / beta2_mp;

                parameters[ index ] += step_size * ( momentum_hat / ( nn::sqrt( velocity_hat ) + epsilon ) );
            }

            jacobian.zero();
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BASE_HPP_
