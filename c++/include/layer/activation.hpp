// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_ACTIVATION_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_ACTIVATION_HPP_

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

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

namespace nn
{
    enum class ParameterMode : u8
    {
        none,
        layer_wise,
        channel_wise,
        element_wise,
    };

    enum class ActivationType : u8
    {
        sigmoid,
        tanh,
        relu,
        leaky_relu,
        elu,
        swish,
        eswish
    };

    template <typename T, ActivationType A, ParameterMode M>
    struct ActivationParameters;

    template <typename T, ActivationType A>
    struct ActivationParameters<T, A, ParameterMode::none> {};

    template <typename T, ActivationType A>
    struct ActivationParameters<T, A, ParameterMode::layer_wise>
    {
        T parameter = T( 0 );
        T jacobian = T( 0 );
        T momentum = T( 0 );
        T velocity = T( 0 );

        ActivationParameters() {}
        explicit ActivationParameters( const T value ) : parameter( value ) {}
    };

    template <typename T, ActivationType A>
    struct ActivationParameters<T, A, ParameterMode::channel_wise>
    {
        Tensor<T, 4> parameters;
        Tensor<T, 4> jacobian;
        Tensor<T, 4> momentum;
        Tensor<T, 4> velocity;

        explicit ActivationParameters( const Shape<4>& shape ) : parameters( shape ) {}
        ActivationParameters( const Shape<4>& shape, const T scalar ) : parameters( shape, scalar ) {}
        ActivationParameters( const Shape<4>& shape, const std::vector<T>& parameters ) : parameters( shape, parameters ) {}
    };

    template <typename T, ActivationType A>
    struct ActivationParameters<T, A, ParameterMode::element_wise>
    {
        Tensor<T, 4> parameters;
        Tensor<T, 4> jacobian;
        Tensor<T, 4> momentum;
        Tensor<T, 4> velocity;

        explicit ActivationParameters( const Shape<4>& shape ) : parameters( shape ) {}
        ActivationParameters( const Shape<4>& shape, const T scalar ) : parameters( shape, scalar ) {}
        ActivationParameters( const Shape<4>& shape, const std::vector<T>& parameters ) : parameters( shape, parameters ) {}
    };

    template <typename T, ActivationType A, ParameterMode M>
    class ActivationLayer : protected ActivationParameters<T, A, M>, public BaseLayer
    {
        //: Members
        public:
            static const ActivationType type = A;
            static const ParameterMode  mode = M;

        //: Constructors
        public:
            ActivationLayer() requires ( is_type<T, none>::value && ( M == ParameterMode::none ) );
            ActivationLayer
            (
                const T parameter                   = T()
            ) requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
            ActivationLayer
            (
                const u8 channels                   = static_cast<u8>(0),
                const T parameter                   = T()
            ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
            ActivationLayer
            (
                const u8 channels,
                const std::vector<T>& data
            ) requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
            ActivationLayer
            (
                const Shape<4>& shape               = Shape<4>( 0, 0, 0, 0 ),
                const T parameter                   = T()
            ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );
            ActivationLayer
            (
                const Shape<4>& shape,
                const std::vector<T>& data
            ) requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );

        //: Destructors
        public:
            virtual ~ActivationLayer();

        //: Methods
        public:
                                        void                set_training_mode    ( TrainingMode training_mode )                                                                                  requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        const Shape<4>&     get_shape                   () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        Size                get_size                    () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        T                   get_parameter               () const                                                                                                        requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        T                   get_jacobian                () const                                                                                                        requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        T                   get_momentum                () const                                                                                                        requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        T                   get_velocity                () const                                                                                                        requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        const Tensor<T, 4>& get_parameters              () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        const Tensor<T, 4>& get_jacobian                () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        const Tensor<T, 4>& get_momentum                () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        const Tensor<T, 4>& get_velocity                () const                                                                                                        requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                reshape                     ( const u8 channels )                                                                                           requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
                                        void                reshape                     ( const Shape<4>& shape )                                                                                       requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );
                                        void                resize                      ( const u8 channels )                                                                                           requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
                                        void                resize                      ( const Shape<4>& shape )                                                                                       requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );
                                        void                set_parameter               ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                fill_parameters             ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                zero_parameter              ()                                                                                                              requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                zero_parameters             ()                                                                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                randomize_parameter         ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                randomize_parameters        ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
            template <typename U>       Error                inference                   ( NodeLayer<U>& layer ) const                                                                                   requires ( is_type<T, none>::value && ( M == ParameterMode::none ) );
            template <typename U>       Error                inference                   ( NodeLayer<U>& layer ) const                                                                                   requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
            template <typename U>       Error                inference                   ( NodeLayer<U>& layer ) const                                                                                   requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
            template <typename U>       Error                inference                   ( NodeLayer<U>& layer ) const                                                                                   requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );
            template <typename U>       Error                backpropagation             ( NodeLayer<U>& layer )                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) );
            template <typename U>       Error                backpropagation             ( NodeLayer<U>& layer )                                                                                         requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
            template <typename U>       Error                backpropagation             ( NodeLayer<U>& layer )                                                                                         requires ( !is_type<T, none>::value && ( M == ParameterMode::channel_wise ) );
            template <typename U>       Error                backpropagation             ( NodeLayer<U>& layer )                                                                                         requires ( !is_type<T, none>::value && ( M == ParameterMode::element_wise ) );
                                        Error                gradient_decent_normal      ( const Dim batch_size, const StepSize step_size )                                                              requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) );
                                        Error                gradient_decent_normal      ( const Dim batch_size, const StepSize step_size )                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        Error                gradient_decent_momentum    ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )                            requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) );
                                        Error                gradient_decent_momentum    ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )                            requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        Error                gradient_decent_adam        ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )   requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise) );
                                        Error                gradient_decent_adam        ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )   requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
        private:
                                        void                set_jacobian                ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                set_momentum                ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                set_velocity                ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                fill_jacobian               ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                fill_momentum               ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                fill_velocity               ( const T value )                                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                zero_jacobian               ()                                                                                                              requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                zero_momentum               ()                                                                                                              requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                zero_velocity               ()                                                                                                              requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                zero_jacobian               ()                                                                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                zero_momentum               ()                                                                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                zero_velocity               ()                                                                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                randomize_jacobian          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                randomize_momentum          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                randomize_velocity          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( M == ParameterMode::layer_wise ) );
                                        void                randomize_jacobian          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                randomize_momentum          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        void                randomize_velocity          ( const T min, const T max )                                                                                    requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
            template <typename U>       inline U            activation                  ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::sigmoid ) );
            template <typename U>       inline U            activation                  ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::tanh ) );
            template <typename U>       inline U            activation                  ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::relu ) );
            template <typename U>       inline U            activation                  ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::swish ) );
            template <typename U>       inline U            activation                  ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::leaky_relu ) );
            template <typename U>       inline U            activation                  ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::elu ) );
            template <typename U>       inline U            activation                  ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::eswish ) );
            template <typename U>       inline U            d_activation                ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::sigmoid ) );
            template <typename U>       inline U            d_activation                ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::tanh ) );
            template <typename U>       inline U            d_activation                ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::relu ) );
            template <typename U>       inline U            d_activation                ( const U value ) const                                                                                         requires ( is_type<T, none>::value && ( M == ParameterMode::none ) && ( A == ActivationType::swish ) );
            template <typename U>       inline U            d_activation                ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::leaky_relu ) );
            template <typename U>       inline U            d_activation                ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::elu ) );
            template <typename U>       inline U            d_activation                ( const T parameter, const U value ) const                                                                      requires ( !is_type<T, none>::value && ( ( M == ParameterMode::layer_wise ) || ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) && ( A == ActivationType::eswish ) );

    //: Operators
        public:
                                        inline T            get_parameter                         ( const Dim4D& indices ) const                                                            requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_jacobian                         ( const Dim4D& indices ) const                                                             requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_momentum                         ( const Dim4D& indices ) const                                                             requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_velocity                         ( const Dim4D& indices ) const                                                             requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_parameter                         ( const Idx index ) const                                                                              requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_jacobian                         ( const Idx index ) const                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_momentum                         ( const Idx index ) const                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T            get_velocity                         ( const Idx index ) const                                                                               requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_parameter                         ( const Dim4D& indices )                                                                  requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_parameter                         ( const Idx index )                                                                                    requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
        private:
                                        inline T&           get_jacobian                         ( const Dim4D& indices )                                                                   requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_momentum                         ( const Dim4D& indices )                                                                   requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_velocity                         ( const Dim4D& indices )                                                                   requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_jacobian                         ( const Idx index )                                                                                     requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_momentum                         ( const Idx index )                                                                                     requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
                                        inline T&           get_velocity                         ( const Idx index )                                                                                     requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) );
    };

    //: Inline Operators
        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_parameter( const Dim4D& indices ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_jacobian( const Dim4D& indices ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->jacobian[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_momentum( const Dim4D& indices ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->momentum[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_velocity( const Dim4D& indices ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->velocity[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_parameter( const Idx index ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_jacobian( const Idx index ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->jacobian[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_momentum( const Idx index ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->momentum[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T ActivationLayer<T, A, M>::get_velocity( const Idx index ) const requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->velocity[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_parameter( const Dim4D& indices ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_jacobian( const Dim4D& indices ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->jacobian[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_momentum( const Dim4D& indices ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->momentum[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_velocity( const Dim4D& indices ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->velocity[ indices ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_parameter( const Idx index ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->parameters[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_jacobian( const Idx index ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->jacobian[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_momentum( const Idx index ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->momentum[ index ];
        }

        template <typename T, ActivationType A, ParameterMode M>
        inline T& ActivationLayer<T, A, M>::get_velocity( const Idx index ) requires ( !is_type<T, none>::value && ( ( M == ParameterMode::channel_wise ) || ( M == ParameterMode::element_wise ) ) )
        {
            return this->velocity[ index ];
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_ACTIVATION_HPP_
