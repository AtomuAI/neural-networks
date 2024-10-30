// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NORMALIZATION_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NORMALIZATION_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>
#include <functional>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

namespace nn
{
    enum class NormalizationType : u8
    {
        batch_wise,
        layer_wise,
        instance_wise,
        group_wise
    };

    template <typename T, NormalizationType N>
    class NormalizationLayer : public BaseLayer
    {
        //: Members
        public:
            static const NormalizationType  type = N;
            static const u8                 dim  = std::conditional_t<(N == NormalizationType::batch_wise), std::integral_constant<u8, 4>, std::integral_constant<u8, 5>>::value;
        protected:
            Tensor<T, dim> mean;
            Tensor<T, dim> variance;
            Tensor<T, dim> beta; // offset parameter
            Tensor<T, dim> gamma; // scaling parameter
            Tensor<T, dim> beta_jacobian; // offset parameter
            Tensor<T, dim> gamma_jacobian; // scaling parameter
            Tensor<T, dim> beta_momentum; // offset parameter
            Tensor<T, dim> gamma_momentum; // scaling parameter
            Tensor<T, dim> beta_velocity; // offset parameter
            Tensor<T, dim> gamma_velocity; // scaling parameter

        //: Constructors
        public:
            NormalizationLayer
            (
                const u8                channels    = 0
            ) requires ( N == NormalizationType::batch_wise );

            NormalizationLayer
            (
                const u8                batches     = 0
            ) requires ( N == NormalizationType::layer_wise );

            NormalizationLayer
            (
                const u8                channels    = 0,
                const u8                batches     = 0
            ) requires ( N == NormalizationType::instance_wise );

            NormalizationLayer
            (
                const u8                channels    = 0,
                const u8                batches     = 0,
                const u8                group_size  = 0
            ) requires ( N == NormalizationType::group_wise );

        //: Destructors
        public:
            virtual ~NormalizationLayer();

        //: Methods
        public:
            const Shape<4>&            get_shape() const requires ( N == NormalizationType::batch_wise );
            const Shape<5>&            get_shape() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            Size                get_size() const;
            const Tensor<T, 4>& get_mean() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_variance() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_beta() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_gamma() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_beta_jacobian() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_gamma_jacobian() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_beta_momentum() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_gamma_momentum() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_beta_velocity() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 4>& get_gamma_velocity() const requires ( N == NormalizationType::batch_wise );
            const Tensor<T, 5>& get_mean() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_variance() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_beta() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_gamma() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_beta_jacobian() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_gamma_jacobian() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_beta_momentum() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_gamma_momentum() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_beta_velocity() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            const Tensor<T, 5>& get_gamma_velocity() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            void                reshape( const u8 channels ) requires ( N == NormalizationType::batch_wise );
            void                reshape( const u8 batches ) requires ( N == NormalizationType::layer_wise );
            void                reshape( const u8 channels, const u8 batches ) requires ( N == NormalizationType::instance_wise );
            void                reshape( const u8 channels, const u8 batches, const u8 group_size ) requires ( N == NormalizationType::group_wise );
            void                resize( const u8 channels ) requires ( N == NormalizationType::batch_wise );
            void                resize( const u8 batches ) requires ( N == NormalizationType::layer_wise );
            void                resize( const u8 channels, const u8 batches ) requires ( N == NormalizationType::instance_wise );
            void                resize( const u8 channels, const u8 batches, const u8 group_size ) requires ( N == NormalizationType::group_wise );
            void set_training_mode( TrainingMode training_mode ) requires ( N == NormalizationType::batch_wise );
            void set_training_mode( TrainingMode training_mode ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            template <typename U> Error stat_analysis( const NodeLayer<U>& layer ) requires ( ( N == NormalizationType::batch_wise ) );
            template <typename U> Error stat_analysis( const NodeLayer<U>& layer ) requires ( ( N == NormalizationType::layer_wise ) );
            template <typename U> Error stat_analysis( const NodeLayer<U>& layer ) requires ( ( N == NormalizationType::instance_wise ) );
            template <typename U> Error stat_analysis( const NodeLayer<U>& layer ) requires ( ( N == NormalizationType::group_wise ) );
            template <typename U> Error inference( NodeLayer<U>& layer ) const requires ( ( N == NormalizationType::batch_wise ) );
            template <typename U> Error inference( NodeLayer<U>& layer ) const requires ( ( N == NormalizationType::layer_wise ) );
            template <typename U> Error inference( NodeLayer<U>& layer ) const requires ( ( N == NormalizationType::instance_wise ) );
            template <typename U> Error inference( NodeLayer<U>& layer ) const requires ( ( N == NormalizationType::group_wise ) );
            template <typename U> Error backpropagation( NodeLayer<U>& layer ) requires ( ( N == NormalizationType::batch_wise ) );
            template <typename U> Error backpropagation( NodeLayer<U>& layer ) requires ( ( N == NormalizationType::layer_wise ) );
            template <typename U> Error backpropagation( NodeLayer<U>& layer ) requires ( ( N == NormalizationType::instance_wise ) );
            template <typename U> Error backpropagation( NodeLayer<U>& layer ) requires ( ( N == NormalizationType::group_wise ) );
                                    Error gradient_decent_normal( const Dim batch_size, const StepSize step_size );
                                    Error gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );
                                    Error gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );

        //: Operators
        public:
            inline T  get_mean              ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_variance          ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_beta              ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_gamma             ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_beta_jacobian     ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_beta_momentum     ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_beta_velocity     ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_gamma_jacobian    ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_gamma_momentum    ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_gamma_velocity    ( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise );
            inline T  get_mean              ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_variance          ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta              ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma             ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_jacobian     ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_momentum     ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_velocity     ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma_jacobian    ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma_momentum    ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma_velocity    ( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_mean              ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_variance          ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta              ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma             ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_jacobian     ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_momentum     ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_beta_velocity     ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma_jacobian    ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gama_momentum     ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T  get_gamma_velocity    ( const Idx index ) const                   requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
        private:
            inline T& get_mean              ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_variance          ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_beta              ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_gamma             ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_beta_jacobian     ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_beta_momentum     ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_beta_velocity     ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_gamma_jacobian    ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_gamma_momentum    ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_gamma_velocity    ( const Dim4D& indices )       requires ( N == NormalizationType::batch_wise );
            inline T& get_mean              ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_variance          ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta              ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma             ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_jacobian     ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_momentum     ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_velocity     ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma_jacobian    ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma_momentum    ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma_velocity    ( const Dim5D& indices )       requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_mean              ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_variance          ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta              ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma             ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_jacobian     ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_momentum     ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_beta_velocity     ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma_jacobian    ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gama_momentum     ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
            inline T& get_gamma_velocity    ( const Idx index )                         requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) );
    };

    //: Operators
        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_mean( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->mean[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_variance( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->variance[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_jacobian( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_momentum( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_velocity( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_jacobian( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_momentum( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_velocity( const Dim4D& indices ) const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_mean( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_variance( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->variance[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_jacobian( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_momentum( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_velocity( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_jacobian( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_momentum( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_velocity( const Dim5D& indices ) const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_mean( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_variance( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->variance[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_jacobian( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_jacobian[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_momentum( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_momentum[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_beta_velocity( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_velocity[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_jacobian( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_jacobian[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gama_momentum( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_momentum[ index ];
        }

        template <typename T, NormalizationType N>
        inline T NormalizationLayer<T, N>::get_gamma_velocity( const Idx index ) const requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_velocity[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_mean( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->mean[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_variance( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->variance[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->beta[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_jacobian( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_momentum( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_velocity( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_jacobian( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_momentum( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_velocity( const Dim4D& indices ) requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_mean( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_variance( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->variance[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_jacobian( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_momentum( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_velocity( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_jacobian( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_jacobian[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_momentum( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_momentum[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_velocity( const Dim5D& indices ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_velocity[ indices ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_mean( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_variance( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->variance[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_jacobian( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_jacobian[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_momentum( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_momentum[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_beta_velocity( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_velocity[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_jacobian( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_jacobian[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gama_momentum( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_momentum[ index ];
        }

        template <typename T, NormalizationType N>
        inline T& NormalizationLayer<T, N>::get_gamma_velocity( const Idx index ) requires ( ( N == NormalizationType::batch_wise) || ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_velocity[ index ];
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NORMALIZATION_HPP_
