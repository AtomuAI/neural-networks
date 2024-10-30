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
#include <functional>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"
#include "bewusstsein_neural_networks/c++/include/core/math.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/normalization.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T, NormalizationType N>
        NormalizationLayer<T, N>::NormalizationLayer( const u8 channels ) requires ( N == NormalizationType::batch_wise ) :
            BaseLayer( LayerType::normalization_layer ), mean( Shape<4>( 1, 1, 1, channels ) ), variance( Shape<4>( 1, 1, 1, channels ) ), gamma( Shape<4>( 1, 1, 1, channels ) ), beta( Shape<4>( 1, 1, 1, channels ) ) {}

        template <typename T, NormalizationType N>
        NormalizationLayer<T, N>::NormalizationLayer( const u8 batches ) requires ( N == NormalizationType::layer_wise ) :
            BaseLayer( LayerType::normalization_layer ), mean( Shape<5>( 1, 1, 1, 1, batches ) ), variance( Shape<5>( 1, 1, 1, 1, batches ) ), gamma( Shape<5>( 1, 1, 1, 1, batches ) ), beta( Shape<5>( 1, 1, 1, 1, batches ) ) {}

        template <typename T, NormalizationType N>
        NormalizationLayer<T, N>::NormalizationLayer( const u8 channels, const u8 batches ) requires ( N == NormalizationType::instance_wise ) :
            BaseLayer( LayerType::normalization_layer ), mean( Shape<5>( 1, 1, 1, channels, batches  ) ), variance( Shape<5>( 1, 1, 1, channels, batches ) ), gamma( Shape<5>( 1, 1, 1, channels, batches ) ), beta( Shape<5>( 1, 1, 1, channels, batches ) ) {}

        template <typename T, NormalizationType N>
        NormalizationLayer<T, N>::NormalizationLayer( const u8 channels, const u8 batches, const u8 group_size ) requires ( N == NormalizationType::group_wise ) :
            BaseLayer( LayerType::normalization_layer ), mean( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) ), variance( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) ), gamma( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) ), beta( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) ) {}

    //: Destructors
        template <typename T, NormalizationType N>
        NormalizationLayer<T, N>::~NormalizationLayer() {}

    //: Methods
        template <typename T, NormalizationType N>
        const Shape<4>& NormalizationLayer<T, N>::get_shape() const requires ( N == NormalizationType::batch_wise )
        {
            return this->mean.get_shape();
        }

        template <typename T, NormalizationType N>
        const Shape<5>& NormalizationLayer<T, N>::get_shape() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean.get_shape();
        }

        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_mean() const requires ( N == NormalizationType::batch_wise )
        {
            return this->mean;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_variance() const requires ( N == NormalizationType::batch_wise )
        {
            return this->variance;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_beta() const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_gamma() const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_beta_jacobian() const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_jacobian;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_gamma_jacobian() const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_jacobian;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_beta_momentum() const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_momentum;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_gamma_momentum() const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_momentum;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_beta_velocity() const requires ( N == NormalizationType::batch_wise )
        {
            return this->beta_velocity;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 4>& NormalizationLayer<T, N>::get_gamma_velocity() const requires ( N == NormalizationType::batch_wise )
        {
            return this->gamma_velocity;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_mean() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->mean;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_variance() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->variance;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_beta() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_gamma() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_beta_jacobian() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_jacobian;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_gamma_jacobian() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_jacobian;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_beta_momentum() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_momentum;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_gamma_momentum() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_momentum;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_beta_velocity() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->beta_velocity;
        }
        template <typename T, NormalizationType N>
        const Tensor<T, 5>& NormalizationLayer<T, N>::get_gamma_velocity() const requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            return this->gamma_velocity;
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::reshape( const u8 channels ) requires ( N == NormalizationType::batch_wise )
        {
            this->variance.reshape( Shape<4>( 1, 1, 1, channels ) );
            this->mean.reshape( Shape<4>( 1, 1, 1, channels ) );
            this->BaseLayer::reshape( Shape<4>( 1, 1, 1, channels ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::reshape( Shape<4>( 1, 1, 1, channels ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::reshape( const u8 batches ) requires ( N == NormalizationType::layer_wise )
        {
            this->variance.reshape( Shape<5>( 1, 1, 1, 1, batches ) );
            this->mean.reshape( Shape<5>( 1, 1, 1, 1, batches ) );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, 1, batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, 1, batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::reshape( const u8 channels, const u8 batches ) requires ( N == NormalizationType::instance_wise )
        {
            this->variance.reshape( Shape<5>( 1, 1, 1, channels, batches ) );
            this->mean.reshape( Shape<5>( 1, 1, 1, channels, batches ) );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, channels, batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, channels, batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::reshape( const u8 channels, const u8 batches, const u8 group_size ) requires ( N == NormalizationType::group_wise )
        {
            this->variance.reshape( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) );
            this->mean.reshape( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::reshape( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::resize( const u8 channels ) requires ( N == NormalizationType::batch_wise )
        {
            this->variance.resize( Shape<4>( 1, 1, 1, channels ) );
            this->mean.resize( Shape<4>( 1, 1, 1, channels ) );
            this->BaseLayer::resize( Shape<4>( 1, 1, 1, channels ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::resize( Shape<4>( 1, 1, 1, channels ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::resize( const u8 batches ) requires ( N == NormalizationType::layer_wise )
        {
            this->variance.resize( Shape<5>( 1, 1, 1, 1, batches ) );
            this->mean.resize( Shape<5>( 1, 1, 1, 1, batches ) );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, 1, batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, 1, batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::resize( const u8 channels, const u8 batches ) requires ( N == NormalizationType::instance_wise )
        {
            this->variance.resize( Shape<5>( 1, 1, 1, channels, batches ) );
            this->mean.resize( Shape<5>( 1, 1, 1, channels, batches ) );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, channels, batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, channels, batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::resize( const u8 channels, const u8 batches, const u8 group_size ) requires ( N == NormalizationType::group_wise )
        {
            this->variance.resize( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) );
            this->mean.resize( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ) );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ), this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity );
            this->BaseLayer::resize( Shape<5>( 1, 1, 1, ( channels / group_size ), batches ), this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::set_training_mode( TrainingMode training_mode ) requires ( N == NormalizationType::batch_wise )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( beta, beta_jacobian, beta_momentum, beta_velocity );
            this->BaseLayer::allocate_training_memory( gamma, gamma_jacobian, gamma_momentum, gamma_velocity );
        }

        template <typename T, NormalizationType N>
        void NormalizationLayer<T, N>::set_training_mode( TrainingMode training_mode ) requires ( ( N == NormalizationType::layer_wise ) || ( N == NormalizationType::instance_wise ) || ( N == NormalizationType::group_wise ) )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( beta, beta_jacobian, beta_momentum, beta_velocity );
            this->BaseLayer::allocate_training_memory( gamma, gamma_jacobian, gamma_momentum, gamma_velocity );
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::stat_analysis( const NodeLayer<U>& layer ) requires ( N == NormalizationType::batch_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            Size layer_norm_size = layer_spacial_size * layer_b_size;

            for ( Dim channel = 0; channel < layer_c_size; channel++ )
            {
                T mean = 0;
                for ( Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    Idx batch_idx = batch * layer_c_size;
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        mean += layer.get_node( index );
                    }
                }

                T variance = 0;
                for ( Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    Idx batch_idx = batch * layer_c_size;
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T diff = layer.get_node( index ) - mean;
                        variance += diff * diff;
                    }
                }

                uint32_t step = this->get_time_step();
                float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_norm_size ) ) / step;
                float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_norm_size ) ) / step;

                this->get_mean( channel )       = this->get_mean( channel )     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                this->get_variance( channel )   = this->get_variance( channel ) * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::stat_analysis( const NodeLayer<U>& layer ) requires ( N == NormalizationType::layer_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            Size layer_norm_size = layer_spacial_size * layer_c_size;

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = 0;
                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        mean += layer.get_node( index );
                    }
                }

                T variance = 0;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T diff = layer.get_node( index ) - mean;
                        variance += diff * diff;
                    }
                }

                uint32_t step = this->get_time_step();
                float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_norm_size ) ) / step;
                float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_norm_size ) ) / step;

                this->get_mean( batch )       = this->get_mean( batch )     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                this->get_variance( batch )   = this->get_variance( batch ) * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::stat_analysis( const NodeLayer<U>& layer ) requires ( N == NormalizationType::instance_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    T mean = 0;
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        mean += layer.get_node( index );
                    }

                    T variance = 0;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T diff = layer.get_node( index ) - mean;
                        variance += diff * diff;
                    }

                    uint32_t step = this->get_time_step();
                    float stepMinusOneOverStep              = static_cast<float>( step - 1 ) / step;
                    float meanOverLayerNormSizeOverStep     = ( mean / static_cast<float>( layer_spacial_size ) ) / step;
                    float varianceOverLayerNormSizeOverStep = ( variance / static_cast<float>( layer_spacial_size ) ) / step;

                    this->get_mean( channel_idx )       = this->get_mean( channel_idx )     * stepMinusOneOverStep + meanOverLayerNormSizeOverStep;
                    this->get_variance( channel_idx )   = this->get_variance( channel_idx ) * stepMinusOneOverStep + varianceOverLayerNormSizeOverStep;
                }
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::stat_analysis( const NodeLayer<U>& layer ) requires ( N == NormalizationType::group_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }
            // TO DO
            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::inference( NodeLayer<U>& layer ) const requires ( N == NormalizationType::batch_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            for ( Dim channel = 0; channel < layer_c_size; channel++ )
            {
                T mean = this->get_mean( channel );
                T variance = this->get_variance( channel );
                T inv_standard_deviation = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon() );
                T beta = this->get_beta( channel );
                T gamma = this->get_gamma( channel );

                for ( Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    Idx batch_idx = batch * layer_c_size;
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;

                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T normalized = ( layer.get_node( index ) - mean ) * inv_standard_deviation;
                        layer.get_node( index ) = ( normalized * gamma ) + beta;
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::inference( NodeLayer<U>& layer ) const requires ( N == NormalizationType::layer_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = this->get_mean( batch );
                T variance = this->get_variance( batch );
                T inv_standard_deviation = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon() );
                T beta = this->get_beta( batch );
                T gamma = this->get_gamma( batch );

                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T normalized = ( layer.get_node( index ) - mean ) * inv_standard_deviation;
                        layer.get_node( index ) = ( normalized * gamma ) + beta;
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::inference( NodeLayer<U>& layer ) const requires ( N == NormalizationType::instance_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;

                    T mean = this->get_mean( channel_idx );
                    T variance = this->get_variance( channel_idx );
                    T inv_standard_deviation = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon() );
                    T beta = this->get_beta( channel_idx );
                    T gamma = this->get_gamma( channel_idx );

                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        T normalized = ( layer.get_node( index ) - mean ) * inv_standard_deviation;
                        layer.get_node( index ) = ( normalized * gamma ) + beta;
                    }
                }
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::inference( NodeLayer<U>& layer ) const requires ( N == NormalizationType::group_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }
            // TODO( Shenmarukai ):
            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::backpropagation( NodeLayer<U>& layer ) requires ( N == NormalizationType::batch_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            Size layer_norm_size = layer_spacial_size * layer_b_size;

            for ( Dim channel = 0; channel < layer_c_size; channel++ )
            {
                T mean = this->get_mean( channel );
                T variance = this->get_variance( channel );
                T cube_root_variance = nn::pow( variance + std::numeric_limits<T>::epsilon(), -1.5 );
                T inv_sqrt_variance = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon());
                T beta = this->get_beta( channel );
                T gamma = this->get_gamma( channel );

                T beta_gradient( 0 );
                T gamma_gradient( 0 );

                for ( Dim batch = 0; batch < layer_b_size; batch++ )
                {
                    Idx batch_idx = batch * layer_c_size;
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        U delta_node = layer.get_delta( index );

                        T node_minus_mean = layer.get_node( index ) - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = T( -0.5 ) * dnormalized * normalized * cube_root_variance;
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

                this->get_beta_jacobian( channel ) += beta_gradient / layer_norm_size;
                this->get_gamma_jacobian( channel ) += gamma_gradient / layer_norm_size;
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::backpropagation( NodeLayer<U>& layer ) requires ( N == NormalizationType::layer_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            Size layer_norm_size = layer_spacial_size * layer_c_size;

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                T mean = this->get_mean( batch );
                T variance = this->get_variance( batch );
                T cube_root_variance = nn::pow( variance + std::numeric_limits<T>::epsilon(), -1.5 );
                T inv_sqrt_variance = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon() );
                T beta = this->get_beta( batch );
                T gamma = this->get_gamma( batch );

                T beta_gradient( 0 );
                T gamma_gradient( 0 );

                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;
                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        U delta_node = layer.get_delta( index );

                        T node_minus_mean = layer.get_node( index ) - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = T( -0.5 ) * dnormalized * normalized * cube_root_variance;
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

                this->get_beta_jacobian( batch ) += beta_gradient / layer_norm_size;
                this->get_gamma_jacobian( batch ) += gamma_gradient / layer_norm_size;
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::backpropagation( NodeLayer<U>& layer ) requires ( N == NormalizationType::instance_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            Size layer_spacial_size = layer_shape.distance( 0, 4 );

            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }

            Dim layer_b_size = layer_shape.batches();
            Dim layer_c_size = layer_shape.channels();

            for ( Dim batch = 0; batch < layer_b_size; batch++ )
            {
                Idx batch_idx = batch * layer_c_size;
                for ( Dim channel = 0; channel < layer_c_size; channel++ )
                {
                    Idx channel_idx = ( batch_idx + channel ) * layer_spacial_size;

                    T mean = this->get_mean( channel_idx );
                    T variance = this->get_variance( channel_idx );
                    T cube_root_variance = nn::pow( variance + std::numeric_limits<T>::epsilon(), -1.5 );
                    T inv_sqrt_variance = T( 1 ) / nn::sqrt( variance + std::numeric_limits<T>::epsilon() );
                    T beta = this->get_beta( channel_idx );
                    T gamma = this->get_gamma( channel_idx );

                    T beta_gradient( 0 );
                    T gamma_gradient( 0 );

                    for ( Idx spacial = 0; spacial < layer_spacial_size; spacial++ )
                    {
                        Idx index = channel_idx + spacial;

                        U delta_node = layer.get_delta( index );

                        T node_minus_mean = layer.get_node( index ) - mean;
                        T normalized = node_minus_mean * inv_sqrt_variance;
                        T dnormalized = delta_node * gamma;
                        T dvariance = T( -0.5 ) * dnormalized * normalized * cube_root_variance;
                        T dmean = -dnormalized * inv_sqrt_variance - 2.0 * dvariance * node_minus_mean / layer_spacial_size;

                        // Calculate gradient for beta
                        beta_gradient += delta_node;

                        // Calculate gradient for gamma
                        gamma_gradient += delta_node * normalized;

                        // Backpropagate through scaling and offset
                        delta_node = dnormalized * inv_sqrt_variance + dvariance * 2.0 * node_minus_mean / layer_spacial_size + dmean / layer_spacial_size;
                        layer.get_delta( index ) = delta_node;
                    }

                    this->get_beta_jacobian( channel_idx ) += beta_gradient / layer_spacial_size;
                    this->get_gamma_jacobian( channel_idx ) += gamma_gradient / layer_spacial_size;
                }
            }

            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        template <typename U>
        Error NormalizationLayer<T, N>::backpropagation( NodeLayer<U>& layer ) requires ( N == NormalizationType::group_wise )
        {
            const Shape<5> layer_shape = layer.get_shape();
            if ( layer_shape.channels() != this->mean.get_shape().channels() ) { return Error::MISMATCHED_SHAPES; }
            if ( layer_shape.batches() != this->mean.get_shape().batches() ) { return Error::MISMATCHED_SHAPES; }
            // TODO( Shenmarukai ):
            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        Error NormalizationLayer<T, N>::gradient_decent_normal( const Dim batch_size, const StepSize step_size )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->beta, this->beta_jacobian, batch_size, step_size );
            this->BaseLayer::gradient_decent_normal( this->gamma, this->gamma_jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        Error NormalizationLayer<T, N>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->beta, this->beta_jacobian, this->beta_momentum, batch_size, step_size, momentum_step_size );
            this->BaseLayer::gradient_decent_momentum( this->gamma, this->gamma_jacobian, this->gamma_momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T, NormalizationType N>
        Error NormalizationLayer<T, N>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->beta, this->beta_jacobian, this->beta_momentum, this->beta_velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->BaseLayer::gradient_decent_adam( this->gamma, this->gamma_jacobian, this->gamma_momentum, this->gamma_velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacro( type_a, normalization_type, type_b )\
        template Error NormalizationLayer<type_a, normalization_type>::inference( NodeLayer<type_b>& layer ) const;\
        template Error NormalizationLayer<type_a, normalization_type>::backpropagation( NodeLayer<type_b>& layer );

    #define ClassMacroC( type_a, normalization_type )\
        template class NormalizationLayer<type_a, normalization_type>;\
        FunctionMacro( type_a, normalization_type, i32 )\
        FunctionMacro( type_a, normalization_type, i64 )\
        FunctionMacro( type_a, normalization_type, f32 )\
        FunctionMacro( type_a, normalization_type, f64 )

    #define ClassMacroB( type_a )\
        ClassMacroC( type_a, NormalizationType::batch_wise )\
        ClassMacroC( type_a, NormalizationType::group_wise )\
        ClassMacroC( type_a, NormalizationType::instance_wise )\
        ClassMacroC( type_a, NormalizationType::layer_wise )

    #define ClassMacro\
        ClassMacroB( i32 )\
        ClassMacroB( i64 )\
        ClassMacroB( f32 )\
        ClassMacroB( f64 )

    ClassMacro

    #undef FunctionMacro
    #undef ClassMacroC
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn
