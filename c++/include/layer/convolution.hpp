// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_CONVOLUTION_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_CONVOLUTION_HPP_

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

namespace nn
{
    enum class ConvolutionType : u8
    {
        down_sample,
        up_sample
    };

    enum class PaddingType : u8
    {
        zero,
        circular
    };

    enum class PaddingSize : u8
    {
        valid,
        same,
        full,
        custom
    };

    template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
    class ConvolutionLayer : public BaseLayer
    {
        //: Members
        public:
            static const ConvolutionType    conv_type = C;
            static const PaddingType        padding_type = P;
            static const PaddingSize        padding_size = S;
        protected:
            Shape<3>        input_dilation;
            Shape<3>        padding;
            Shape<3>        inv_padding;
            Shape<3>        stride;
            Shape<3>        dilation;
            Tensor<T, 4>    filter;
            Tensor<T, 4>    jacobian;
            Tensor<T, 4>    momentum;
            Tensor<T, 4>    velocity;

        //: Constructors
        public:
            ConvolutionLayer
            (
                const Shape<4>& filter_shape            = Shape<4>( 0 ),
                const Shape<3>& stride                  = Shape<3>( 0 ),
                const Shape<3>& dilation                = Shape<3>( 0 ),
                const T scalar                          = T()
            ) requires ( ( C == ConvolutionType::down_sample ) && ( S != PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape,
                const Shape<3>& stride,
                const Shape<3>& dilation,
                const std::vector<T> data
            ) requires ( ( C == ConvolutionType::down_sample ) && ( S != PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape            = Shape<4>( 0 ),
                const Shape<3>& padding                 = Shape<3>( 0 ),
                const Shape<3>& stride                  = Shape<3>( 0 ),
                const Shape<3>& dilation                = Shape<3>( 0 ),
                const T scalar                          = T()
            ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape,
                const Shape<3>& padding,
                const Shape<3>& stride,
                const Shape<3>& dilation,
                const std::vector<T> data
            ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape            = Shape<4>( 0 ),
                const Shape<3>& input_dilation          = Shape<3>( 0 ),
                const Shape<3>& stride                  = Shape<3>( 0 ),
                const Shape<3>& dilation                = Shape<3>( 0 ),
                const T scalar                          = T()
            ) requires ( ( C == ConvolutionType::up_sample ) && ( S != PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape,
                const Shape<3>& input_dilation,
                const Shape<3>& stride,
                const Shape<3>& dilation,
                const std::vector<T> data
            ) requires ( ( C == ConvolutionType::up_sample ) && ( S != PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape            = Shape<4>( 0 ),
                const Shape<3>& input_dilation          = Shape<3>( 0 ),
                const Shape<3>& padding                 = Shape<3>( 0 ),
                const Shape<3>& stride                  = Shape<3>( 0 ),
                const Shape<3>& dilation                = Shape<3>( 0 ),
                const T scalar                          = T()
            ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::custom ) );

            ConvolutionLayer
            (
                const Shape<4>& filter_shape,
                const Shape<3>& input_dilation,
                const Shape<3>& padding,
                const Shape<3>& stride,
                const Shape<3>& dilation,
                const std::vector<T> data
            ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::custom ) );

        //: Destructors
        public:
            virtual ~ConvolutionLayer();

        //: Methods
        public:
                                                void                reshape                     ( const Shape<4>& shape );
                                                void                resize                      ( const Shape<4>& shape );
                                                Shape<5>            calculate_output_shape      ( const Shape<5>& input_shape ) const requires ( C == ConvolutionType::down_sample );
                                                Shape<5>            calculate_output_shape      ( const Shape<5>& input_shape ) const requires ( C == ConvolutionType::up_sample );
                                                const Shape<4>&     get_shape                   () const;
                                                Size                get_size                    () const;
                                                const Tensor<T, 4>& get_filter                  () const;
                                                const Tensor<T, 4>& get_jacobian                () const;
                                                const Tensor<T, 4>& get_momentum                () const;
                                                const Tensor<T, 4>& get_velocity                () const;
                                                const Shape<3>&     get_stride                  () const;
                                                const Shape<3>&     get_padding                 () const;
                                                const Shape<3>&     get_dilation                () const;
                                                const Shape<3>&     get_input_dilation          () const;
                                                const Shape<3>&     get_inverse_padding         () const;
                                                void                set_training_mode           ( const TrainingMode training_mode );
                                                void                fill_filter                 ( const T value );
                                                void                zero_filter                 ();
                                                void                randomize_filter            ( const T min, const T max );
                                                void                initialize                  ( const Shape<5>& input_layer_shape, const Shape<5>& output_layer_shape, const InitializationType initialization, const DistributionType distribution );
            template <typename U, typename V>   Error                inference                  ( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const;
            template <typename U, typename V>   Error                backpropagation            ( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer );
                                                Error                gradient_decent_normal     ( const Dim batch_size, const StepSize step_size );
                                                Error                gradient_decent_momentum   ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );
                                                Error                gradient_decent_adam       ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );
        private:
                                                void                fill_jacobian               ( const T value );
                                                void                fill_momentum               ( const T value );
                                                void                fill_velocity               ( const T value );
                                                void                zero_jacobian               ();
                                                void                zero_momentum               ();
                                                void                zero_velocity               ();
                                                void                randomize_jacobian          ( const T min, const T max );
                                                void                randomize_momentum          ( const T min, const T max );
                                                void                randomize_velocity          ( const T min, const T max );
                                                inline Dim          stride_pad_dilation_dim     ( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim pad_dim_size, const Dim dilation_dim_size ) const;
                                                inline Dim          stride_dilation_dim         ( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim dilation_dim_size ) const;
                                                inline Dim          circular_dim                ( const Dim dim, const Dim dim_size ) const;
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::valid ) );
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero) );
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) );
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::valid ) );
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero) );
            template <typename U, typename V>   inline V            filter_inference            ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) );
            template <typename U, typename V>   inline T            filter_backpropagation      ( const NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer, const Idx in_b_idx, const Idx out_b_idx, const Dim filter_c_dim, const Dim filter_z_dim, const Dim filter_y_dim, const Dim filter_x_dim );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::down_sample ) && ( S == PaddingSize::valid ) );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::down_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero) );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( S == PaddingSize::valid ) );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::circular ) );
            template <typename U, typename V>   inline U            layer_backpropagation       ( const NodeLayer<V>& output_layer, const Idx out_b_idx, const Dim in_c_dim, const Dim in_z_dim, const Dim in_y_dim, const Dim in_x_dim ) requires ( ( C == ConvolutionType::up_sample ) && ( ( S == PaddingSize::same ) || ( S == PaddingSize::full ) || ( S == PaddingSize::custom ) ) && ( P == PaddingType::zero) );

    //: Operators
    public:
                                                inline T            get_filter                  ( const Dim4D& indices ) const;
                                                inline T            get_jacobian                ( const Dim4D& indices ) const;
                                                inline T            get_momentum                ( const Dim4D& indices ) const;
                                                inline T            get_velocity                ( const Dim4D& indices ) const;
                                                inline T            get_filter                  ( const Idx index ) const;
                                                inline T            get_jacobian                ( const Idx index ) const;
                                                inline T            get_momentum                ( const Idx index ) const;
                                                inline T            get_velocity                ( const Idx index ) const;
                                                inline T&           get_filter                  ( const Dim4D& indices );
                                                inline T&           get_filter                  ( const Idx index );
    private:
                                                inline T&           get_jacobian                ( const Dim4D& indices );
                                                inline T&           get_momentum                ( const Dim4D& indices );
                                                inline T&           get_velocity                ( const Dim4D& indices );
                                                inline T&           get_jacobian                ( const Idx index );
                                                inline T&           get_momentum                ( const Idx index );
                                                inline T&           get_velocity                ( const Idx index );
    };

    //: Inline Operators
        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_filter( const Dim4D& indices ) const
        {
            return this->filter[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_jacobian( const Dim4D& indices ) const
        {
            return this->jacobian[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_momentum( const Dim4D& indices ) const
        {
            return this->momentum[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_velocity( const Dim4D& indices ) const
        {
            return this->velocity[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_filter( const Idx index ) const
        {
            return this->filter[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_jacobian( const Idx index ) const
        {
            return this->jacobian[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_momentum( const Idx index ) const
        {
            return this->momentum[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T ConvolutionLayer<T, C, P, S>::get_velocity( const Idx index ) const
        {
            return this->velocity[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_filter( const Dim4D& indices )
        {
            return this->filter[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_jacobian( const Dim4D& indices )
        {
            return this->jacobian[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_momentum( const Dim4D& indices )
        {
            return this->momentum[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_velocity( const Dim4D& indices )
        {
            return this->velocity[ indices ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_filter( const Idx index )
        {
            return this->filter[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_jacobian( const Idx index )
        {
            return this->jacobian[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_momentum( const Idx index )
        {
            return this->momentum[ index ];
        }

        template <typename T, ConvolutionType C, PaddingType P, PaddingSize S>
        inline T& ConvolutionLayer<T, C, P, S>::get_velocity( const Idx index )
        {
            return this->velocity[ index ];
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_CONVOLUTION_HPP_
