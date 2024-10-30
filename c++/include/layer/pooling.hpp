// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_POOLING_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_POOLING_HPP_

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
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

namespace nn
{
    enum class PoolingType : u8
    {
        max,
        average
    };

    template <PoolingType P>
    class PoolingLayer : public BaseLayer
    {
        //: Members
        public:
            static const PoolingType type = P;
        protected:
            Shape<3>    pool;
            Shape<3>    stride;
            Shape<3>    dilation;

        //: Constructors
        public:
            PoolingLayer
            (
                const Shape<3>& shape       = Shape<3>( 0 ),
                const Shape<3>& stride      = Shape<3>( 0 ),
                const Shape<3>& dilation    = Shape<3>( 0 )
            );

        //: Destructors
        public:
            virtual ~PoolingLayer();

        //: Methods
        public:
            void                                            reshape                         ( const Shape<3>& shape );
            void                                            resize                          ( const Shape<3>& shape );
            void                                            set_stride                      ( const Shape<3>& stride );
            void                                            set_dilation                    ( const Shape<3>& dilation );
            Shape<5>                                        calculate_output_shape          ( const Shape<5>& input_shape ) const;
            const Shape<3>&                                 get_shape                       () const;
            const Shape<3>&                                 get_stride                      () const;
            const Shape<3>&                                 get_dilation                    () const;
            template <typename U, typename V> Error          inference                       ( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const requires ( P == PoolingType::max );
            template <typename U, typename V> Error          inference                       ( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const requires ( P == PoolingType::average );
            template <typename U, typename V> Error          backpropagation                 ( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const requires ( P == PoolingType::max );
            template <typename U, typename V> Error          backpropagation                 ( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const requires ( P == PoolingType::average );
        private:
            inline Dim                                      stride_dilation_dim             ( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim dilation_dim_size ) const;
            template <typename U> inline U                  pooling_window                  ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::max );
            template <typename U> inline U                  pooling_window                  ( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::average );
            template <typename U> inline Idx                pooling_window_backpropagation  ( NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::max );
            template <typename U, typename V> inline void   pooling_window_backpropagation  ( NodeLayer<U>& input_layer, const V scaled_output_delta, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::average );
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_POOLING_HPP_
