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

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/pooling.hpp"

namespace nn
{
    //: Constructors
        template <PoolingType P>
        PoolingLayer<P>::PoolingLayer( const Shape<3>& pool, const Shape<3>& stride, const Shape<3>& dilation ) :
            BaseLayer( LayerType::pooling_layer ), pool( pool ), stride( stride ), dilation( dilation )
        {}

    //: Destructors
        template <PoolingType P>
        PoolingLayer<P>::~PoolingLayer() {}

    //: Methods
        template <PoolingType P>
        void PoolingLayer<P>::reshape( const Shape<3>& shape )
        {
            this->pool.reshape( shape.width(), shape.height(), shape.depth() );
        }

        template <PoolingType P>
        void PoolingLayer<P>::resize( const Shape<3>& shape )
        {
            this->pool.resize( shape.width(), shape.height(), shape.depth() );
        }

        template <PoolingType P>
        void PoolingLayer<P>::set_stride( const Shape<3>& stride )
        {
            this->stride = stride;
        }

        template <PoolingType P>
        void PoolingLayer<P>::set_dilation( const Shape<3>& dilation )
        {
            this->dilation = dilation;
        }

        template <PoolingType P>
        Shape<5> PoolingLayer<P>::calculate_output_shape( const Shape<5>& input_shape ) const
        {
            Shape<5> shape
            (
                ( ( input_shape.width() - ( this->dilation.width() * ( this->pool.width() - 1 ) ) - 1 ) / this->stride.width() ) + 1,
                ( ( input_shape.height() - ( this->dilation.height() * ( this->pool.height() - 1 ) ) - 1 ) / this->stride.height() ) + 1,
                ( ( input_shape.depth() - ( this->dilation.depth() * ( this->pool.depth() - 1 ) ) - 1 ) / this->stride.depth() ) + 1,
                input_shape.channels(),
                input_shape.batches()
            );

            return shape;
        }

        template <PoolingType P>
        const Shape<3>& PoolingLayer<P>::get_shape() const
        {
            return this->pool;
        }

        template <PoolingType P>
        const Shape<3>& PoolingLayer<P>::get_stride() const
        {
            return this->stride;
        }

        template <PoolingType P>
        const Shape<3>& PoolingLayer<P>::get_dilation() const
        {
            return this->dilation;
        }

        template <PoolingType P>
        inline Dim PoolingLayer<P>::stride_dilation_dim( const Dim in_dim, const Dim out_dim, const Dim stride_dim_size, const Dim dilation_dim_size ) const
        {
            return in_dim * stride_dim_size + out_dim * dilation_dim_size;
        }

        template <PoolingType P>
        template <typename U>
        inline U PoolingLayer<P>::pooling_window( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::max )
        {
            const Shape<5> input_shape = input_layer.get_shape();

            const Shape<3> pool_shape = this->get_shape();
            const Shape<3> stride_shape = this->get_stride();
            const Shape<3> dilation_shape = this->get_dilation();

            U max_value = std::numeric_limits<U>::lowest();

            Dim in_c_dim = out_c_dim % input_shape.channels();
            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim pool_z_dim = 0; pool_z_dim < pool_shape.depth(); ++pool_z_dim )
            {
                Idx in_z_dim = this->stride_dilation_dim( out_z_dim, pool_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool in_z_bound = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim pool_y_dim = 0; pool_y_dim < pool_shape.height(); ++pool_y_dim )
                {
                    Idx in_y_dim = this->stride_dilation_dim( out_y_dim, pool_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool in_y_bound = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim pool_x_dim = 0; pool_x_dim < pool_shape.width(); ++pool_x_dim )
                    {
                        Idx in_x_dim = this->stride_dilation_dim( out_x_dim, pool_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool in_bound = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                        U current_value = input_layer.get_node( in_index );
                        if ( current_value > max_value )
                        {
                            max_value = current_value;
                        }
                    }
                }
            }

            return max_value;
        }

        template <PoolingType P>
        template <typename U>
        inline U PoolingLayer<P>::pooling_window( const NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::average )
        {
            const Shape<5> input_shape = input_layer.get_shape();

            const Shape<3> pool_shape = this->get_shape();
            const Shape<3> stride_shape = this->get_stride();
            const Shape<3> dilation_shape = this->get_dilation();

            U sum_value( 0 );

            Dim in_c_dim = out_c_dim % input_shape.channels();
            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim pool_z_dim = 0; pool_z_dim < pool_shape.depth(); ++pool_z_dim )
            {
                Idx in_z_dim = this->stride_dilation_dim( out_z_dim, pool_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool in_z_bound = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim pool_y_dim = 0; pool_y_dim < pool_shape.height(); ++pool_y_dim )
                {
                    Idx in_y_dim = this->stride_dilation_dim( out_y_dim, pool_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool in_y_bound = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim pool_x_dim = 0; pool_x_dim < pool_shape.width(); ++pool_x_dim )
                    {
                        Idx in_x_dim = this->stride_dilation_dim( out_x_dim, pool_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool in_bound = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                        sum_value += input_layer.get_node( in_index );
                    }
                }
            }

            return sum_value;
        }

        template <PoolingType P>
        template <typename U>
        inline Idx PoolingLayer<P>::pooling_window_backpropagation( NodeLayer<U>& input_layer, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::max )
        {
            const Shape<5>& input_shape = input_layer.get_shape();

            const Shape<3> pool_shape = this->get_shape();
            const Shape<3> stride_shape = this->get_stride();
            const Shape<3> dilation_shape = this->get_dilation();

            Idx max_index = 0;
            U max_value = input_layer.get_node( max_index );

            Dim in_c_dim = out_c_dim % input_shape.channels();
            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim pool_z_dim = 0; pool_z_dim < pool_shape.depth(); ++pool_z_dim )
            {
                Idx in_z_dim = this->stride_dilation_dim( out_z_dim, pool_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool in_z_bound = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim pool_y_dim = 0; pool_y_dim < pool_shape.height(); ++pool_y_dim )
                {
                    Idx in_y_dim = this->stride_dilation_dim( out_y_dim, pool_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool in_y_bound = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim pool_x_dim = 0; pool_x_dim < pool_shape.width(); ++pool_x_dim )
                    {
                        Idx in_x_dim = this->stride_dilation_dim( out_x_dim, pool_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool in_bound = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                        U current_value = input_layer.get_node( in_index );

                        if ( current_value > max_value )
                        {
                            max_index = in_index;
                            max_value = current_value;
                            input_layer.get_delta( in_index ) = U( 0 );
                        }
                    }
                }
            }

            return max_index;
        }

        template <PoolingType P>
        template <typename U, typename V>
        inline void PoolingLayer<P>::pooling_window_backpropagation( NodeLayer<U>& input_layer, const V scaled_output_delta, const Idx in_b_idx, const Dim out_c_dim, const Dim out_z_dim, const Dim out_y_dim, const Dim out_x_dim ) const requires ( P == PoolingType::average )
        {
            const Shape<5> input_shape = input_layer.get_shape();

            const Shape<3> pool_shape = this->get_shape();
            const Shape<3> stride_shape = this->get_stride();
            const Shape<3> dilation_shape = this->get_dilation();

            Dim in_c_dim = out_c_dim % input_shape.channels();
            bool in_c_bound = input_shape.within_channels( in_c_dim );
            Idx in_c_idx = input_shape.channel_index( in_b_idx, in_c_dim );

            for ( Dim pool_z_dim = 0; pool_z_dim < pool_shape.depth(); ++pool_z_dim )
            {
                Idx in_z_dim = this->stride_dilation_dim( out_z_dim, pool_z_dim, stride_shape.depth(), dilation_shape.depth() );
                bool in_z_bound = input_shape.within_depth( in_z_dim ) && in_c_bound;
                Idx in_z_idx = input_shape.depth_index( in_c_idx, in_z_dim );

                for ( Dim pool_y_dim = 0; pool_y_dim < pool_shape.height(); ++pool_y_dim )
                {
                    Idx in_y_dim = this->stride_dilation_dim( out_y_dim, pool_y_dim, stride_shape.height(), dilation_shape.height() );
                    bool in_y_bound = input_shape.within_height( in_y_dim ) && in_z_bound;
                    Idx in_y_idx = input_shape.height_index( in_z_idx, in_y_dim );

                    for ( Dim pool_x_dim = 0; pool_x_dim < pool_shape.width(); ++pool_x_dim )
                    {
                        Idx in_x_dim = this->stride_dilation_dim( out_x_dim, pool_x_dim, stride_shape.width(), dilation_shape.width() );
                        bool in_bound = input_shape.within_width( in_x_dim ) && in_y_bound;
                        Idx in_index = input_shape.width_index( in_y_idx, in_x_dim );

                        input_layer.get_delta( in_index ) = scaled_output_delta;
                    }
                }
            }
        }

        template <PoolingType P>
        template <typename U, typename V>
        Error PoolingLayer<P>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const requires ( P == PoolingType::max )
        {
            const Shape<5> output_shape = output_layer.get_shape();
            const Shape<5> input_shape = input_layer.get_shape();

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
                                Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                                output_layer.get_node( out_index ) = this->pooling_window( input_layer, in_b_idx, out_c_dim, out_z_dim, out_y_dim, out_x_dim );
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }

        template <PoolingType P>
        template <typename U, typename V>
        Error PoolingLayer<P>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const requires ( P == PoolingType::average )
        {
            const Shape<5> output_shape = output_layer.get_shape();
            const Shape<5> input_shape = input_layer.get_shape();

            if ( output_shape != this->calculate_output_shape( input_shape ) ) { return Error::MISMATCHED_SHAPES; }

            const Shape<3> pool_shape = this->get_shape();

            const float pooling_scalar = 1.0f / ( pool_shape.width() * pool_shape.height() * pool_shape.depth() );

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
                                Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                                output_layer.get_node( out_index ) = this->pooling_window( input_layer, in_b_idx, out_c_dim, out_z_dim, out_y_dim, out_x_dim ) * pooling_scalar;
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }

        template <PoolingType P>
        template <typename U, typename V>
        Error PoolingLayer<P>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const requires ( P == PoolingType::max )
        {
            const Shape<5> output_shape = output_layer.get_shape();
            const Shape<5> input_shape = input_layer.get_shape();

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
                                Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                                Idx max_index = this->pooling_window_backpropagation( input_layer, in_b_idx, out_c_dim, out_z_dim, out_y_dim, out_x_dim );
                                input_layer.get_delta( max_index ) = output_layer.get_delta( out_index );
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }

        template <PoolingType P>
        template <typename U, typename V>
        Error PoolingLayer<P>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer ) const requires ( P == PoolingType::average )
        {
            const Shape<5> output_shape = output_layer.get_shape();
            const Shape<5> input_shape = input_layer.get_shape();

            if ( output_shape != this->calculate_output_shape( input_shape ) ) { return Error::MISMATCHED_SHAPES; }

            const Shape<3> pool_shape = this->get_shape();

            float pooling_scalar = 1.0f / ( pool_shape.width() * pool_shape.height() * pool_shape.depth() );

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
                                Idx out_index = output_shape.width_index( out_y_idx, out_x_dim );

                                V out_delta = output_layer.get_delta( out_index );
                                this->pooling_window_backpropagation( input_layer, ( out_delta * pooling_scalar ), in_b_idx, out_c_dim, out_z_dim, out_y_dim, out_x_dim );
                            }
                        }
                    }
                }
            }

            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacroB( pooling_type, type_a, type_b )\
        template Error PoolingLayer<pooling_type>::inference( const NodeLayer<type_a>& input_layer, NodeLayer<type_b>& output_layer ) const;\
        template Error PoolingLayer<pooling_type>::backpropagation( NodeLayer<type_a>& input_layer, const NodeLayer<type_b>& output_layer ) const;

    #define FunctionMacro( pooling_type, type_a )\
        FunctionMacroB( pooling_type, type_a, i32 )\
        FunctionMacroB( pooling_type, type_a, i64 )\
        FunctionMacroB( pooling_type, type_a, f32 )\
        FunctionMacroB( pooling_type, type_a, f64 )

    #define ClassMacroB( pooling_type )\
        template class PoolingLayer<pooling_type>;\
        FunctionMacro( pooling_type, i32 )\
        FunctionMacro( pooling_type, i64 )\
        FunctionMacro( pooling_type, f32 )\
        FunctionMacro( pooling_type, f64 )

    #define ClassMacro\
        ClassMacroB( PoolingType::average )\
        ClassMacroB( PoolingType::max )

    ClassMacro

    #undef FunctionMacroB
    #undef FunctionMacro
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn
