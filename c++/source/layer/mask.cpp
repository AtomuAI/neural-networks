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

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/mask.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        MaskLayer::MaskLayer( const Shape<4>& shape, const bool scalar ) :
            BaseLayer( LayerType::mask_layer ), mask( shape, scalar ) {}

        MaskLayer::MaskLayer( const LayerType type, const Shape<4>& shape, const bool scalar ) :
            BaseLayer( type ), mask( shape, scalar ) {}

    //: Destructors
        MaskLayer::~MaskLayer() {}

    //: Methods
        void MaskLayer::reshape( const Shape<4>& shape )
        {
            this->mask.reshape( shape );
        }

        void MaskLayer::resize( const Shape<4>& shape )
        {
            this->mask.resize( shape );
        }

        const Shape<4>& MaskLayer::get_shape() const
        {
            return this->mask.get_shape();
        }

        Size MaskLayer::get_size() const
        {
            return this->mask.get_size();
        }

        const Tensor<bool, 4>& MaskLayer::get_mask() const
        {
            return this->mask;
        }

        void MaskLayer::reshape( const Shape<4>& shape )
        {
            this->mask.reshape( shape );
        }

        void MaskLayer::fill( const bool value )
        {
            this->mask.fill( value );
        }

        template <typename U>
        Error MaskLayer::inference( NodeLayer<U>& layer ) const
        {
            Shape<4> shape = layer.get_shape();
            Size spacial_size = shape.distance( 0, 4 );

            if ( spacial_size != this->mask.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim batch = 0; batch < shape.batches(); batch++ )
            {
                Index batch_index = batch * spacial_size;
                for ( Index spacial = 0; spacial < spacial_size; spacial++ )
                {
                    Index index = batch_index + spacial;
                    layer.get_node( index ) += layer.get_node( index ) * this->get_mask( spacial );
                }
            }

            return Error::NONE;
        }

        template <typename U>
        Error MaskLayer::backpropagation( NodeLayer<U>& layer ) const
        {
            Shape<4> shape = layer.get_shape();
            Size spacial_size = shape.distance( 0, 4 );

            if ( spacial_size != this->mask.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }

            for ( Dim batch = 0; batch < shape.batches(); batch++ )
            {
                Index batch_index = batch * spacial_size;
                for ( Index spacial = 0; spacial < spacial_size; spacial++ )
                {
                    Index index = batch_index + spacial;
                    layer.get_delta( index ) += layer.get_delta( index ) * this->get_mask( spacial );
                }
            }

            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacro( type )\
        template Error MaskLayer::inference( NodeLayer<type>& layer ) const;\
        template Error MaskLayer::backpropagation( NodeLayer<type>& layer ) const;

    #define ClassMacro\
        FunctionMacro( bool )\
        FunctionMacro( i32 )\
        FunctionMacro( i64 )\
        FunctionMacro( f32 )\
        FunctionMacro( f64 )

    ClassMacro

    #undef FunctionMacro
    #undef ClassMacro
} // namespace nn
