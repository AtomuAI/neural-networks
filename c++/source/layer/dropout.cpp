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
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/mask.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/dropout.hpp"

namespace nn
{
    //: Constructors
        DropoutLayer::DropoutLayer( const Shape<4>& shape, const double dropout_rate ):
            MaskLayer( LayerType::dropout_layer, shape ), rate( dropout_rate ) {}

    //: Destructors
        DropoutLayer::~DropoutLayer() {}

    //: Methods
        void DropoutLayer::reshape( const Shape<4>& shape )
        {
            this->mask.reshape( shape );
        }

        void DropoutLayer::resize( const Shape<4>& shape )
        {
            this->mask.resize( shape );
        }

        const Shape<4>& DropoutLayer::get_shape() const
        {
            return this->mask.get_shape();
        }

        Size DropoutLayer::get_size() const
        {
            return this->mask.get_size();
        }

        double DropoutLayer::get_rate() const
        {
            return this->rate;
        }

        template <typename U>
        Error DropoutLayer::inference( NodeLayer<U>& layer )
        {
            Size mask_size = this->mask.get_size();

            for ( int index = 0; index < mask_size; index++ )
            {
                this->get_mask( index ) = rated_gen( this->rate );
            }

            return this->MaskLayer::inference( layer );
        }

        template <typename U>
        Error DropoutLayer::backpropagation( NodeLayer<U>& layer ) const
        {
            return this->MaskLayer::backpropagation( layer );
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacro( type )\
        template Error DropoutLayer::inference( NodeLayer<type>& layer );\
        template Error DropoutLayer::backpropagation( NodeLayer<type>& layer ) const;

    #define ClassMacro\
        FunctionMacro( i32 )\
        FunctionMacro( i64 )\
        FunctionMacro( f32 )\
        FunctionMacro( f64 )

    ClassMacro

    #undef FunctionMacro
    #undef ClassMacro
} // namespace nn
