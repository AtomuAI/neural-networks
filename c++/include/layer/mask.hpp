// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_MASK_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_MASK_HPP_

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
#include "bewusstsein_neural_networks/c++/include/core/tensor.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/core/step_size.hpp"
#include "bewusstsein_neural_networks/c++/include/core/beta.hpp"
#include "bewusstsein_neural_networks/c++/include/core/epsilon.hpp"
#include "bewusstsein_neural_networks/c++/include/core/training_mode.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

namespace nn
{
    class MaskLayer : public BaseLayer
    {
        //: Members
        protected:
            Tensor<bool, 4>   mask;

        //: Constructors
        public:
            MaskLayer
            (
                const Shape<4>&             shape   = Shape<4>(),
                const bool                  scalar  = false
            );
        protected:
            MaskLayer
            (
                const LayerType             type    = LayerType::mask_layer,
                const Shape<4>&             shape   = Shape<4>(),
                const bool                  scalar  = false
            );

        //: Destructors
        public:
            virtual ~MaskLayer();

        //: Methods
        public:
            void reshape( const Shape<4>& shape );
            void resize( const Shape<4>& shape );
            const Shape<4>& get_shape() const;
            Size get_size() const;
            const Tensor<bool, 4>& get_mask() const;
            void fill( const bool value );

            template <typename U>
            Error inference( NodeLayer<U>& layer ) const;
            template <typename U>
            Error backpropagation( NodeLayer<U>& layer ) const;

        //: Operators
        public:
            inline bool  get_mask( const Dim4D& indices ) const;
            inline bool  get_mask( const Idx index ) const;
            inline bool& get_mask( const Dim4D& indices );
            inline bool& get_mask( const Idx index );
    };

    //: Inline Operators
        inline bool MaskLayer::get_mask( const Dim4D& indices ) const
        {
            return this->mask[ indices ];
        }

        inline bool MaskLayer::get_mask( const Idx index ) const
        {
            return this->mask[ index ];
        }

        inline bool& MaskLayer::get_mask( const Dim4D& indices )
        {
            return this->mask[ indices ];
        }

        inline bool& MaskLayer::get_mask( const Idx index )
        {
            return this->mask[ index ];
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_MASK_HPP_
