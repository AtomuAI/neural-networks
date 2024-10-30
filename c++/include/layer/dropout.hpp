// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DROPOUT_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DROPOUT_HPP_

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

namespace nn
{
    class DropoutLayer : public MaskLayer
    {
        //: Members
        protected:
            double                  rate;

        //: Constructors
        public:
            DropoutLayer
            (
                const Shape<4>& shape       = Shape<4>(),
                const double dropout_rate   = 0
            );

        //: Destructors
        public:
            virtual ~DropoutLayer();

        //: Methods
        public:
            void reshape( const Shape<4>& shape );
            void resize( const Shape<4>& shape );
            const Shape<4>& get_shape() const;
            Size get_size() const;
            double get_rate() const;

            template <typename U>
            Error inference( NodeLayer<U>& layer );
            template <typename U>
            Error backpropagation( NodeLayer<U>& layer ) const;
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DROPOUT_HPP_
