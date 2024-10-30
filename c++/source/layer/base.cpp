// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
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

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"

namespace nn
{
    //: Constructors
        BaseLayer::BaseLayer( const LayerType layer_type, const TrainingMode training_mode ) :
            layer_type( layer_type ), training_mode( training_mode ), time_step( 1 ) {}

    //: Destructors
        BaseLayer::~BaseLayer() {}

    //: Methods
        LayerType BaseLayer::get_layer_type() const
        {
            return this->layer_type;
        }

        const Counter<u64>& BaseLayer::get_timer() const
        {
            return this->time_step;
        }

        u64 BaseLayer::get_time_step() const
        {
            return this->time_step.get_count();
        }

        void BaseLayer::tick_time_step()
        {
            this->time_step.tick();
        }

        void BaseLayer::reset_time_step()
        {
            this->time_step.reset();
        }
} // namespace nn
