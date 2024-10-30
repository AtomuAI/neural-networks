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

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T>
        NodeLayer<T>::NodeLayer( const Shape<5>& shape, const T scalar ) :
            BaseLayer( LayerType::node_layer ), nodes( shape, scalar ) {}

        template <typename T>
        NodeLayer<T>::NodeLayer( const Shape<5>& shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::node_layer ), nodes( shape, data ) {}

    //: Destructors
        template <typename T> NodeLayer<T>::~NodeLayer() {}

    //: Methods
        template <typename T>
        void NodeLayer<T>::set_training_mode( const TrainingMode training_mode )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( this->nodes, this->delta );
        }

        template <typename T>
        const Shape<5>& NodeLayer<T>::get_shape() const
        {
            return this->nodes.get_shape();
        }

        template <typename T>
        Size NodeLayer<T>::get_size() const
        {
            return this->nodes.get_size();
        }

        template <typename T>
        const Tensor<T, 5>& NodeLayer<T>::get_nodes() const
        {
            return this->nodes;
        }

        template <typename T>
        const Tensor<T, 5>& NodeLayer<T>::get_delta() const
        {
            return this->delta;
        }

        template <typename T>
        void NodeLayer<T>::reshape( const Shape<5>& shape )
        {
            this->BaseLayer::reshape( shape, this->nodes, this->delta );
        }

        template <typename T>
        void NodeLayer<T>::resize( const Shape<5>& shape )
        {
            this->BaseLayer::resize( shape, this->nodes, this->delta );
        }

        template <typename T>
        void NodeLayer<T>::fill_nodes( const T value )
        {
            this->nodes.fill( value );
        }

        template <typename T>
        void NodeLayer<T>::fill_delta( const T value )
        {
            this->delta.fill( value );
        }

        template <typename T>
        void NodeLayer<T>::zero_nodes()
        {
            this->nodes.zero();
        }

        template <typename T>
        void NodeLayer<T>::zero_delta()
        {
            this->nodes.zero();
        }

        template <typename T>
        void NodeLayer<T>::randomize_nodes( const T min, const T max )
        {
            this->nodes.randomize( min, max );
        }

        template <typename T>
        void NodeLayer<T>::randomize_delta( const T min, const T max )
        {
            this->nodes.randomize( min, max );
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define ClassMacro( type )\
        template class NodeLayer<type>;\

    ClassMacro( bool )
    ClassMacro( i32 )
    ClassMacro( i64 )
    ClassMacro( f32 )
    ClassMacro( f64 )

    #undef ClassMacro
} // namespace nn
