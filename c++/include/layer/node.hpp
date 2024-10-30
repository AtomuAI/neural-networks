// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NODE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NODE_HPP_

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

namespace nn
{
    template <typename T>
    class NodeLayer : public BaseLayer
    {
        //: Members
        protected:
            Tensor<T, 5>    nodes;
            Tensor<T, 5>    delta;

        //: Constructors
        public:
            NodeLayer
            (
                const Shape<5>&         shape           = Shape<5>( 0 ),
                const T                 scalar          = T()
            );

            NodeLayer
            (
                const Shape<5>&         shape,
                const std::vector<T>&   data
            );

        //: Destructors
        public:
            virtual ~NodeLayer();

        //: Methods
        public:
            void                    reshape                     ( const Shape<5>& shape );
            void                    resize                      ( const Shape<5>& shape );
            void                    set_training_mode           ( const TrainingMode training_mode );
            const Shape<5>&         get_shape                   () const;
            Size                    get_size                    () const;
            const Tensor<T, 5>&     get_nodes                   () const;
            const Tensor<T, 5>&     get_delta                   () const;
            void                    fill_nodes                  ( const T value );
            void                    zero_nodes                  ();
            void                    randomize_nodes             ( const T min, const T max );
        private:
            void                    fill_delta                  ( const T value );
            void                    zero_delta                  ();
            void                    randomize_delta             ( const T min, const T max );

        //: Operators
        public:
            inline T                get_node                    ( const Dim5D& indices ) const;
            inline T                get_delta                   ( const Dim5D& indices ) const;
            inline T                get_node                    ( const Idx index ) const;
            inline T                get_delta                   ( const Idx index ) const;
            inline T&               get_node                    ( const Dim5D& indices );
            inline T&               get_delta                   ( const Dim5D& indices );
            inline T&               get_node                    ( const Idx index );
            inline T&               get_delta                   ( const Idx index );
            inline NodeLayer<T>     operator=                   ( const NodeLayer<T>& other ) const;
            inline NodeLayer<T>     operator+                   ( const NodeLayer<T>& other ) const;
            inline NodeLayer<T>     operator-                   ( const NodeLayer<T>& other ) const;
            inline NodeLayer<T>     operator*                   ( const NodeLayer<T>& other ) const;
            inline NodeLayer<T>     operator/                   ( const NodeLayer<T>& other ) const;
            inline NodeLayer<T>&    operator+=                  ( const NodeLayer<T>& other );
            inline NodeLayer<T>&    operator-=                  ( const NodeLayer<T>& other );
            inline NodeLayer<T>&    operator*=                  ( const NodeLayer<T>& other );
            inline NodeLayer<T>&    operator/=                  ( const NodeLayer<T>& other );
            inline NodeLayer<T>     operator+                   ( const T scalar ) const;
            inline NodeLayer<T>     operator-                   ( const T scalar ) const;
            inline NodeLayer<T>     operator*                   ( const T scalar ) const;
            inline NodeLayer<T>     operator/                   ( const T scalar ) const;
            inline NodeLayer<T>&    operator+=                  ( const T scalar );
            inline NodeLayer<T>&    operator-=                  ( const T scalar );
            inline NodeLayer<T>&    operator*=                  ( const T scalar );
            inline NodeLayer<T>&    operator/=                  ( const T scalar );
    };

    //: Inline Operators
        template <typename T>
        inline T NodeLayer<T>::get_node( const Dim5D& indices ) const
        {
            return this->nodes[ indices ];
        }

        template <typename T>
        inline T NodeLayer<T>::get_delta( const Dim5D& indices ) const
        {
            return this->delta[ indices ];
        }

        template <typename T>
        inline T NodeLayer<T>::get_node( const Idx index ) const
        {
            return this->nodes[ index ];
        }

        template <typename T>
        inline T NodeLayer<T>::get_delta( const Idx index ) const
        {
            return this->delta[ index ];
        }

        template <typename T>
        inline T& NodeLayer<T>::get_node( const Dim5D& indices )
        {
            return this->nodes[ indices ];
        }

        template <typename T>
        inline T& NodeLayer<T>::get_delta( const Dim5D& indices )
        {
            return this->delta[ indices ];
        }

        template <typename T>
        inline T& NodeLayer<T>::get_node( const Idx index )
        {
            return this->nodes[ index ];
        }

        template <typename T>
        inline T& NodeLayer<T>::get_delta( const Idx index )
        {
            return this->delta[ index ];
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator+( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + other.nodes;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator-( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - other.nodes;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator*( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * other.nodes;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator/( const NodeLayer<T>& other ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / other.nodes;
            return result;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator+=( const NodeLayer<T>& other )
        {
            this->nodes += other.nodes;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator-=( const NodeLayer<T>& other )
        {
            this->nodes -= other.nodes;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator*=( const NodeLayer<T>& other )
        {
            this->nodes *= other.nodes;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator/=( const NodeLayer<T>& other )
        {
            this->nodes /= other.nodes;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator+( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + scalar;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator-( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - scalar;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator*( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * scalar;
            return result;
        }

        template <typename T>
        inline NodeLayer<T> NodeLayer<T>::operator/( const T scalar ) const
        {
            NodeLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / scalar;
            return result;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator+=( const T scalar )
        {
            this->nodes += scalar;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator-=( const T scalar )
        {
            this->nodes -= scalar;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator*=( const T scalar )
        {
            this->nodes *= scalar;
            return *this;
        }

        template <typename T>
        inline NodeLayer<T>& NodeLayer<T>::operator/=( const T scalar )
        {
            this->nodes /= scalar;
            return *this;
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_NODE_HPP_
