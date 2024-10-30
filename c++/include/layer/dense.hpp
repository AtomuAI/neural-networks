// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DENSE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DENSE_HPP_

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
    template <typename T>
    class DenseLayer : public BaseLayer
    {
        //: Members
        protected:
            Shape<5>        input_shape;
            Shape<5>        output_shape;
            Tensor<T, 2>    weights;
            Tensor<T, 2>    jacobian;
            Tensor<T, 2>    momentum;
            Tensor<T, 2>    velocity;

        //: Constructors
        public:
            DenseLayer
            (
                const Shape<5>& input_shape         = Shape<5>(),
                const Shape<5>& output_shape        = Shape<5>(),
                const T scalar                      = T()
            );

            DenseLayer
            (
                const Shape<5>& input_shape,
                const Shape<5>& output_shape,
                const std::vector<T>& data
            );

        //: Destructors
        public:
            virtual ~DenseLayer();

        //: Methods
        public:
            void                reshape                     ( const Shape<5>& input_shape, const Shape<5>& output_shape );
            void                resize                      ( const Shape<5>& input_shape, const Shape<5>& output_shape );
            void                set_training_mode           ( const TrainingMode training_mode );
            const Shape<2>&     get_shape                   () const;
            Size                get_size                    () const;
            const Shape<5>&     get_input_shape             () const;
            const Shape<5>&     get_output_shape            () const;
            const Tensor<T, 2>& get_weights                  () const;
            const Tensor<T, 2>& get_jacobian                  () const;
            const Tensor<T, 2>& get_momentum                  () const;
            const Tensor<T, 2>& get_velocity                  () const;
            void                fill_weights                        ( const T value );
            void                zero_weights                        ();
            void                randomize_weights                   ( const T min, const T max );
            void                initialize_weights                  ( const InitializationType type, const DistributionType distribution );
        private:
            void                fill_jacobian                        ( const T value );
            void                fill_momentum                        ( const T value );
            void                fill_velocity                        ( const T value );
            void                zero_jacobian                        ();
            void                zero_momentum                        ();
            void                zero_velocity                        ();
            void                randomize_jacobian                   ( const T min, const T max );
            void                randomize_momentum                   ( const T min, const T max );
            void                randomize_velocity                   ( const T min, const T max );
        public:
            template <typename U, typename V> Error  inference          ( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const;
            template <typename U, typename V> Error  backpropagation    ( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer );
                                                Error gradient_decent_normal      ( const Dim batch_size, const StepSize step_size );
                                                Error gradient_decent_momentum    ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );
                                                Error gradient_decent_adam        ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );

        //: Operators
        public:
            inline T                get_weight  ( const Dim2D& indices ) const;
            inline T&               get_weight  ( const Dim2D& indices );
            inline T                get_weight  ( const Idx index ) const;
            inline T&               get_weight  ( const Idx index );
            inline T                get_jacobian( const Dim2D& indices ) const;
            inline T                get_jacobian( const Idx index ) const;
            inline T                get_momentum( const Dim2D& indices ) const;
            inline T                get_momentum( const Idx index ) const;
            inline T                get_velocity( const Dim2D& indices ) const;
            inline T                get_velocity( const Idx index ) const;
            inline DenseLayer<T>    operator+   ( const DenseLayer& other ) const;
            inline DenseLayer<T>    operator-   ( const DenseLayer& other ) const;
            inline DenseLayer<T>    operator*   ( const DenseLayer& other ) const;
            inline DenseLayer<T>    operator/   ( const DenseLayer& other ) const;
            inline DenseLayer<T>&   operator+=  ( const DenseLayer& other );
            inline DenseLayer<T>&   operator-=  ( const DenseLayer& other );
            inline DenseLayer<T>&   operator*=  ( const DenseLayer& other );
            inline DenseLayer<T>&   operator/=  ( const DenseLayer& other );
            inline DenseLayer<T>    operator+   ( const T scalar ) const;
            inline DenseLayer<T>    operator-   ( const T scalar ) const;
            inline DenseLayer<T>    operator*   ( const T scalar ) const;
            inline DenseLayer<T>    operator/   ( const T scalar ) const;
            inline DenseLayer<T>&   operator+=  ( const T scalar );
            inline DenseLayer<T>&   operator-=  ( const T scalar );
            inline DenseLayer<T>&   operator*=  ( const T scalar );
            inline DenseLayer<T>&   operator/=  ( const T scalar );
        private:
            inline T&               get_jacobian( const Dim2D& indices );
            inline T&               get_jacobian( const Idx index );
            inline T&               get_momentum( const Dim2D& indices );
            inline T&               get_momentum( const Idx index );
            inline T&               get_velocity( const Dim2D& indices );
            inline T&               get_velocity( const Idx index );
    };

    //: Inline Operators
        template <typename T>
        inline T DenseLayer<T>::get_weight( const Dim2D& indices ) const
        {
            return this->weights[ indices ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_jacobian( const Dim2D& indices ) const
        {
            return this->jacobian[ indices ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_momentum( const Dim2D& indices ) const
        {
            return this->momentum[ indices ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_velocity( const Dim2D& indices ) const
        {
            return this->velocity[ indices ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_weight( const Idx index ) const
        {
            return this->weights[ index ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_jacobian( const Idx index ) const
        {
            return this->jacobian[ index ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_momentum( const Idx index ) const
        {
            return this->momentum[ index ];
        }

        template <typename T>
        inline T DenseLayer<T>::get_velocity( const Idx index ) const
        {
            return this->velocity[ index ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_weight( const Dim2D& indices )
        {
            return this->weights[ indices ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_jacobian( const Dim2D& indices )
        {
            return this->jacobian[ indices ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_momentum( const Dim2D& indices )
        {
            return this->momentum[ indices ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_velocity( const Dim2D& indices )
        {
            return this->velocity[ indices ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_weight( const Idx index )
        {
            return this->weights[ index ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_jacobian( const Idx index )
        {
            return this->jacobian[ index ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_momentum( const Idx index )
        {
            return this->momentum[ index ];
        }

        template <typename T>
        inline T& DenseLayer<T>::get_velocity( const Idx index )
        {
            return this->velocity[ index ];
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator+( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights + other.weights;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator-( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights - other.weights;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator*( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights * other.weights;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator/( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights / other.weights;
            return result;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator+=( const DenseLayer<T>& other )
        {
            this->weights += other.weights;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator-=( const DenseLayer<T>& other )
        {
            this->weights -= other.weights;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator*=( const DenseLayer<T>& other )
        {
            this->weights *= other.weights;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator/=( const DenseLayer<T>& other )
        {
            this->weights /= other.weights;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator+( const T scalar ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights + scalar;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator-( const T scalar ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights - scalar;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator*( const T scalar ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights * scalar;
            return result;
        }

        template <typename T>
        inline DenseLayer<T> DenseLayer<T>::operator/( const T scalar ) const
        {
            DenseLayer<T> result( this->input_shape, this->output_shape );
            result.weights = this->weights / scalar;
            return result;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator+=( const T scalar )
        {
            this->weights += scalar;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator-=( const T scalar )
        {
            this->weights -= scalar;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator*=( const T scalar )
        {
            this->weights *= scalar;
            return *this;
        }

        template <typename T>
        inline DenseLayer<T>& DenseLayer<T>::operator/=( const T scalar )
        {
            this->weights /= scalar;
            return *this;
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_DENSE_HPP_
