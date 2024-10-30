// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BIAS_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BIAS_HPP_

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
#include "bewusstsein_neural_networks/c++/include/core/math.hpp"
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
    template <typename T>
    class BiasLayer : public BaseLayer
    {
        //: Members
        protected:
            Tensor<T, 4> bias;
            Tensor<T, 4> jacobian;
            Tensor<T, 4> momentum;
            Tensor<T, 4> velocity;

        //: Constructors
        public:
            BiasLayer
            (
                const Shape<4>& shape               = Shape<4>( { 0 } ),
                const T scalar                      = T()
            );

            BiasLayer
            (
                const Shape<4>& shape,
                const std::vector<T>& data
            );

        //: Destructors
        public:
            virtual ~BiasLayer();

        //: Methods
        public:
                                        void                    reshape                     ( const Shape<4>& shape );
                                        void                    resize                      ( const Shape<4>& shape );
                                        void                    set_training_mode           ( const TrainingMode training_mode );
                                        const Shape<4>&         get_shape                   () const;
                                        Size                    get_size                    () const;
                                        const Tensor<T, 4>&     get_bias                    () const;
                                        const Tensor<T, 4>&     get_jacobian                () const;
                                        const Tensor<T, 4>&     get_momentum                () const;
                                        const Tensor<T, 4>&     get_velocity                () const;
                                        void                    fill_bias                   ( const T value );
                                        void                    zero_bias                   ();
                                        void                    randomize_bias              ( const T min, const T max );
            template <typename U>       Error                   inference                   ( NodeLayer<U>& layer ) const;
            template <typename U>       Error                   backpropagation             ( const NodeLayer<U>& layer );
                                        Error                   gradient_decent_normal      ( const Dim batch_size, const StepSize step_size );
                                        Error                   gradient_decent_momentum    ( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size );
                                        Error                   gradient_decent_adam        ( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon );
        private:
                                        void                    fill_jacobian               ( const T value );
                                        void                    fill_momentum               ( const T value );
                                        void                    fill_velocity               ( const T value );
                                        void                    zero_jacobian               ();
                                        void                    zero_momentum               ();
                                        void                    zero_velocity               ();
                                        void                    randomize_jacobian          ( const T min, const T max );
                                        void                    randomize_momentum          ( const T min, const T max );
                                        void                    randomize_velocity          ( const T min, const T max );

    //: Operators
        public:
                                        inline T                get_bias                    ( const Dim4D& indices ) const;
                                        inline T                get_jacobian                ( const Dim4D& indices ) const;
                                        inline T                get_momentum                ( const Dim4D& indices ) const;
                                        inline T                get_velocity                ( const Dim4D& indices ) const;
                                        inline T                get_bias                    ( const Idx index ) const;
                                        inline T                get_jacobian                ( const Idx index ) const;
                                        inline T                get_momentum                ( const Idx index ) const;
                                        inline T                get_velocity                ( const Idx index ) const;
                                        inline T&               get_bias                    ( const Dim4D& indices );
        private:
                                        inline T&               get_jacobian                ( const Dim4D& indices );
                                        inline T&               get_momentum                ( const Dim4D& indices );
                                        inline T&               get_velocity                ( const Dim4D& indices );
        public:
                                        inline T&               get_bias                    ( const Idx index );
        private:
                                        inline T&               get_jacobian                ( const Idx index );
                                        inline T&               get_momentum                ( const Idx index );
                                        inline T&               get_velocity                ( const Idx index );
        public:
                                        inline BiasLayer<T>     operator+                   ( const BiasLayer<T>& other ) const;
                                        inline BiasLayer<T>     operator-                   ( const BiasLayer<T>& other ) const;
                                        inline BiasLayer<T>     operator*                   ( const BiasLayer<T>& other ) const;
                                        inline BiasLayer<T>     operator/                   ( const BiasLayer<T>& other ) const;
                                        inline BiasLayer<T>&    operator+=                  ( const BiasLayer<T>& other );
                                        inline BiasLayer<T>&    operator-=                  ( const BiasLayer<T>& other );
                                        inline BiasLayer<T>&    operator*=                  ( const BiasLayer<T>& other );
                                        inline BiasLayer<T>&    operator/=                  ( const BiasLayer<T>& other );
                                        inline BiasLayer<T>     operator+                   ( const T scalar ) const;
                                        inline BiasLayer<T>     operator-                   ( const T scalar ) const;
                                        inline BiasLayer<T>     operator*                   ( const T scalar ) const;
                                        inline BiasLayer<T>     operator/                   ( const T scalar ) const;
                                        inline BiasLayer<T>&    operator+=                  ( const T scalar );
                                        inline BiasLayer<T>&    operator-=                  ( const T scalar );
                                        inline BiasLayer<T>&    operator*=                  ( const T scalar );
                                        inline BiasLayer<T>&    operator/=                  ( const T scalar );
    };

    //: Inline Operators
        template <typename T>
        inline T BiasLayer<T>::get_bias( const Dim4D& indices ) const
        {
            return this->bias[ indices ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_jacobian( const Dim4D& indices ) const
        {
            return this->jacobian[ indices ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_momentum( const Dim4D& indices ) const
        {
            return this->momentum[ indices ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_velocity( const Dim4D& indices ) const
        {
            return this->velocity[ indices ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_bias( const Idx index ) const
        {
            return this->bias[ index ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_jacobian( const Idx index ) const
        {
            return this->jacobian[ index ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_momentum( const Idx index ) const
        {
            return this->momentum[ index ];
        }

        template <typename T>
        inline T BiasLayer<T>::get_velocity( const Idx index ) const
        {
            return this->velocity[ index ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_bias( const Dim4D& indices )
        {
            return this->bias[ indices ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_jacobian( const Dim4D& indices )
        {
            return this->jacobian[ indices ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_momentum( const Dim4D& indices )
        {
            return this->momentum[ indices ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_velocity( const Dim4D& indices )
        {
            return this->velocity[ indices ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_bias( const Idx index )
        {
            return this->bias[ index ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_jacobian( const Idx index )
        {
            return this->jacobian[ index ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_momentum( const Idx index )
        {
            return this->momentum[ index ];
        }

        template <typename T>
        inline T& BiasLayer<T>::get_velocity( const Idx index )
        {
            return this->velocity[ index ];
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator+(  const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias + other.bias;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator-( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias - other.bias;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator*( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias * other.bias;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator/( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias / other.bias;
            return result;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator+=( const BiasLayer<T>& other )
        {
            this->bias += other.bias;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator-=( const BiasLayer<T>& other )
        {
            this->bias -= other.bias;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator*=( const BiasLayer<T>& other )
        {
            this->bias *= other.bias;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator/=( const BiasLayer<T>& other )
        {
            this->bias /= other.bias;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator+( const T scalar ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias + scalar;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator-( const T scalar ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias - scalar;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator*( const T scalar ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias * scalar;
            return result;
        }

        template <typename T>
        inline BiasLayer<T> BiasLayer<T>::operator/( const T scalar ) const
        {
            BiasLayer<T> result( this->bias.get_shape() );
            result.bias = this->bias / scalar;
            return result;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator+=( const T scalar )
        {
            this->bias += scalar;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator-=( const T scalar )
        {
            this->bias -= scalar;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator*=( const T scalar )
        {
            this->bias *= scalar;
            return *this;
        }

        template <typename T>
        inline BiasLayer<T>& BiasLayer<T>::operator/=( const T scalar )
        {
            this->bias /= scalar;
            return *this;
        }
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_BIAS_HPP_
