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

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/bias.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T>
        BiasLayer<T>::BiasLayer( const Shape<4>& shape, const T scalar ) :
            BaseLayer( LayerType::bias_layer ), bias( shape, scalar ) {}

        template <typename T>
        BiasLayer<T>::BiasLayer( const Shape<4>& shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::bias_layer ), bias( shape, data ) {}

    //: Destructors
        template <typename T>
        BiasLayer<T>::~BiasLayer() {}

    //: Methods
        template <typename T>
        void BiasLayer<T>::set_training_mode( const TrainingMode training_mode )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( this->bias, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T>
        const Shape<4>& BiasLayer<T>::get_shape() const
        {
            return this->bias.get_shape();
        }

        template <typename T>
        Size BiasLayer<T>::get_size() const
        {
            return this->bias.get_size();
        }

        template <typename T>
        const Tensor<T, 4>& BiasLayer<T>::get_bias() const
        {
            return this->bias;
        }

        template <typename T>
        const Tensor<T, 4>& BiasLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const Tensor<T, 4>& BiasLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const Tensor<T, 4>& BiasLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        void BiasLayer<T>::reshape( const Shape<4>& shape )
        {
            this->BaseLayer::reshape( shape, this->bias, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T>
        void BiasLayer<T>::resize( const Shape<4>& shape )
        {
            this->BaseLayer::resize( shape, this->bias, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T>
        void BiasLayer<T>::fill_bias( const T value )
        {
            this->bias.fill( value );
        }

        template <typename T>
        void BiasLayer<T>::fill_jacobian( const T value )
        {
            this->jacobian.fill( value );
        }

        template <typename T>
        void BiasLayer<T>::fill_momentum( const T value )
        {
            this->momentum.fill( value );
        }

        template <typename T>
        void BiasLayer<T>::fill_velocity( const T value )
        {
            this->velocity.fill( value );
        }

        template <typename T>
        void BiasLayer<T>::zero_bias()
        {
            this->bias.zero();
        }

        template <typename T>
        void BiasLayer<T>::zero_jacobian()
        {
            this->jacobian.zero();
        }

        template <typename T>
        void BiasLayer<T>::zero_momentum()
        {
            this->momentum.zero();
        }

        template <typename T>
        void BiasLayer<T>::zero_velocity()
        {
            this->velocity.zero();
        }

        template <typename T>
        void BiasLayer<T>::randomize_bias( const T min, const T max )
        {
            this->bias.randomize( min, max );
        }

        template <typename T>
        void BiasLayer<T>::randomize_jacobian( const T min, const T max )
        {
            this->jacobian.randomize( min, max );
        }

        template <typename T>
        void BiasLayer<T>::randomize_momentum( const T min, const T max )
        {
            this->momentum.randomize( min, max );
        }

        template <typename T>
        void BiasLayer<T>::randomize_velocity( const T min, const T max )
        {
            this->velocity.randomize( min, max );
        }

        template <typename T>
        template <typename U>
        Error BiasLayer<T>::inference( NodeLayer<U>& layer ) const
        {
            const Shape<5> shape = layer.get_shape();
            if (shape.distance( 0, 4 ) != this->bias.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }
            Size spacial_size = shape.distance( 0, 4 );

            for ( Dim batch = 0; batch < shape.batches(); batch++ )
            {
                Idx batch_index = batch * spacial_size;
                for ( Idx spacial = 0; spacial < spacial_size; spacial++ )
                {
                    Idx index = batch_index + spacial;
                    layer.get_node( index ) += this->get_bias( spacial );
                }
            }

            return Error::NONE;
        }

        template <typename T>
        template <typename U>
        Error BiasLayer<T>::backpropagation( const NodeLayer<U>& layer )
        {
            const Shape<5> shape = layer.get_shape();
            if (shape.distance( 0, 4 ) != this->bias.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }
            Size spacial_size = shape.distance( 0, 4 );

            for ( Dim batch = 0; batch < shape.batches(); batch++ )
            {
                Idx batch_index = batch * spacial_size;
                for ( Idx spacial = 0; spacial < spacial_size; spacial++ )
                {
                    Idx index = batch_index + spacial;
                    this->get_jacobian( spacial ) += layer.get_delta( index );
                }
            }

            return Error::NONE;
        }

        template <typename T>
        Error BiasLayer<T>::gradient_decent_normal( const Dim batch_size, const StepSize step_size )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->bias, this->jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T>
        Error BiasLayer<T>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->bias, this->jacobian, this->momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T>
        Error BiasLayer<T>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->bias, this->jacobian, this->momentum, this->velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacro( type_a, type_b )\
        template Error BiasLayer<type_a>::inference( NodeLayer<type_b>& layer ) const;\
        template Error BiasLayer<type_a>::backpropagation( const NodeLayer<type_b>& layer );

    #define ClassMacroB( type )\
        template class BiasLayer<type>;\
        FunctionMacro( type, i32 )\
        FunctionMacro( type, i64 )\
        FunctionMacro( type, f32 )\
        FunctionMacro( type, f64 )

    #define ClassMacro\
        ClassMacroB( i32 )\
        ClassMacroB( i64 )\
        ClassMacroB( f32 )\
        ClassMacroB( f64 )

    ClassMacro

    #undef FunctionMacro
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn

/*
#define CLASS( type_a, type_b )\
        template class BiasLayer<type_a>;\
        template Error BiasLayer<type_a>::inference( NodeLayer<type_b>& layer ) const;\
        template Error BiasLayer<type_a>::backpropagation( const NodeLayer<type_b>& layer );

    CLASS( bool, bool )
    CLASS( bool, i32 )
    CLASS( bool, i64 )
    CLASS( bool, f32 )
    CLASS( bool, f64 )

    CLASS( i32, bool )
    CLASS( i32, i32 )
    CLASS( i32, i64 )
    CLASS( i32, f32 )
    CLASS( i32, f64 )

    CLASS( i64, bool )
    CLASS( i64, i32 )
    CLASS( i64, i64 )
    CLASS( i64, f32 )
    CLASS( i64, f64 )

    CLASS( f32, bool )
    CLASS( f32, i32 )
    CLASS( f32, i64 )
    CLASS( f32, f32 )
    CLASS( f32, f64 )

    CLASS( f64, bool )
    CLASS( f64, i32 )
    CLASS( f64, i64 )
    CLASS( f64, f32 )
    CLASS( f64, f64 )

    #undef CLASS
*/