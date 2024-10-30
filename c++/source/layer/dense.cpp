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
#include "bewusstsein_neural_networks/c++/include/core/initialization_type.hpp"
#include "bewusstsein_neural_networks/c++/include/core/distribution_type.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/layer/dense.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T>
        DenseLayer<T>::DenseLayer( const Shape<5>& input_shape, const Shape<5>& output_shape, const T scalar ) :
            BaseLayer( LayerType::dense_layer ), input_shape( input_shape ), output_shape( output_shape ), weights( Shape<2>( input_shape.volume(), output_shape.volume() ), scalar ) {}

        template <typename T>
        DenseLayer<T>::DenseLayer( const Shape<5>& input_shape, const Shape<5>& output_shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::dense_layer ), input_shape( input_shape ), output_shape( output_shape )
        {
            Size total_input_size = this->input_shape.volume();
            Size total_output_size = this->output_shape.volume();
            if ( data.size() != total_input_size * total_output_size )
            {
                throw std::invalid_argument( "Data size does not match the total size of the input and output shapes." );
            }
            weights = Tensor<T, 2>( Shape<2>( static_cast<Dim>( total_input_size ), static_cast<Dim>( total_output_size ) ), data );
        }

    //: Destructors
        template <typename T>
        DenseLayer<T>::~DenseLayer() {}

    //: Methods
        template <typename T>
        void DenseLayer<T>::set_training_mode( const TrainingMode training_mode )
        {
            this->training_mode = training_mode;
            this->BaseLayer::allocate_training_memory( this->weights, this->jacobian, this->momentum, this->velocity );
        }

        /*
        template <typename T>
        void DenseLayer<T>::initialize_weights( const InitializationType type, const DistributionType distribution )
        {
            const Shape<2> weights_shape = weights.get_shape();
            Size input_layer_size = weights_shape.width();
            Size output_layer_size = weights_shape.height();
            T variance = 0;
            T std_dev = 0;
            switch (type)
            {
                case InitializationType::xavier_glorot: { variance = 2.0 / (  input_layer_size + output_layer_size ); std_dev = nn::sqrt( variance ); break; }
                case InitializationType::he: { variance = 2.0 / ( input_layer_size ); std_dev = nn::sqrt( variance ); break; }
                case InitializationType::lecun: { variance = 1.0 / ( input_layer_size ); std_dev = nn::sqrt( variance ); break; }
                default: {throw std::invalid_argument( "Initialization Type is invalid/uninitialized" ); break; }
            }

            switch (distribution)
            {
                case DistributionType::normal: { this->weights.fill_normal_distribution( 0, std_dev ); break; }
                case DistributionType::poisson: { filter.fill_poisson_distribution(); break; }
                case DistributionType::binomial: { filter.fill_binomial_distribution(); break; }
                case DistributionType::exponential: { filter.fill_exponential_distribution(); break; }
                case DistributionType::uniform: { filter.fill_uniform_distribution(); break; }
                case DistributionType::bernoulli: { filter.fill_bernoulli_distribution(); break; }
                case DistributionType::beta: { filter.fill_beta_distribution(); break; }
                case DistributionType::weibull: { filter.fill_weibull_distribution(); break; }
                case DistributionType::gamma: { filter.fill_gamma_distribution(); break; }
                case DistributionType::chi_squared: { filter.fill_chi_squared_distribution(); break; }
                case DistributionType::log_normal: { filter.fill_log_normal_distribution(); break; }
                case DistributionType::f: { filter.fill_f_distribution(); break; }
                case DistributionType::discrete_uniform: { filter.fill_discrete_uniform_distribution(); break; }
                default: { throw std::invalid_argument( "Distribution Type is invalid/uninitialized" ); break; }
            };
        }
        */

        template <typename T>
        void DenseLayer<T>::fill_weights( const T value )
        {
            this->weights.fill( value );
        }

        template <typename T>
        void DenseLayer<T>::fill_jacobian( const T value )
        {
            this->jacobian.fill( value );
        }

        template <typename T>
        void DenseLayer<T>::fill_momentum( const T value )
        {
            this->momentum.fill( value );
        }

        template <typename T>
        void DenseLayer<T>::fill_velocity( const T value )
        {
            this->velocity.fill( value );
        }

        template <typename T>
        void DenseLayer<T>::zero_weights()
        {
            this->weights.zero();
        }

        template <typename T>
        void DenseLayer<T>::zero_jacobian()
        {
            this->jacobian.zero();
        }

        template <typename T>
        void DenseLayer<T>::zero_momentum()
        {
            this->momentum.zero();
        }

        template <typename T>
        void DenseLayer<T>::zero_velocity()
        {
            this->velocity.zero();
        }

        template <typename T>
        void DenseLayer<T>::randomize_weights( const T min, const T max )
        {
            this->weights.randomize( min, max );
        }

        template <typename T>
        void DenseLayer<T>::randomize_jacobian( const T min, const T max )
        {
            this->jacobian.randomize( min, max );
        }

        template <typename T>
        void DenseLayer<T>::randomize_momentum( const T min, const T max )
        {
            this->momentum.randomize( min, max );
        }

        template <typename T>
        void DenseLayer<T>::randomize_velocity( const T min, const T max )
        {
            this->velocity.randomize( min, max );
        }

        template <typename T>
        const Shape<2>& DenseLayer<T>::get_shape() const
        {
            return this->weights.get_shape();
        }

        template <typename T>
        Size DenseLayer<T>::get_size() const
        {
            return this->weights.get_size();
        }

        template <typename T>
        const Shape<5>& DenseLayer<T>::get_input_shape() const
        {
            return this->input_shape;
        }

        template <typename T>
        const Shape<5>& DenseLayer<T>::get_output_shape() const
        {
            return this->output_shape;
        }

        template <typename T>
        const Tensor<T, 2>& DenseLayer<T>::get_weights() const
        {
            return this->weights;
        }

        template <typename T>
        const Tensor<T, 2>& DenseLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const Tensor<T, 2>& DenseLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const Tensor<T, 2>& DenseLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        void DenseLayer<T>::reshape( const Shape<5>& input_shape, const Shape<5>& output_shape )
        {
            Size total_input_size = this->input_shape.volume();
            Size total_output_size = this->output_shape.volume();
            this->BaseLayer::reshape( Shape<2>( static_cast<Dim>( total_input_size ), static_cast<Dim>( total_output_size ) ), this->weights, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T>
        void DenseLayer<T>::resize( const Shape<5>& input_shape, const Shape<5>& output_shape )
        {
            Size total_input_size = this->input_shape.volume();
            Size total_output_size = this->output_shape.volume();
            this->BaseLayer::reshape( Shape<2>( static_cast<Dim>( total_input_size ), static_cast<Dim>( total_output_size ) ), this->weights, this->jacobian, this->momentum, this->velocity );
        }

        template <typename T>
        template <typename U, typename V>
        Error DenseLayer<T>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            if ( this->weights.get_shape().width() != input_layer.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }
            if ( this->weights.get_shape().height() != output_layer.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }

            const Shape<5> input_shape = input_layer.get_shape();
            Size input_spacial_size = input_shape.distance( 0, 4 );
            const Shape<5> output_shape = output_layer.get_shape();
            Size output_spacial_size = output_shape.distance( 0, 4 );

            for ( Dim batch = 0; batch < input_shape.batches(); batch++ )
            {
                Idx input_batch_index = batch * input_spacial_size;
                Idx output_batch_index = batch * output_spacial_size;

                for ( Idx output_index = 0; output_index < output_spacial_size; ++output_index )
                {
                    Idx out_index = output_batch_index + output_index;
                    V sum( 0 );

                    for ( Idx input_index = 0; input_index < input_spacial_size; ++input_index )
                    {
                        Idx in_index = input_batch_index + input_index;
                        Idx weight_index = ( output_index * input_spacial_size ) + input_index;
                        sum += V( this->get_weight( weight_index ) * input_layer.get_node( in_index ) );
                    }

                    output_layer.get_node( out_index ) = sum;
                }
            }

            return Error::NONE;
        }

        template <typename T>
        template <typename U, typename V>
        Error DenseLayer<T>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            if ( this->weights.get_shape().width() != input_layer.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }
            if ( this->weights.get_shape().height() != output_layer.get_shape().volume() ) { return Error::MISMATCHED_SHAPES; }

            const Shape<5> input_shape = input_layer.get_shape();
            Size input_spacial_size = input_shape.distance( 0, 4 );
            const Shape<5> output_shape = output_layer.get_shape();
            Size output_spacial_size = output_shape.distance( 0, 4 );

            for ( Dim batch = 0; batch < input_shape.batches(); batch++ )
            {
                Idx input_batch_index = batch * input_spacial_size;
                Idx output_batch_index = batch * output_spacial_size;

                for ( Idx input_index = 0; input_index < input_spacial_size; ++input_index )
                {
                    Idx in_index = input_batch_index + input_index;
                    U input_node = input_layer.get_node( in_index );

                    U delta( 0 );

                    for ( Idx output_index = 0; output_index < output_spacial_size; ++output_index )
                    {
                        Idx out_index = output_batch_index + output_index;
                        Idx weight_index = ( output_index * input_spacial_size ) + input_index;

                        V out_delta = output_layer.get_delta( out_index );
                        T weight = this->get_weight( weight_index );

                        this->get_jacobian( weight_index ) += input_node * out_delta;

                        delta += U( weight * out_delta );
                    }

                    input_layer.get_delta( in_index ) = delta;
                }
            }

            return Error::NONE;
        }

        template <typename T>
        Error DenseLayer<T>::gradient_decent_normal( const Dim batch_size, const StepSize step_size )
        {
            if ( this->training_mode != TrainingMode::normal ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_normal( this->weights, this->jacobian, batch_size, step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T>
        Error DenseLayer<T>::gradient_decent_momentum( const Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            if ( this->training_mode != TrainingMode::momentum ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_momentum( this->weights, this->jacobian, this->momentum, batch_size, step_size, momentum_step_size );
            this->tick_time_step();
            return Error::NONE;
        }

        template <typename T>
        Error DenseLayer<T>::gradient_decent_adam( const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            if ( this->training_mode != TrainingMode::adam ) { return Error::INCORRECT_TRAINING_MODE; }
            this->BaseLayer::gradient_decent_adam( this->weights, this->jacobian, this->momentum, this->velocity, batch_size, step_size, beta1, beta2, epsilon );
            this->tick_time_step();
            return Error::NONE;
        }
} // namespace nn

//: Specializations
namespace nn
{
    #define FunctionMacroB( type_a, type_b, type_c )\
        template Error DenseLayer<type_a>::inference( const NodeLayer<type_b>& input_layer, NodeLayer<type_c>& output_layer ) const;\
        template Error DenseLayer<type_a>::backpropagation( NodeLayer<type_b>& input_layer, const NodeLayer<type_c>& output_layer );

    #define FunctionMacro( type_a, type_b )\
        FunctionMacroB( type_a, type_b, i32 )\
        FunctionMacroB( type_a, type_b, i64 )\
        FunctionMacroB( type_a, type_b, f32 )\
        FunctionMacroB( type_a, type_b, f64 )

    #define ClassMacroB( type )\
        template class DenseLayer<type>;\
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

    #undef FunctionMacroB
    #undef FunctionMacro
    #undef ClassMacroB
    #undef ClassMacro
} // namespace nn
