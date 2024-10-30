// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>

int main()
{
    std::vector<std::string> types =
    {
        //"int8_t",
        //"int16_t",
        //"int32_t",
        //"int64_t",
        "float"
        //"double",
        //"std::complex<int8_t>",
        //"std::complex<int16_t>",
        //"std::complex<int32_t>",
        //"std::complex<int64_t>",
        //"std::complex<float>",
        //"std::complex<double>"
    };

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/dense_layer/dense_layer.cpp");
    if (!file)
    {
        std::cerr << "Unable to create file" << std::endl;
        return 1;
    }

    // Write the common parts of the file
    file <<
R"(// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>

//: Library Headers
#include <bewusstsein_util.hpp>

//: Project Headers
#include "c++/bewusstsein_neural_networks/source/core/counter/counter.hpp"
#include "c++/bewusstsein_neural_networks/source/core/step_size/step_size.hpp"
#include "c++/bewusstsein_neural_networks/source/core/beta/beta.hpp"
#include "c++/bewusstsein_neural_networks/source/core/epsilon/epsilon.hpp"
#include "c++/bewusstsein_neural_networks/source/core/training_mode/training_mode.hpp"
#include "c++/bewusstsein_neural_networks/source/core/initialization_type/initialization_type.hpp"
#include "c++/bewusstsein_neural_networks/source/core/distribution_type/distribution_type.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/dense_layer/dense_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        DenseLayer<T>::DenseLayer() :
            BaseLayer( LayerType::dense_layer, "" ) {}

        template <typename T>
        DenseLayer<T>::DenseLayer( const std::string& name, const util::Shape& input_shape, const util::Shape& output_shape ) :
            BaseLayer( LayerType::dense_layer, name, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1 ) || ( output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions" );
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            this->weights = util::Tensor<T>( util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ) );
        }

        template <typename T>
        DenseLayer<T>::DenseLayer( const std::string& name, const util::Shape& input_shape, const util::Shape& output_shape, const T scalar ) :
            BaseLayer( LayerType::dense_layer, name, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1 ) || ( output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions" );
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            weights = util::Tensor<T>(util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ), scalar );
        }

        template <typename T>
        DenseLayer<T>::DenseLayer( const std::string& name, const util::Shape& input_shape, const util::Shape& output_shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::dense_layer, name, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1 ) || ( output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions" );
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            if ( data.size() != total_input_size * total_output_size )
            {
                throw std::invalid_argument( "Data size does not match the total size of the input and output shapes." );
            }
            weights = util::Tensor<T>( util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ), data );
        }

        template <typename T>
        DenseLayer<T>::DenseLayer( const std::string& name, const util::Shape& input_shape, const util::Shape& output_shape, TrainingMode training_mode ) :
            BaseLayer( LayerType::dense_layer, name, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1) || (  output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions");
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            this->weights = util::Tensor<T>( util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ) );

            this->allocate_training_memory( training_mode );
        }
        */

        template <typename T>
        DenseLayer<T>::DenseLayer( const util::Shape& input_shape, const util::Shape& output_shape, const T scalar, const TrainingMode training_mode ) :
            BaseLayer( LayerType::dense_layer, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1 ) || ( output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions" );
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            weights = util::Tensor<T>( util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ), scalar );

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        DenseLayer<T>::DenseLayer( const util::Shape& input_shape, const util::Shape& output_shape, const std::vector<T>& data, const TrainingMode training_mode ) :
            BaseLayer( LayerType::dense_layer, input_shape, output_shape )
        {
            if( ( input_shape.size() > 5 ) || ( input_shape.size() < 1 ) || ( output_shape.size() > 5 ) || ( output_shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Denselayer can have input and output shapes of at least 1 and at most 5 dimensions");
            }

            util::Size total_input_size = this->input_shape.volume();
            util::Size total_output_size = this->output_shape.volume();
            if ( data.size() != total_input_size * total_output_size )
            {
                throw std::invalid_argument( "Data size does not match the total size of the input and output shapes." );
            }
            weights = util::Tensor<T>( util::Shape( static_cast<util::Dim>( total_input_size ), static_cast<util::Dim>( total_output_size ) ), data );

            this->allocate_training_memory( training_mode );
        }

    //: Destructors
        template <typename T>
        DenseLayer<T>::~DenseLayer() {}

    //: Operators
        template <typename T>
        T DenseLayer<T>::operator[]( const util::Indices& indices ) const
        {
            return this->weights[ indices ];
        }

        template <typename T>
        T DenseLayer<T>::operator[]( const util::Index index ) const
        {
            return this->weights[ index ];
        }

        template <typename T>
        T& DenseLayer<T>::operator[]( const util::Indices& indices )
        {
            return this->weights[ indices ];
        }

        template <typename T>
        T& DenseLayer<T>::operator[]( const util::Index index )
        {
            return this->weights[ index ];
        }

        template <typename T>
        DenseLayer<T> DenseLayer<T>::operator+( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( "", this->input_shape, this->output_shape );
            result.weights = this->weights + other.weights;
            return result;
        }

        template <typename T>
        DenseLayer<T> DenseLayer<T>::operator-( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( "", this->input_shape, this->output_shape );
            result.weights = this->weights - other.weights;
            return result;
        }

        template <typename T>
        DenseLayer<T> DenseLayer<T>::operator*( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( "", this->input_shape, this->output_shape );
            result.weights = this->weights * other.weights;
            return result;
        }

        template <typename T>
        DenseLayer<T> DenseLayer<T>::operator/( const DenseLayer<T>& other ) const
        {
            DenseLayer<T> result( "", this->input_shape, this->output_shape );
            result.weights = this->weights / other.weights;
            return result;
        }

        template <typename T>
        DenseLayer<T>& DenseLayer<T>::operator+=( const DenseLayer<T>& other )
        {
            this->weights += other.weights;
            return *this;
        }

        template <typename T>
        DenseLayer<T>& DenseLayer<T>::operator-=( const DenseLayer<T>& other )
        {
            this->weights -= other.weights;
            return *this;
        }

        template <typename T>
        DenseLayer<T>& DenseLayer<T>::operator*=( const DenseLayer<T>& other )
        {
            this->weights *= other.weights;
            return *this;
        }

        template <typename T>
        DenseLayer<T>& DenseLayer<T>::operator/=( const DenseLayer<T>& other )
        {
            this->weights /= other.weights;
            return *this;
        }

    //: Methods
        template <typename T>
        void DenseLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape shape = this->weights.get_shape();
                    this->jacobian.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape shape = this->weights.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape shape = this->weights.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    this->velocity.resize( shape );
                    break;
                }
                default:
                {
                    throw std::invalid_argument( "Invalid training mode selection" );
                }
            }
        }

        template <typename T>
        void DenseLayer<T>::initialize( const InitializationType type, const DistributionType distribution )
        {
            util::Shape weights_shape = weights.get_shape();
            util::Size input_layer_size = weights_shape.width();
            util::Size output_layer_size = weights_shape.height();
            T variance = 0;
            T std_dev = 0;
            switch (type)
            {
                case InitializationType::xavier_glorot: { variance = 2.0 / (  input_layer_size + output_layer_size ); std_dev = sqrt( variance ); break; }
                case InitializationType::he: { variance = 2.0 / ( input_layer_size ); std_dev = sqrt( variance ); break; }
                case InitializationType::lecun: { variance = 1.0 / ( input_layer_size ); std_dev = sqrt( variance ); break; }
                default: {throw std::invalid_argument( "Initialization Type is invalid/uninitialized" ); break; }
            }

            switch (distribution)
            {
                case DistributionType::normal: { this->weights.fill_normal_distribution( 0, std_dev ); break; }
                /*
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
                */
                default: { throw std::invalid_argument( "Distribution Type is invalid/uninitialized" ); break; }
            };
        }

        template <typename T>
        void DenseLayer<T>::randomize( const T min, const T max )
        {
            this->weights.randomize( min, max );
        }

        template <typename T>
        void DenseLayer<T>::print() const
        {
            std::cout << "DenseLayer<T>.Print:" << std::endl << "{" << std::endl;
            std::cout << "Weights:" << std::endl;
            this->weights.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void DenseLayer<T>::print_jacobian() const
        {
            std::cout << "DenseLayer<T>.PrintJacobian:" << std::endl << "{" << std::endl;
            std::cout << "Jacobian:" << std::endl;
            this->jacobian.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void DenseLayer<T>::info() const
        {
            std::cout << "DenseLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Weights:" << std::endl;
            this->weights.info();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        util::Shape DenseLayer<T>::get_shape() const
        {
            return this->weights.get_shape();
        }

        template <typename T>
        util::Shape DenseLayer<T>::get_input_shape() const
        {
            return this->input_shape;
        }

        template <typename T>
        util::Shape DenseLayer<T>::get_output_shape() const
        {
            return this->output_shape;
        }

        template <typename T>
        const util::Tensor<T>& DenseLayer<T>::get_weights() const
        {
            return this->weights;
        }

        template <typename T>
        const util::Tensor<T>& DenseLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const util::Tensor<T>& DenseLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const util::Tensor<T>& DenseLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        void DenseLayer<T>::save_model( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_model_to_file( file );
            file.close();
        }

        template <typename T>
        void DenseLayer<T>::save_state( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->save_state_to_file( file );
            file.close();
        }

        template <typename T>
        void DenseLayer<T>::load_model( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file( file );
            file.close();
        }

        template <typename T>
        void DenseLayer<T>::load_state( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_state_from_file( file );
            file.close();
        }

        template <typename T>
        void DenseLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            this->weights.save( file );
        }

        template <typename T>
        void DenseLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Weight Tensor
            this->weights.save( file );

            // Save Jacobian Tensor
            this->jacobian.save( file );

            // Save momentum Tensor
            this->momentum.save( file );

            // Save velocity Tensor
            this->velocity.save( file );
        }

        template <typename T>
        void DenseLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            this->weights.load( file );
        }

        template <typename T>
        void DenseLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Weight Tensor
            this->weights.load( file );

            // Load Jacobian Tensor
            this->jacobian.load( file );

            // Load momentum Tensor
            this->momentum.load( file );

            // Load velocity Tensor
            this->velocity.load( file );
        }

        template <typename T>
        template <typename U, typename V>
        void DenseLayer<T>::inference( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer ) const
        {
            util::Shape input_shape = input_layer.get_shape();
            util::Size input_spacial_size = input_shape.distance( 0, input_shape.size() - 1 );
            util::Shape output_shape = output_layer.get_shape();
            util::Size output_spacial_size = output_shape.distance( 0, output_shape.size() - 1 );

            for ( util::Dim batch = 0; batch < input_shape.batches(); batch++ )
            {
                util::Index input_batch_index = batch * input_spacial_size;
                util::Index output_batch_index = batch * output_spacial_size;

                for ( util::Index output_index = 0; output_index < output_spacial_size; ++output_index )
                {
                    util::Index out_index = output_batch_index + output_index;
                    T sum = 0;

                    for ( util::Index input_index = 0; input_index < input_spacial_size; ++input_index )
                    {
                        util::Index in_index = input_batch_index + input_index;
                        util::Index weight_index = ( output_index * input_spacial_size ) + input_index;
                        sum += this->weights[ weight_index ] * input_layer[ in_index ];
                    }

                    output_layer[ out_index ] = sum;
                }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void DenseLayer<T>::inference_batch( const NodeLayer<U>& input_layer, NodeLayer<V>& output_layer, const util::Dim batch ) const
        {
            util::Shape input_shape = input_layer.get_shape();
            util::Size input_spacial_size = input_shape.distance( 0, input_shape.size() - 1 );
            util::Shape output_shape = output_layer.get_shape();
            util::Size output_spacial_size = output_shape.distance( 0, output_shape.size() - 1 );

            util::Index input_batch_index = batch * input_spacial_size;
            util::Index output_batch_index = batch * output_spacial_size;

            for ( util::Index output_index = 0; output_index < output_spacial_size; ++output_index )
            {
                util::Index out_index = output_batch_index + output_index;
                V sum = 0;

                for ( util::Index input_index = 0; input_index < input_spacial_size; ++input_index )
                {
                    util::Index in_index = input_batch_index + input_index;
                    util::Index weight_index = ( output_index * input_spacial_size ) + input_index;
                    sum += this->weights[ weight_index ] * input_layer[ in_index ];
                }

                output_layer[ out_index ] = sum;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void DenseLayer<T>::backpropagation( NodeLayer<U>& input_layer, const NodeLayer<V>& output_layer )
        {
            util::Shape input_shape = input_layer.get_shape();
            util::Size input_spacial_size = input_shape.distance( 0, input_shape.size() - 1 );
            util::Shape output_shape = output_layer.get_shape();
            util::Size output_spacial_size = output_shape.distance( 0, output_shape.size() - 1 );

            for ( util::Dim batch = 0; batch < input_shape.batches(); batch++ )
            {
                util::Index input_batch_index = batch * input_spacial_size;
                util::Index output_batch_index = batch * output_spacial_size;

                for ( util::Index input_index = 0; input_index < input_spacial_size; ++input_index )
                {
                    util::Index in_index = input_batch_index + input_index;
                    U input_node = input_layer[ in_index ];

                    U delta = 0;

                    for ( util::Index output_index = 0; output_index < output_spacial_size; ++output_index )
                    {
                        util::Index out_index = output_batch_index + output_index;
                        util::Index weight_index = ( output_index * input_spacial_size ) + input_index;

                        V out_delta = output_layer.get_delta( out_index );
                        T weight = this->weights[ weight_index ];

                        this->jacobian[ weight_index ] += input_node * out_delta;

                        delta += weight * out_delta;
                    }

                    input_layer.get_delta( in_index ) = delta;
                }
            }
        }

        template <typename T>
        void DenseLayer<T>::gradient_decent( const util::Dim batch_size, const StepSize step_size )
        {
            util::Size weights_size = this->weights.get_size();

            for ( util::Index index = 0; index < weights_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                this->weights[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( weights[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void DenseLayer<T>::gradient_decent_momentum( const util::Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            util::Size weights_size = this->weights.get_size();

            for ( util::Index index = 0; index < weights_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;
                T momentum_value = this->momentum[ index ];

                this->momentum[ index ] = gradient;

                this->weights[ index ] += step_size * (( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->weights[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void DenseLayer<T>::gradient_decent_adam( const util::Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            uint32_t step = this->time_step.get_count();
            Beta beta1_mp = 1 - pow( beta1, step );
            Beta beta2_mp = 1 - pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            util::Size weights_size = this->weights.get_size();

            for ( util::Index index = 0; index < weights_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                // Update momentum
                T momentum_value = this->momentum[ index ];
                momentum_value = ( beta1 * momentum_value ) + ( beta1_m * gradient );
                this->momentum[ index ] = momentum_value;

                // Update velocity
                T velocity_value = this->velocity[index];
                velocity_value = ( beta2 * velocity_value ) + ( beta2_m * ( gradient * gradient ) );
                this->velocity[ index ] = velocity_value;

                // Compute bias-corrected momentum and velocity
                T momentum_hat = momentum_value / beta1_mp;
                T velocity_hat = velocity_value / beta2_mp;

                // Update weight
                this->weights[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->weights[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& t : types)
    {
        std::string code =
R"(
    template class DenseLayer<{t}>;)";
        // Replace placeholders in the code with actual types
        size_t pos;
        while ((pos = code.find("{t}")) != std::string::npos)
            code.replace(pos, 3, t);

        file << code;
    }

    file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            for (const auto& v : types)
            {
                std::string code =
R"(
    template void DenseLayer<{t}>::inference( const NodeLayer<{u}>& input_layer, NodeLayer<{v}>& output_layer ) const;)";
                // Replace placeholders in the code with actual types
                size_t pos;
                while ((pos = code.find("{t}")) != std::string::npos)
                    code.replace(pos, 3, t);
                while ((pos = code.find("{u}")) != std::string::npos)
                    code.replace(pos, 3, u);
                while ((pos = code.find("{v}")) != std::string::npos)
                    code.replace(pos, 3, v);

                file << code;
            }
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            for (const auto& v : types)
            {
                std::string code =
R"(
    template void DenseLayer<{t}>::inference_batch( const NodeLayer<{u}>& input_layer, NodeLayer<{v}>& output_layer, const util::Dim batch_index ) const;)";
                // Replace placeholders in the code with actual types
                size_t pos;
                while ((pos = code.find("{t}")) != std::string::npos)
                    code.replace(pos, 3, t);
                while ((pos = code.find("{u}")) != std::string::npos)
                    code.replace(pos, 3, u);
                while ((pos = code.find("{v}")) != std::string::npos)
                    code.replace(pos, 3, v);

                file << code;
            }
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            for (const auto& v : types)
            {
                std::string code =
R"(
    template void DenseLayer<{t}>::backpropagation( NodeLayer<{u}>& input_layer, const NodeLayer<{v}>& output_layer );)";
                // Replace placeholders in the code with actual types
                size_t pos;
                while ((pos = code.find("{t}")) != std::string::npos)
                    code.replace(pos, 3, t);
                while ((pos = code.find("{u}")) != std::string::npos)
                    code.replace(pos, 3, u);
                while ((pos = code.find("{v}")) != std::string::npos)
                    code.replace(pos, 3, v);

                file << code;
            }
        }
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}