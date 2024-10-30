// Copyright 2024 Shane W. Mulcahy

//: C++ Headers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/activation_layer/activation_layer.cpp");
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

//: Library Headers
#include <bewusstsein_util.hpp>

//: Project Headers
#include "c++/bewusstsein_neural_networks/source/core/step_size/step_size.hpp"
#include "c++/bewusstsein_neural_networks/source/core/beta/beta.hpp"
#include "c++/bewusstsein_neural_networks/source/core/epsilon/epsilon.hpp"
#include "c++/bewusstsein_neural_networks/source/core/training_mode/training_mode.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/ops/bewusstsein_nn_ops.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/activation_layer/activation_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        ActivationLayer<T>::ActivationLayer() :
            BaseLayer( LayerType::activation_layer , "" ) {}

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type ) :
            BaseLayer( LayerType::activation_layer, name ), type( type ), parameters( { 1, 1, 1, 1 } ) {}

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, float parameter ) :
            BaseLayer( LayerType::activation_layer, name ), type( type ), parameters( { 1, 1, 1, 1 }, parameter ) {}

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, const util::Shape& shape ) :
            BaseLayer( LayerType::activation_layer, name ), type(type), parameters( util::Shape( shape, 4 ) )
        {
            if (( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }
        }

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, const util::Shape& shape, float parameter ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( util::Shape( shape, 4 ), parameter )
        {
            if ( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( " Biaslayer can have at least 1 and at most 4 dimensions" );
            }
        }

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, const util::Shape& shape, const std::vector<T>& data ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( util::Shape( shape, 4 ), data )
        {
            if ( ( shape.size() > 4 ) || (shape.size() < 1  ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }
        }

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, float parameter, TrainingMode training_mode ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( { 1, 1, 1, 1 }, parameter )
        {
            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const std::string& name, const ActivationType type, const util::Shape& shape, TrainingMode training_mode ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( util::Shape( shape, 4 ) )
        {
            if ( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument("Biaslayer can have at least 1 and at most 4 dimensions");
            }

            this->allocate_training_memory( training_mode );
        }
        */

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const ActivationType type, const util::Shape& shape, const T parameter, const TrainingMode training_mode ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( util::Shape( shape, 4 ), parameter )
        {
            if ( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        ActivationLayer<T>::ActivationLayer( const ActivationType type, const util::Shape& shape, const std::vector<T>& data, TrainingMode training_mode ) :
            BaseLayer( LayerType::activation_layer ), type( type ), parameters( util::Shape( shape, 4 ), data )
        {
            if ( ( shape.size() > 4)  || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }

    //: Destructors
        template <typename T>
        ActivationLayer<T>::~ActivationLayer() {}

    //: Methods
        template <typename T>
        void ActivationLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape shape = this->parameters.get_shape();
                    this->jacobian.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape shape = this->parameters.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape shape = this->parameters.get_shape();
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
        util::Shape ActivationLayer<T>::get_shape() const
        {
            return parameters.get_shape();
        }

        template <typename T>
        util::Size ActivationLayer<T>::get_size() const
        {
            return parameters.get_size();
        }

        template <typename T>
        ActivationType ActivationLayer<T>::get_type() const
        {
            return this->type;
        }

        template <typename T>
        const util::Tensor<T>& ActivationLayer<T>::get_parameters() const
        {
            return this->parameters;
        }

        template <typename T>
        const util::Tensor<T>& ActivationLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const util::Tensor<T>& ActivationLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const util::Tensor<T>& ActivationLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        void ActivationLayer<T>::info() const
        {
            std::cout << "ActivationLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Parameters:" << std::endl;
            parameters.info();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ActivationLayer<T>::print() const
        {
            std::cout << "ActivationLayer<T>.Print:" << std::endl << "{" << std::endl;
            std::cout << "Parameters:" << std::endl;
            parameters.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ActivationLayer<T>::print_jacobian() const
        {
            std::cout << "ActivationLayer<T>.PrintJacobian:" << std::endl << "{" << std::endl;
            std::cout << "Jacobian:" << std::endl;
            jacobian.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void ActivationLayer<T>::save_model( const std::string& file_name ) const
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
        void ActivationLayer<T>::save_state( const std::string& file_name ) const
        {
            std::ofstream file( file_name, std::ios::binary | std::ios_base::out );
            if ( !file.is_open() )
            {
                throw std::runtime_error("Could not open output file: " + file_name);
            }
            this->save_state_to_file(file);
            file.close();
        }

        template <typename T>
        void ActivationLayer<T>::load_model( const std::string& file_name )
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
        void ActivationLayer<T>::load_state( const std::string& file_name )
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
        void ActivationLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Activation Type
            file.write(reinterpret_cast<const char*>( &this->type ), sizeof( ActivationType ) );

            // Save Parameters Tensor
            this->parameters.save( file );
        }

        template <typename T>
        void ActivationLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name To File
            this->save_layer_type_and_name_to_file( file );

            // Save Activation Type
            file.write( reinterpret_cast<const char*>( &this->type ), sizeof( ActivationType ) );

            // Save Parameters Tensor
            this->parameters.save( file );

            // Save Jacobian Tensor
            this->jacobian.save( file );

            // Save momentum Tensor
            this->momentum.save( file );

            // Save velocity Tensor
            this->velocity.save( file );

            // Save Time Variable
            file.write( reinterpret_cast<const char*>( &this->t ), sizeof( int ) );
        }

        template <typename T>
        void ActivationLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Activation Type
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( ActivationType ) );

            // Load Parameters Tensor
            this->parameters.load( file );
        }

        template <typename T>
        void ActivationLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Activation Type
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( ActivationType ) );

            // Load Parameters Tensor
            this->parameters.load( file );

            // Save Jacobian Tensor
            this->jacobian.load( file );

            // Save momentum Tensor
            this->momentum.load( file );

            // Save velocity Tensor
            this->velocity.load( file );

            // Save Time Variable
            file.read( reinterpret_cast<char*>( &this->t ), sizeof( int ) );
        }

        template <typename T>
        template <typename U>
        void ActivationLayer<T>::inference( NodeLayer<U>& layer ) const
        {
            switch (this->type)
            {
                case ActivationType::sigmoid: { util::operation( layer.get_nodes(), sigmoid<U>() ); break; }

                case ActivationType::tanh: { util::operation( layer.get_nodes(), htan<U>() ); break; }

                case ActivationType::relu: { util::operation( layer.get_nodes(), relu<U>() ); break; }

                case ActivationType::leakyrelu: { util::operation( layer.get_nodes(), leaky_relu<U, T>(), parameters[ 0 ] ); break; }

                case ActivationType::elu: { util::operation( layer.get_nodes(), elu<U, T>(), parameters[ 0 ] ); break; }

                case ActivationType::swish: { util::operation( layer.get_nodes(), swish<U>() ); break; }

                case ActivationType::eswish: { util::operation( layer.get_nodes(), eswish<U, T>(), parameters[ 0 ] ); break; }

                case ActivationType::pararelu: { util::channel_operation( layer.get_nodes(), leaky_relu<U, T>(), parameters ); break; }

                default: { throw std::invalid_argument( "Activation Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        template <typename U>
        void ActivationLayer<T>::backpropagation(NodeLayer<U>& layer)
        {
            switch (this->type)
            {
                case ActivationType::sigmoid: { util::operation( layer.get_nodes(), layer.get_delta(), d_sigmoid<U>() ); break; }

                case ActivationType::tanh: { util::operation( layer.get_nodes(), layer.get_delta(), d_htan<U>() ); break; }

                case ActivationType::relu: { util::operation( layer.get_nodes(), layer.get_delta(), d_relu<U>() ); break; }

                case ActivationType::leakyrelu: { util::operation( layer.get_nodes(), layer.get_delta(), d_leaky_relu<U, T>(), parameters[ 0 ] ); break; }

                case ActivationType::elu: { util::operation( layer.get_nodes(), layer.get_delta(), d_elu<U, T>(), parameters[ 0 ]); break; }

                case ActivationType::swish: { util::operation( layer.get_nodes(), layer.get_delta(), d_swish<U>() ); break; }

                case ActivationType::eswish: { util::operation( layer.get_nodes(), layer.get_delta(), d_eswish<U, T>(), parameters[ 0 ] ); break; }

                case ActivationType::pararelu: { util::channel_operation( layer.get_nodes(), layer.get_delta(), d_leaky_relu<U, T>(), parameters, jacobian ); break; }

                default: { throw std::invalid_argument( "Activation Type is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        void ActivationLayer<T>::gradient_decent(  const util::Dim batch_size, const StepSize step_size)
        {
            util::Size layer_size = this->parameters.get_size();

            for ( util::Index index = 0; index < layer_size; ++index )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                this->parameters[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->parameters[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void ActivationLayer<T>::gradient_decent_momentum(const util::Dim batch_size, const StepSize step_size, const StepSize momentum_step_size)
        {
            util::Size layer_size = this->parameters.get_size();

            for (util::Index index = 0; index < layer_size; ++index)
            {
                T gradient = this->jacobian[ index ] / batch_size;
                T momentum_value = this->momentum[index];

                this->momentum[ index ] = gradient;

                this->parameters[ index ] += step_size * ( ( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->parameters[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void ActivationLayer<T>::gradient_decent_adam( const util::Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            uint32_t step = this->time_step.get_count();
            Beta beta1_mp = 1 - pow( beta1, step );
            Beta beta2_mp = 1 - pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            util::Size layer_size = this->parameters.get_size();

            for ( util::Index index = 0; index < layer_size; ++index )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                // Update momentum
                T momentum_value = this->momentum[ index ];
                momentum_value = ( beta1 * momentum_value ) + ( beta1_m * gradient );
                this->momentum[ index ] = momentum_value;

                // Update velocity
                T velocity_value = this->velocity[ index ];
                velocity_value = ( beta2 * velocity_value ) + ( beta2_m * ( gradient * gradient ) );
                this->velocity[ index ] = velocity_value;

                // Compute bias-corrected momentum and velocity
                T momentum_hat = momentum_value / beta1_mp;
                T velocity_hat = velocity_value / beta2_mp;

                // Update weight
                this->parameters[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->parameters[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
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
    template class ActivationLayer<{t}>;)";
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
            std::string code =
R"(
    template void ActivationLayer<{t}>::inference( NodeLayer<{u}>& layer ) const;)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{t}")) != std::string::npos)
                code.replace(pos, 3, t);
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);

            file << code;
        }
    }

        file <<
R"(
)";

    for (const auto& t : types)
    {
        for (const auto& u : types)
        {
            std::string code =
R"(
    template void ActivationLayer<{t}>::backpropagation(NodeLayer<{u}>& layer);)";
            // Replace placeholders in the code with actual types
            size_t pos;
            while ((pos = code.find("{t}")) != std::string::npos)
                code.replace(pos, 3, t);
            while ((pos = code.find("{u}")) != std::string::npos)
                code.replace(pos, 3, u);

            file << code;
        }
    }

    file <<
R"(
} // namespace nn
)";

    return 0;
}