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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/bias_layer/bias_layer.cpp");
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
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/bias_layer/bias_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        template <typename T>
        BiasLayer<T>::BiasLayer( const util::Shape& shape, const T scalar, const TrainingMode training_mode ) :
            BaseLayer( LayerType::bias_layer ), nodes( util::Shape( shape, 4 ), scalar )
        {
            if( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }

        template <typename T>
        BiasLayer<T>::BiasLayer( const util::Shape& shape, const std::vector<T>& data, TrainingMode training_mode ) :
            BaseLayer( LayerType::bias_layer ), nodes( util::Shape( shape, 4 ), data )
        {
            if( ( shape.size() > 4 ) || ( shape.size() < 1 ) )
            {
                throw std::invalid_argument( "Biaslayer can have at least 1 and at most 4 dimensions" );
            }

            this->allocate_training_memory( training_mode );
        }

    //: Destructors
        template <typename T>
        BiasLayer<T>::~BiasLayer() {}

    //: Operators
        template <typename T>
        T BiasLayer<T>::operator[]( const util::Indices& indices ) const
        {
            return this->nodes[ indices ];
        }

        template <typename T>
        T BiasLayer<T>::operator[]( const util::Index index ) const
        {
            return this->nodes[ index ];
        }

        template <typename T>
        T& BiasLayer<T>::operator[]( const util::Indices& indices )
        {
            return this->nodes[ indices ];
        }

        template <typename T>
        T& BiasLayer<T>::operator[]( const util::Index index )
        {
            return this->nodes[ index ];
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator+(  const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + other.nodes;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator-( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - other.nodes;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator*( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * other.nodes;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator/( const BiasLayer<T>& other ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / other.nodes;
            return result;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator+=( const BiasLayer<T>& other )
        {
            this->nodes += other.nodes;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator-=( const BiasLayer<T>& other )
        {
            this->nodes -= other.nodes;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator*=( const BiasLayer<T>& other )
        {
            this->nodes *= other.nodes;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator/=( const BiasLayer<T>& other )
        {
            this->nodes /= other.nodes;
            return *this;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator+( const T scalar ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes + scalar;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator-( const T scalar ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes - scalar;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator*( const T scalar ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes * scalar;
            return result;
        }

        template <typename T>
        BiasLayer<T> BiasLayer<T>::operator/( const T scalar ) const
        {
            BiasLayer<T> result( this->nodes.get_shape() );
            result.nodes = this->nodes / scalar;
            return result;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator+=( const T scalar )
        {
            this->nodes += scalar;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator-=( const T scalar )
        {
            this->nodes -= scalar;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator*=( const T scalar )
        {
            this->nodes *= scalar;
            return *this;
        }

        template <typename T>
        BiasLayer<T>& BiasLayer<T>::operator/=( const T scalar )
        {
            this->nodes /= scalar;
            return *this;
        }


    //: Methods
        template <typename T>
        void BiasLayer<T>::allocate_training_memory( TrainingMode training_mode )
        {
            switch ( training_mode )
            {
                case TrainingMode::off:
                {
                    break;
                }
                case TrainingMode::normal:
                {
                    util::Shape shape = this->nodes.get_shape();
                    this->jacobian.resize( shape );
                    break;
                }
                case TrainingMode::momentum:
                {
                    util::Shape shape = this->nodes.get_shape();
                    this->jacobian.resize( shape );
                    this->momentum.resize( shape );
                    break;
                }
                case TrainingMode::adam:
                {
                    util::Shape shape = this->nodes.get_shape();
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
        const util::Shape& BiasLayer<T>::get_shape() const
        {
            return this->nodes.get_shape();
        }

        template <typename T>
        util::Size BiasLayer<T>::get_size() const
        {
            return this->nodes.get_size();
        }

        template <typename T>
        const util::Tensor<T>& BiasLayer<T>::get_nodes() const
        {
            return this->nodes;
        }

        template <typename T>
        const util::Tensor<T>& BiasLayer<T>::get_jacobian() const
        {
            return this->jacobian;
        }

        template <typename T>
        const util::Tensor<T>& BiasLayer<T>::get_momentum() const
        {
            return this->momentum;
        }

        template <typename T>
        const util::Tensor<T>& BiasLayer<T>::get_velocity() const
        {
            return this->velocity;
        }

        template <typename T>
        void BiasLayer<T>::info() const
        {
            std::cout << "BiasLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Nodes:" << std::endl;
            this->nodes.info();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void BiasLayer<T>::print() const
        {
            std::cout << "BiasLayer<T>.Print:" << std::endl << "{" << std::endl;
            std::cout << "Nodes:" << std::endl;
            this->nodes.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void BiasLayer<T>::print_jacobian() const
        {
            std::cout << "BiasLayer<T>.Info:" << std::endl << "{" << std::endl;
            std::cout << "Jacobian:" << std::endl;
            this->jacobian.print();
            std::cout << "}" << std::endl;
        }

        template <typename T>
        void BiasLayer<T>::reshape( const util::Shape& shape )
        {
            this->nodes.reshape( shape );
        }

        template <typename T>
        void BiasLayer<T>::fill( const T value )
        {
            this->nodes.fill( value );
        }

        template <typename T>
        void BiasLayer<T>::zero()
        {
            this->nodes.zero();
        }

        template <typename T>
        void BiasLayer<T>::randomize( const T min, const T max )
        {
            this->nodes.randomize( min, max );
        }

        template <typename T>
        void BiasLayer<T>::save_model( const std::string& file_name ) const
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
        void BiasLayer<T>::save_state( const std::string& file_name ) const
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
        void BiasLayer<T>::load_model( const std::string& file_name )
        {
            std::ifstream file( file_name, std::ios::binary | std::ios_base::in );
            if ( !file.is_open() )
            {
                throw std::runtime_error( "Could not open output file: " + file_name );
            }
            this->load_model_from_file(file);
            file.close();
        }

        template <typename T>
        void BiasLayer<T>::load_state( const std::string& file_name )
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
        void BiasLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Parameters Tensor
            this->nodes.save( file );
        }

        template <typename T>
        void BiasLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Type and Name To File
            this->save_layer_type_and_name_to_file( file );

            // Save Parameters Tensor
            this->nodes.save( file );

            // Save Jacobian Tensor
            this->jacobian.save( file );

            // Save momentum Tensor
            this->momentum.save( file );

            // Save velocity Tensor
            this->velocity.save( file );

            // Save Time Variable
            file.write(reinterpret_cast<const char*>( &this->t ), sizeof( int ));
        }

        template <typename T>
        void BiasLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Parameters Tensor
            this->nodes.load( file );
        }

        template <typename T>
        void BiasLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Type and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Parameters Tensor
            this->nodes.load( file );

            // Save Jacobian Tensor
            this->jacobian.load( file );

            // Save momentum Tensor
            this->momentum.load( file );

            // Save velocity Tensor
            this->velocity.load( file );

            // Save Time Variable
            file.read(reinterpret_cast<char*>( &this->t ), sizeof( int ));
        }

        template <typename T>
        template <typename U>
        void BiasLayer<T>::inference( NodeLayer<U>& layer ) const
        {
            util::Shape shape = layer.get_shape();
            util::Size spacial_size = shape.distance( 0, shape.size() - 1 );

            for ( util::Dim batch = 0; batch < shape.batches(); batch++ )
            {
                util::Index batch_index = batch * spacial_size;
                for ( util::Index spacial = 0; spacial < spacial_size; spacial++ )
                {
                    util::Index index = batch_index + spacial;
                    layer[ index] += nodes[ spacial ];
                }
            }
        }

        template <typename T>
        template <typename U>
        void BiasLayer<T>::inference_batch( NodeLayer<U>& layer, const util::Dim batch ) const
        {
            util::Shape shape = layer.get_shape();
            util::Size spacial_size = shape.distance( 0, shape.size() - 1 );

            util::Index batch_index = batch * spacial_size;
            for ( util::Index spacial = 0; spacial < spacial_size; spacial++ )
            {
                util::Index index = batch_index + spacial;
                layer[ index ] += nodes[ spacial ];
            }
        }

        template <typename T>
        template <typename U>
        void BiasLayer<T>::backpropagation( const NodeLayer<U>& layer )
        {
            util::Shape shape = layer.get_shape();
            util::Size spacial_size = shape.distance( 0, shape.size() - 1 );

            for ( util::Dim batch = 0; batch < shape.batches(); batch++ )
            {
                util::Index batch_index = batch * spacial_size;
                for ( util::Index spacial = 0; spacial < spacial_size; spacial++ )
                {
                    util::Index index = batch_index + spacial;
                    this->jacobian[ spacial ] += layer.get_delta( index );
                }
            }
        }

        template <typename T>
        void BiasLayer<T>::gradient_decent( const util::Dim batch_size, const StepSize step_size )
        {
            util::Size layer_size = this->nodes.get_size();

            for ( util::Index index = 0; index < layer_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;

                this->nodes[ index ] += step_size * gradient;

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->nodes[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void BiasLayer<T>::gradient_decent_momentum( const util::Dim batch_size, const StepSize step_size, const StepSize momentum_step_size )
        {
            util::Size layer_size = this->nodes.get_size();

            for ( util::Index index = 0; index < layer_size; index++ )
            {
                T gradient = this->jacobian[ index ] / batch_size;
                T momentum_value = this->momentum[ index ];

                this->momentum[ index ] = gradient;

                this->nodes[ index ] += step_size * ( ( momentum_step_size * momentum_value ) + gradient );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->nodes[ index ] ) ) { throw std::invalid_argument( "Value is NaN" ); }
                #endif
            }

            this->jacobian.zero();
            this->time_step.tick();
        }

        template <typename T>
        void BiasLayer<T>::gradient_decent_adam( const util::Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
        {
            uint32_t step = this->time_step.get_count();
            Beta beta1_mp = 1 - pow( beta1, step );
            Beta beta2_mp = 1 - pow( beta2, step );
            Beta beta1_m = 1 - beta1;
            Beta beta2_m = 1 - beta2;

            util::Size layer_size = this->nodes.get_size();

            for ( util::Index index = 0; index < layer_size; index++ )
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
                this->nodes[ index ] += step_size * ( momentum_hat / ( sqrt( velocity_hat ) + epsilon ) );

                #ifdef BEWUSSTSEIN_NN_DEBUG
                    if ( isnan( this->nodes[ index ] ) ) {  throw std::invalid_argument("Value is NaN" ); }
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
    template class BiasLayer<{t}>;)";
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
    template void BiasLayer<{t}>::inference( NodeLayer<{u}>& input_layer ) const;)";
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
    template void BiasLayer<{t}>::inference_batch( NodeLayer<{u}>& layer, const util::Dim batch ) const;)";
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
    template void BiasLayer<{t}>::backpropagation( const NodeLayer<{u}>& input_layer );)";
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