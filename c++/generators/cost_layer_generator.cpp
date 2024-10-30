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

    std::ofstream file("c++/bewusstsein_neural_networks/source/layers/cost_layer/cost_layer.cpp");
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
#include "c++/bewusstsein_neural_networks/source/layers/base_layer/base_layer.hpp"
#include "c++/bewusstsein_neural_networks/source/layers/node_layer/node_layer.hpp"

//: This Header
#include "c++/bewusstsein_neural_networks/source/layers/cost_layer/cost_layer.hpp"

//: Definitions
namespace nn
{
    //: Constructors
        /*
        template <typename T>
        CostLayer<T>::CostLayer() :
            BaseLayer( LayerType::cost_layer, "" ) {}
        */

        template <typename T>
        CostLayer<T>::CostLayer( const CostType type, const util::Dim num_examples ) :
            BaseLayer( LayerType::cost_layer ), type( type ), num_examples( num_examples ) {}

    //: Destructors
        template <typename T>
        CostLayer<T>::~CostLayer() {}


    //: Methods
        template <typename T>
        CostType CostLayer<T>::get_CostType() const
        {
            return this->error;
        }

        template <typename T>
        util::Dim CostLayer<T>::get_num_examples() const
        {
            return this->error;
        }

        template <typename T>
        T CostLayer<T>::get_error() const
        {
            return this->error;
        }

        template <typename T>
        void CostLayer<T>::save_model( const std::string& file_name ) const
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
        void CostLayer<T>::save_state( const std::string& file_name ) const
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
        void CostLayer<T>::load_model( const std::string& file_name )
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
        void CostLayer<T>::load_state( const std::string& file_name )
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
        void CostLayer<T>::save_model_to_file( std::ofstream& file ) const
        {
            // Save Layer Uype and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Cost Uype
            file.write( reinterpret_cast<const char*>( &this->type ), sizeof( CostType ) );
        }

        template <typename T>
        void CostLayer<T>::save_state_to_file( std::ofstream& file ) const
        {
            // Save Layer Uype and Name
            this->save_layer_type_and_name_to_file( file );

            // Save Cost Uype
            file.write( reinterpret_cast<const char*>( &this->type ), sizeof( CostType ) );

            // Save Number of Examples
            file.write( reinterpret_cast<const char*>( &this->num_examples ), sizeof( int ) );
        }

        template <typename T>
        void CostLayer<T>::load_model_from_file( std::ifstream& file )
        {
            // Load Layer Uype and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Cost Uype
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( CostType ) );
        }

        template <typename T>
        void CostLayer<T>::load_state_from_file( std::ifstream& file )
        {
            // Load Layer Uype and Name
            this->load_layer_type_and_name_from_file( file );

            // Load Cost Uype
            file.read( reinterpret_cast<char*>( &this->type ), sizeof( CostType ) );

            // Load Number of Examples
            file.read( reinterpret_cast<char*>( &this->num_examples ), sizeof( int ) );
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::inference( const NodeLayer<U>& layer, const NodeLayer<V>& target )
        {
            switch ( this->type )
            {
                case CostType::mean_squared_error: { this->error = layer_mean_squared_error( layer, target, this->num_examples ); break; }

                case CostType::categorical_cross_entropy: { this->error = layer_categorical_cross_entropy( layer, target, this->num_examples ); break; }

                case CostType::softmax_categorical_cross_entropy: { this->error = layer_softmax_categorical_cross_entropy( layer, target, this->num_examples ); break; }

                case CostType::hellinger_distance: { this->error = layer_hellinger_distance( layer, target, this->num_examples ); break; }

                case CostType::kullback_leibler_divergence: { this->error = layer_kullback_leibler_divergence( layer, target, this->num_examples ); break; }

                case CostType::generalized_kullback_leibler_divergence: { this->error = layer_generalized_kullback_leibler_divergence( layer, target, this->num_examples ); break; }

                case CostType::itakura_saito_distance: { this->error = layer_itakura_saito_distance(layer, target, this->num_examples ); break; }

                default: {throw std::invalid_argument( "Cost Uype is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::backpropagation( NodeLayer<U>& layer, const NodeLayer<V>& target ) const
        {
            switch (this->type)
            {
                case CostType::mean_squared_error: { layer_mean_squared_error_derivative( layer, target, this->num_examples ); break; }

                case CostType::categorical_cross_entropy: { layer_categorical_cross_entropy_derivative( layer, target, this->num_examples ); break; }

                case CostType::softmax_categorical_cross_entropy: { layer_softmax_categorical_cross_entropy_derivative( layer, target, this->num_examples ); break; }

                case CostType::hellinger_distance: { layer_hellinger_distance_derivative( layer, target, this->num_examples ); break; }

                case CostType::kullback_leibler_divergence: { layer_kullback_leibler_divergence_derivative( layer, target, this->num_examples ); break; }

                case CostType::generalized_kullback_leibler_divergence: { layer_generalized_kullback_leibler_divergence_derivative( layer, target, this->num_examples ); break; }

                case CostType::itakura_saito_distance: { layer_itakura_saito_distance_derivative(layer, target, this->num_examples ); break; }

                default: {throw std::invalid_argument( "Cost Uype is invalid/uninitialized" ); break; }
            }
        }

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::mean_squared_error( const V target, const U value ) const
        {
            return pow( ( target - value ), 2 );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::categorical_cross_entropy( const V target, const U value ) const
        {
            return target * ( ( value > 0 ) ? log( value ) : log( 1e-8 ) );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::hellinger_distance( const V target, const U value ) const
        {
            return 0;
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::kullback_leibler_divergence( const V target, const U value ) const
        {
            return 0;
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::generalized_kullback_leibler_divergence( const V target, const U value ) const
        {
            return 0;
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::itakura_saito_distance( const V target, const U value ) const
        {
            return 0;
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::mean_squared_error_derivative( const V target, const U value ) const
        {
            return ( target - value );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::categorical_cross_entropy_derivative( const V target, const U value ) const
        {
            return -( target * ( ( value > 0 ) ? log( value ) : log( 1e-8 ) ) );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::softmax_categorical_cross_entropy_derivative( const V target, const U value ) const
        {
            return ( target - value );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::hellinger_distance_derivative( const V target, const U value ) const
        {
            return ( ( sqrt( target ) - sqrt( value ) ) / ( /*sqrt(2)*/( 1.4142135623730950*sqrt( target ) ) + 1e-8 ) );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::kullback_leibler_divergence_derivative( const V target, const U value ) const
        {
            return -( value / ( target + 1e-8 ) );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::generalized_kullback_leibler_divergence_derivative( const V target, const U value ) const
        {
            return ( ( target - value ) / ( target + 1e-8 ) );
        };

        template <typename T>
        template <typename U, typename V>
        U CostLayer<T>::itakura_saito_distance_derivative( const V target, const U value ) const
        {
            return ( ( target - value ) / ( pow( target, 2 ) + 1e-8 ) );
        };

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_mean_squared_error( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                sum += this->mean_squared_error( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_categorical_cross_entropy( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples  ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                sum += this->categorical_cross_entropy( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_softmax_categorical_cross_entropy( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            util::Shape layer_shape = layer.get_shape();
            int spacial_size = std::accumulate( layer_shape.begin(), layer_shape.end() - 1, 1, std::multiplies<int>() );

            U sum = 0;

            for ( int i = 0; i < layer_shape.batches(); ++i )
            {
                U exp_sum = 0;
                int batch_index = i * spacial_size;
                for ( int j = 0; j < spacial_size; ++j )
                {
                    int index = batch_index + j;
                    U value = exp(layer[ index ]);
                    layer[ index ] = value;
                    exp_sum += value;
                }
                for ( int j = 0; j < spacial_size; ++j )
                {
                    int index = batch_index + j;
                    if ( ( exp_sum != INFINITY ) && ( exp_sum != 0 ) && ( exp_sum != -INFINITY ) )
                    {
                        layer[ index ] /= exp_sum;
                    }
                    else if ( exp_sum == INFINITY )
                    {
                        layer[ index ] = 0;
                    }
                    else
                    {
                        layer[ index ] = std::numeric_limits<float>::max();
                    }

                    sum += this->categorical_cross_entropy( target[ index ], layer[ index ] );
                }
            }

            return sum / layer.get_size();
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_hellinger_distance( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                sum += this->hellinger_distance( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_kullback_leibler_divergence( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[i];
                const U target_value = target[i];
                sum += this->kullback_leibler_divergence( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_generalized_kullback_leibler_divergence( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                sum += this->generalized_kullback_leibler_divergence( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        T CostLayer<T>::layer_itakura_saito_distance( const NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            U sum = 0;
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                sum += this->itakura_saito_distance( target_value, node_value );
            }
            return sum / size;
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_mean_squared_error_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->mean_squared_error_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_categorical_cross_entropy_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->categorical_cross_entropy_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_softmax_categorical_cross_entropy_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->softmax_categorical_cross_entropy_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_hellinger_distance_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->hellinger_distance_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_kullback_leibler_divergence_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->kullback_leibler_divergence_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_generalized_kullback_leibler_divergence_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->generalized_kullback_leibler_divergence_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }

        template <typename T>
        template <typename U, typename V>
        void CostLayer<T>::layer_itakura_saito_distance_derivative( NodeLayer<U>& layer, const NodeLayer<V>& target, const util::Dim num_examples ) const
        {
            int size = layer.get_size();
            for ( util::Index i = 0; i < size; ++i )
            {
                const U node_value = layer[ i ];
                const U target_value = target[ i ];
                U delta = this->itakura_saito_distance_derivative( target_value, node_value ) / num_examples;
                layer.get_delta( i ) = delta;
            }
        }
} // namespace nn

//: Specializations
namespace nn
{)";

    for (const auto& t : types)
    {
        std::string code =
R"(
    template class CostLayer<{t}>;)";
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
    template void CostLayer<{t}>::inference( const NodeLayer<{u}>& layer, const NodeLayer<{v}>& target );)";
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
    template void CostLayer<{t}>::backpropagation( NodeLayer<{u}>& layer, const NodeLayer<{v}>& target ) const;)";
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