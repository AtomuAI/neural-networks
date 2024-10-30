// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_COST_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_COST_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <complex>
#include <vector>
#include <string>
#include <limits>
#include <fstream>
#include <functional>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/error.hpp"
#include "bewusstsein_neural_networks/c++/include/core/counter.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/base.hpp"
#include "bewusstsein_neural_networks/c++/include/layer/node.hpp"
#include "bewusstsein_neural_networks/c++/include/core/cost.hpp"

namespace nn
{
    enum class CostType : u8
    {
        mean_squared_error,
        categorical_cross_entropy,
        //softmax_categorical_cross_entropy,
        hellinger_distance,
        kullback_leibler_divergence,
        generalized_kullback_leibler_divergence,
        itakura_saito_distance
    };

    template <CostType C>
    class CostLayer : public BaseLayer
    {
        //: Members
        public:
            static const CostType   type = C;
        protected:
            Dim                     num_examples;

        //: Constructors
        public:
            CostLayer
            (
                const Dim num_examples    = 0
            );

        //: Destructors
        public:
            virtual ~CostLayer();

        //: Methods
        public:
            Dim   get_num_examples    () const;

            template <typename U, typename V>
            Error inference                 ( NodeLayer<U>& layer, const NodeLayer<V>& target ) const;
            template <typename U, typename V>
            Error backpropagation        ( NodeLayer<U>& layer, const NodeLayer<V>& target ) const;

        private:
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::mean_squared_error );
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::categorical_cross_entropy );
            //template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::softmax_categorical_cross_entropy );
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::hellinger_distance );
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::kullback_leibler_divergence );
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::generalized_kullback_leibler_divergence );
            template <typename U, typename V> inline U cost     ( const V target, const U value ) const requires ( C == CostType::itakura_saito_distance );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::mean_squared_error );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::categorical_cross_entropy );
            //template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::softmax_categorical_cross_entropy );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::hellinger_distance );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::kullback_leibler_divergence );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::generalized_kullback_leibler_divergence );
            template <typename U, typename V> inline U d_cost   ( const V target, const U value ) const requires ( C == CostType::itakura_saito_distance );
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_LAYER_COST_HPP_
