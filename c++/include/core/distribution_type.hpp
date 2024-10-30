// Copyright 2024 Shane Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DISTRIBUTION_TYPE_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DISTRIBUTION_TYPE_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    enum class DistributionType : i8
    {
        normal,
        poisson,
        binomial,
        exponential,
        uniform,
        bernoulli,
        beta,
        weibull,
        gamma,
        chi_squared,
        log_normal,
        f,
        discrete_uniform,
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DISTRIBUTION_TYPE_HPP_
