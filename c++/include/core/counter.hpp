// Copyright 2024 Shane Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COUNTER_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COUNTER_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    template<typename T>
    class Counter
    {
        // Members
        private:
            T count;

        // Constructors
        public:
            Counter();
            explicit Counter( T count );

        // Destructors
        public:
            virtual ~Counter();

        // Methods
        public:
            void reset();
            void tick();
            T get_count() const;
            void set_count( T count );
    };

    template <typename T>
    Counter<T>::Counter() : count( 0 ) {}

    template <typename T>
    Counter<T>::Counter( T count ) : count( count ) {}

    template <typename T>
    Counter<T>::~Counter() {}

    template <typename T>
    void Counter<T>::reset() { this->count = 0; }

    template <typename T>
    void Counter<T>::tick() { ++this->count; }

    template <typename T>
    T Counter<T>::get_count() const { return this->count; }

    template <typename T>
    void Counter<T>::set_count( T count ) { this->count = count; }

} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_COUNTER_HPP_
