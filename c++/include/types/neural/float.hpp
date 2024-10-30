// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

namespace nn
{
    template <typename T>
    class Float;

    template<typename T> struct is_neural_float;

    template <typename T>
    class Float
    {
        private:
            T value;

        public:
            constexpr Float();
            constexpr Float( T value );

        public:
                                    constexpr void      operator=   ( const Float<T> other );
            template <typename U>   constexpr void      operator=   ( const Complex<Float<U>> other );
                                    constexpr Float<T>  operator+   ( const Float<T> other ) const;
                                    constexpr Float<T>  operator-   ( const Float<T> other ) const;
                                    constexpr Float<T>  operator*   ( const Float<T> other ) const;
                                    constexpr Float<T>  operator/   ( const Float<T> other ) const;
                                    constexpr Float<T>& operator+=  ( const Float<T> other );
                                    constexpr Float<T>& operator-=  ( const Float<T> other );
                                    constexpr Float<T>& operator*=  ( const Float<T> other );
                                    constexpr Float<T>& operator/=  ( const Float<T> other );
                                    constexpr bool      operator==  ( const Float<T> other ) const;
                                    constexpr bool      operator!=  ( const Float<T> other ) const;
                                    constexpr bool      operator>   ( const Float<T> other ) const;
                                    constexpr bool      operator<   ( const Float<T> other ) const;
                                    constexpr bool      operator>=  ( const Float<T> other ) const;
                                    constexpr bool      operator<=  ( const Float<T> other ) const;
                                    constexpr Float<T>  operator-   () const;
                                    constexpr Float<T>  operator+   () const;
                                    constexpr operator  T           () const;
    };

    template<typename T>
    constexpr Float<T>::Float() : value( 0.0 ) {}

    template<typename T>
    constexpr Float<T>::Float( T value ) : value( value ) {}

    template<typename T>
    constexpr void Float<T>::operator=( const Float<T> other )
    {
        this->value = other.value;
    }

    template<typename T>
    template<typename U>
    constexpr void Float<T>::operator=( const Complex<Float<U>> other )
    {
        this->value = other.real();
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator+( const Float<T> other ) const
    {
        return Float<T>( this->value + other.value );
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator-( const Float<T> other ) const
    {
        return Float<T>( this->value - other.value );
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator*( const Float<T> other ) const
    {
        return Float<T>( this->value * other.value );
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator/( const Float<T> other ) const
    {
        return Float<T>( this->value / other.value );
    }

    template<typename T>
    constexpr Float<T>& Float<T>::operator+=( const Float<T> other )
    {
        this->value += other.value;
        return *this;
    }

    template<typename T>
    constexpr Float<T>& Float<T>::operator-=( const Float<T> other )
    {
        this->value -= other.value;
        return *this;
    }

    template<typename T>
    constexpr Float<T>& Float<T>::operator*=( const Float<T> other )
    {
        this->value *= other.value;
        return *this;
    }

    template<typename T>
    constexpr Float<T>& Float<T>::operator/=( const Float<T> other )
    {
        this->value /= other.value;
        return *this;
    }

    template<typename T>
    constexpr bool Float<T>::operator==( const Float<T> other ) const
    {
        return ( this->value == other.value );
    }

    template<typename T>
    constexpr bool Float<T>::operator!=( const Float<T> other ) const
    {
        return ( this->value != other.value );
    }

    template<typename T>
    constexpr bool Float<T>::operator>( const Float<T> other ) const
    {
        return ( this->value > other.value );
    }

    template<typename T>
    constexpr bool Float<T>::operator<( const Float<T> other ) const
    {
        return ( this->value < other.value );
    }

    template<typename T>
    constexpr bool Float<T>::operator>=( const Float<T> other ) const
    {
        return ( this->value >= other.value );
    }

    template<typename T>
    constexpr bool Float<T>::operator<=( const Float<T> other ) const
    {
        return ( this->value <= other.value );
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator-() const
    {
        return Float<T>( -this->value );
    }

    template<typename T>
    constexpr Float<T> Float<T>::operator+() const
    {
        return *this;
    }

    template<typename T>
    constexpr Float<T>::operator T() const
    {
        return this->value;
    }

    template<typename T>
    struct is_float<Float<T>>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT_HPP_
