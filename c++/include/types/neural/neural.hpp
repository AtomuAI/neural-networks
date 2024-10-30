// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

namespace nn
{
    template <typename T>
    class Neural;

    template<typename T> struct is_neural_int;

    template <typename T>
    class Neural
    {
        private:
            T value;

        public:
            constexpr Neural();
            constexpr Neural( T value );

        public:
            constexpr               void            operator=   ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            template <typename U>   constexpr void  operator=   ( const Complex<Neural<U>> other )  required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator+   ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator-   ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator*   ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator/   ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator+=  ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator-=  ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator*=  ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator/=  ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               bool            operator==  ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               bool            operator!=  ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator!   () const                            required ( is_int<T>::value );
            constexpr               bool            operator||  ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               Neural<T>       operator%   ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               bool            operator&&  ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               Neural<T>&      operator%=  ( const Neural<T> other )           required ( is_int<T>::value );
            constexpr               Neural<T>       operator&   ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               Neural<T>&      operator&=  ( const Neural<T> other )           required ( is_int<T>::value );
            constexpr               Neural<T>       operator|   ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               Neural<T>&      operator|=  ( const Neural<T> other )           required ( is_int<T>::value );
            constexpr               Neural<T>       operator^   ( const Neural<T> other ) const     required ( is_int<T>::value );
            constexpr               Neural<T>&      operator^=  ( const Neural<T> other )           required ( is_int<T>::value );
            constexpr               Neural<T>       operator~   () const                            required ( is_int<T>::value );
            constexpr               Neural<T>       operator<<  ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator<<= ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator>>  ( const Neural<T> other ) const     required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator>>= ( const Neural<T> other )           required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>&      operator++  ()                                  required ( is_int<T>::value );
            constexpr               Neural<T>       operator++  ( int )                             required ( is_int<T>::value );
            constexpr               Neural<T>&      operator--  ()                                  required ( is_int<T>::value );
            constexpr               Neural<T>       operator--  ( int )                             required ( is_int<T>::value );
            constexpr               Neural<T>       operator-   () const                            required ( is_int<T>::value || is_float<T>::value );
            constexpr               Neural<T>       operator+   () const                            required ( is_int<T>::value || is_float<T>::value );
            constexpr               operator        T           () const;
    };

    template<typename T>
    constexpr Neural<T>::Neural() : value( false ) {}

    template<typename T>
    constexpr Neural<T>::Neural( T value ) : value( value ) {}

    template<typename T>
    constexpr void Neural<T>::operator=( const Neural<T> other )
    {
        this->value = other.value;
    }

    template<typename T>
    template<typename U>
    constexpr void Neural<T>::operator=( const Complex<Neural<U>> other )
    {
        this->value = other.real();
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator+( const Neural<T> other ) const
    {
        return Neural<T>( this->value + other.value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator-( const Neural<T> other ) const
    {
        return Neural<T>( this->value - other.value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator*( const Neural<T> other ) const
    {
        return Neural<T>( this->value * other.value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator/( const Neural<T> other ) const
    {
        return Neural<T>( this->value / other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator+=( const Neural<T> other )
    {
        this->value += other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator-=( const Neural<T> other )
    {
        this->value -= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator*=( const Neural<T> other )
    {
        this->value *= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator/=( const Neural<T> other )
    {
        this->value /= other.value;
        return *this;
    }

    template<typename T>
    constexpr bool Neural<T>::operator==( const Neural<T> other ) const
    {
        return ( this->value == other.value );
    }

    template<typename T>
    constexpr bool Neural<T>::operator!=( const Neural<T> other ) const
    {
        return ( this->value != other.value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator!() const
    {
        return Neural<T>( !this->value );
    }

    template<typename T>
    constexpr bool Neural<T>::operator&&( const Neural<T> other ) const
    {
        return ( this->value && other.value );
    }

    template<typename T>
    constexpr bool Neural<T>::operator||( const Neural<T> other ) const
    {
        return ( this->value || other.value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator%( const Neural<T> other ) const
    {
        return Neural<T>( this->value % other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator%=( const Neural<T> other )
    {
        this->value %= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator&( const Neural<T> other ) const
    {
        return Neural<T>( this->value & other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator&=( const Neural<T> other )
    {
        this->value &= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator|( const Neural<T> other ) const
    {
        return Neural<T>( this->value | other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator|=( const Neural<T> other )
    {
        this->value |= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator^( const Neural<T> other ) const
    {
        return Neural<T>( this->value ^ other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator^=( const Neural<T> other )
    {
        this->value ^= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator~() const
    {
        return Neural<T>( ~this->value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator<<( const Neural<T> other ) const
    {
        return Neural<T>( this->value << other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator<<=( const Neural<T> other )
    {
        this->value <<= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator>>( const Neural<T> other ) const
    {
        return Neural<T>( this->value >> other.value );
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator>>=( const Neural<T> other )
    {
        this->value >>= other.value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator++()
    {
        ++this->value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator++( int )
    {
        Neural<T> temp = *this;
        ++this->value;
        return temp;
    }

    template<typename T>
    constexpr Neural<T>& Neural<T>::operator--()
    {
        --this->value;
        return *this;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator--( int )
    {
        Neural<T> temp = *this;
        --this->value;
        return temp;
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator-() const
    {
        return Float<T>( -this->value );
    }

    template<typename T>
    constexpr Neural<T> Neural<T>::operator+() const
    {
        return *this;
    }

    template<typename T>
    constexpr Neural<T>::operator T() const
    {
        return this->value;
    }

    template<typename T>
    struct is_int<Neural<T>>
    {
        static const bool value = true;
    };

    #ifdef INT8
        #ifdef NEURAL_INT8
            typedef Neural<i8> ni8;
            template<typename T>
            struct is_ni8
            {
                static const bool value = false;
            };
            template<>
            struct is_ni8<ni8>
            {
                static const bool value = true;
            };
        #endif // NEURAL_INT8
    #endif // INT8
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT_HPP_
