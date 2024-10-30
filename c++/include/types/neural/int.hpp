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
    class Int;

    template<typename T> struct is_neural_int;

    template <typename T>
    class Int
    {
        private:
            T value;

        public:
            constexpr Int();
            constexpr Int( T value );

        public:
            constexpr               void            operator=   ( const Int<T> other );
            template <typename U>   constexpr void  operator=   ( const Complex<Int<U>> other );
            constexpr               Int<T>          operator+   ( const Int<T> other ) const;
            constexpr               Int<T>          operator-   ( const Int<T> other ) const;
            constexpr               Int<T>          operator*   ( const Int<T> other ) const;
            constexpr               Int<T>          operator/   ( const Int<T> other ) const;
            constexpr               Int<T>&         operator+=  ( const Int<T> other );
            constexpr               Int<T>&         operator-=  ( const Int<T> other );
            constexpr               Int<T>&         operator*=  ( const Int<T> other );
            constexpr               Int<T>&         operator/=  ( const Int<T> other );
            constexpr               bool            operator==  ( const Int<T> other ) const;
            constexpr               bool            operator!=  ( const Int<T> other ) const;
            constexpr               Int<T>          operator!   () const;
            constexpr               bool            operator&&  ( const Int<T> other ) const;
            constexpr               bool            operator||  ( const Int<T> other ) const;
            constexpr               Int<T>          operator%   ( const Int<T> other ) const;
            constexpr               Int<T>&         operator%=  ( const Int<T> other );
            constexpr               Int<T>          operator&   ( const Int<T> other ) const;
            constexpr               Int<T>&         operator&=  ( const Int<T> other );
            constexpr               Int<T>          operator|   ( const Int<T> other ) const;
            constexpr               Int<T>&         operator|=  ( const Int<T> other );
            constexpr               Int<T>          operator^   ( const Int<T> other ) const;
            constexpr               Int<T>&         operator^=  ( const Int<T> other );
            constexpr               Int<T>          operator~   () const;
            constexpr               Int<T>          operator<<  ( const Int<T> other ) const;
            constexpr               Int<T>&         operator<<= ( const Int<T> other );
            constexpr               Int<T>          operator>>  ( const Int<T> other ) const;
            constexpr               Int<T>&         operator>>= ( const Int<T> other );
            constexpr               Int<T>&         operator++  ();
            constexpr               Int<T>          operator++  ( int );
            constexpr               Int<T>&         operator--  ();
            constexpr               Int<T>          operator--  ( int );
            constexpr               Int<T>          operator-   () const;
            constexpr               Int<T>          operator+   () const;
            constexpr               operator        T           () const;
    };

    template<typename T>
    constexpr Int<T>::Int() : value( false ) {}

    template<typename T>
    constexpr Int<T>::Int( T value ) : value( value ) {}

    template<typename T>
    constexpr void Int<T>::operator=( const Int<T> other )
    {
        this->value = other.value;
    }

    template<typename T>
    template<typename U>
    constexpr void Int<T>::operator=( const Complex<Int<U>> other )
    {
        this->value = other.real();
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator+( const Int<T> other ) const
    {
        return Int<T>( this->value + other.value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator-( const Int<T> other ) const
    {
        return Int<T>( this->value - other.value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator*( const Int<T> other ) const
    {
        return Int<T>( this->value * other.value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator/( const Int<T> other ) const
    {
        return Int<T>( this->value / other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator+=( const Int<T> other )
    {
        this->value += other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator-=( const Int<T> other )
    {
        this->value -= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator*=( const Int<T> other )
    {
        this->value *= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator/=( const Int<T> other )
    {
        this->value /= other.value;
        return *this;
    }

    template<typename T>
    constexpr bool Int<T>::operator==( const Int<T> other ) const
    {
        return ( this->value == other.value );
    }

    template<typename T>
    constexpr bool Int<T>::operator!=( const Int<T> other ) const
    {
        return ( this->value != other.value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator!() const
    {
        return Int<T>( !this->value );
    }

    template<typename T>
    constexpr bool Int<T>::operator&&( const Int<T> other ) const
    {
        return ( this->value && other.value );
    }

    template<typename T>
    constexpr bool Int<T>::operator||( const Int<T> other ) const
    {
        return ( this->value || other.value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator%( const Int<T> other ) const
    {
        return Int<T>( this->value % other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator%=( const Int<T> other )
    {
        this->value %= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator&( const Int<T> other ) const
    {
        return Int<T>( this->value & other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator&=( const Int<T> other )
    {
        this->value &= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator|( const Int<T> other ) const
    {
        return Int<T>( this->value | other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator|=( const Int<T> other )
    {
        this->value |= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator^( const Int<T> other ) const
    {
        return Int<T>( this->value ^ other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator^=( const Int<T> other )
    {
        this->value ^= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator~() const
    {
        return Int<T>( ~this->value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator<<( const Int<T> other ) const
    {
        return Int<T>( this->value << other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator<<=( const Int<T> other )
    {
        this->value <<= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator>>( const Int<T> other ) const
    {
        return Int<T>( this->value >> other.value );
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator>>=( const Int<T> other )
    {
        this->value >>= other.value;
        return *this;
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator++()
    {
        ++this->value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator++( int )
    {
        Int<T> temp = *this;
        ++this->value;
        return temp;
    }

    template<typename T>
    constexpr Int<T>& Int<T>::operator--()
    {
        --this->value;
        return *this;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator--( int )
    {
        Int<T> temp = *this;
        --this->value;
        return temp;
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator-() const
    {
        return Float<T>( -this->value );
    }

    template<typename T>
    constexpr Int<T> Int<T>::operator+() const
    {
        return *this;
    }

    template<typename T>
    constexpr Int<T>::operator T() const
    {
        return this->value;
    }

    template<typename T>
    struct is_int<Int<T>>
    {
        static const bool value = true;
    };

    #ifdef INT8
        #ifdef NEURAL_INT8
            typedef Int<i8> ni8;
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
