// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_BOOL_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_BOOL_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef NEURAL_BOOL

namespace nn
{
    class Bool;

    typedef Bool            nbool;
    template<typename T> struct is_nbool;

    class Bool
    {
        private:
            bool value;

        public:
            Bool();
            Bool( bool value );

        public:
            inline                  void        operator=   ( const Bool& other );
            template<typename U>    inline void operator=   ( const Complex<U>& other );
            inline                  void        operator=   ( const bool& other );
            constexpr               Bool        operator+   ( const Bool& other ) const;
            constexpr               Bool        operator+   ( const bool& other ) const;
            constexpr               Bool        operator-   ( const Bool& other ) const;
            constexpr               Bool        operator-   ( const bool& other ) const;
            constexpr               Bool        operator*   ( const Bool& other ) const;
            constexpr               Bool        operator*   ( const bool& other ) const;
            constexpr               Bool        operator/   ( const Bool& other ) const;
            constexpr               Bool        operator/   ( const bool& other ) const;
            inline                  Bool&       operator+=  ( const Bool& other );
            inline                  Bool&       operator+=  ( const bool& other );
            inline                  Bool&       operator-=  ( const Bool& other );
            inline                  Bool&       operator-=  ( const bool& other );
            inline                  Bool&       operator*=  ( const Bool& other );
            inline                  Bool&       operator*=  ( const bool& other );
            inline                  Bool&       operator/=  ( const Bool& other );
            inline                  Bool&       operator/=  ( const bool& other );
            constexpr               bool        operator==  ( const Bool& other ) const;
            constexpr               bool        operator==  ( const bool& other ) const;
            constexpr               bool        operator!=  ( const Bool& other ) const;
            constexpr               bool        operator!=  ( const bool& other ) const;
            constexpr               Bool        operator!   () const;
            constexpr               bool        operator&&  ( const Bool& other ) const;
            constexpr               bool        operator&&  ( const bool& other ) const;
            constexpr               bool        operator||  ( const Bool& other ) const;
            constexpr               bool        operator||  ( const bool& other ) const;
            constexpr               operator    bool        () const;
    };

    Bool::Bool() : value( false ) {}

    Bool::Bool( bool value ) : value( value ) {}

    inline void Bool::operator=( const Bool& other )
    {
        this->value = other.value;
    }

    template<typename U>
    inline void Bool::operator=( const Complex<U>& other )
    {
        this->value = other.real();
    }

    inline void Bool::operator=( const bool& other )
    {
        this->value = other;
    }

    constexpr Bool Bool::operator+( const Bool& other ) const
    {
        return Bool( this->value || other.value );
    }

    constexpr Bool Bool::operator+( const bool& other ) const
    {
        return Bool( this->value || other );
    }

    constexpr Bool Bool::operator-( const Bool& other ) const
    {
        return Bool( this->value || !other.value );
    }

    constexpr Bool Bool::operator-( const bool& other ) const
    {
        return Bool( this->value || !other );
    }

    constexpr Bool Bool::operator*( const Bool& other ) const
    {
        return Bool( this->value && other.value );
    }

    constexpr Bool Bool::operator*( const bool& other ) const
    {
        return Bool( this->value && other );
    }

    constexpr Bool Bool::operator/( const Bool& other ) const
    {
        return Bool( this->value && !other.value );
    }

    constexpr Bool Bool::operator/( const bool& other ) const
    {
        return Bool( this->value && !other );
    }

    inline Bool& Bool::operator+=( const Bool& other )
    {
        this->value = ( this->value || other.value );
        return *this;
    }

    inline Bool& Bool::operator+=( const bool& other )
    {
        this->value = ( this->value || other );
        return *this;
    }

    inline Bool& Bool::operator-=( const Bool& other )
    {
        this->value = ( this->value || !other.value );
        return *this;
    }

    inline Bool& Bool::operator-=( const bool& other )
    {
        this->value = ( this->value || !other );
        return *this;
    }

    inline Bool& Bool::operator*=( const Bool& other )
    {
        this->value = ( this->value && other.value );
        return *this;
    }

    inline Bool& Bool::operator*=( const bool& other )
    {
        this->value = ( this->value && other );
        return *this;
    }

    inline Bool& Bool::operator/=( const Bool& other )
    {
        this->value = ( this->value && !other.value );
        return *this;
    }

    inline Bool& Bool::operator/=( const bool& other )
    {
        this->value = ( this->value && !other );
        return *this;
    }

    constexpr bool Bool::operator==( const Bool& other ) const
    {
        return ( this->value == other.value );
    }

    constexpr bool Bool::operator==( const bool& other ) const
    {
        return ( this->value == other );
    }

    constexpr bool Bool::operator!=( const Bool& other ) const
    {
        return ( this->value != other.value );
    }

    constexpr bool Bool::operator!=( const bool& other ) const
    {
        return ( this->value != other );
    }

    constexpr Bool Bool::operator!() const
    {
        return Bool( !this->value );
    }

    constexpr bool Bool::operator&&( const Bool& other ) const
    {
        return ( this->value && other.value );
    }

    constexpr bool Bool::operator&&( const bool& other ) const
    {
        return ( this->value && other );
    }

    constexpr bool Bool::operator||( const Bool& other ) const
    {
        return ( this->value || other.value );
    }

    constexpr bool Bool::operator||( const bool& other ) const
    {
        return ( this->value || other );
    }

    constexpr Bool::operator bool() const
    {
        return this->value;
    }

    template <typename T>
    constexpr T operator*( const T lhs, const Bool rhs ) requires ( !is_type<T, bool>::value )
    {
        return lhs * rhs;
    }

    template <typename T>
    constexpr T operator/( const T lhs, const Bool rhs ) requires ( !is_type<T, bool>::value )
    {
        return lhs * !rhs;
    }

    template<>
    struct is_bool<nbool>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_BOOL
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_BOOL_HPP_
