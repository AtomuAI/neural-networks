// The template and inlines for the -*- C++ -*- complex number classes.

// Copyright (C) 1997-2023 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file include/complex
 *  This is a Standard C++ Library header.
 */

//
// ISO C++ 14882: 26.2  Complex Numbers
// Note: this is not a conforming implementation.
// Initially implemented by Ulrich Drepper <drepper@cygnus.com>
// Improved by Gabriel Dos Reis <dosreis@cmla.ens-cachan.fr>
//

#define COMPLEX_INT8
#define COMPLEX_INT16
#define COMPLEX_INT32
#define COMPLEX_INT64
#define COMPLEX_INT128
#define COMPLEX_UINT8
#define COMPLEX_UINT16
#define COMPLEX_UINT32
#define COMPLEX_UINT64
#define COMPLEX_UINT128
#define COMPLEX_FLOAT16
#define COMPLEX_FLOAT32
#define COMPLEX_FLOAT64
#define COMPLEX_FLOAT128

#ifndef _GLIBCXX_COMPLEX
#define _GLIBCXX_COMPLEX 1

#pragma GCC system_header

#include <bits/c++config.h>
#include <bits/cpp_type_traits.h>
#include <ext/type_traits.h>
#include <cmath>
#include <sstream>

// Get rid of a macro possibly defined in <complex.h>
#undef complex

#if __cplusplus > 201703L
# define __cpp_lib_constexpr_complex 201711L
#endif

//: Types Header
#include "bewusstsein_neural_networks/c++/include/core/types.hpp"

namespace nn
{

  /**
   * @defgroup complex_numbers Complex Numbers
   * @ingroup numerics
   *
   * Classes and functions for complex numbers.
   * @{
   */

  // Forward declarations.

  template<typename _Tp> class complex;
    #ifdef COMPLEX_INT8
        template<> class complex<i8>;
    #endif
    #ifdef COMPLEX_INT16
        template<> class complex<i16>;
    #endif
    #ifdef COMPLEX_INT32
        template<> class complex<i32>;
    #endif
    #ifdef COMPLEX_INT64
        template<> class complex<i64>;
    #endif
    #ifdef COMPLEX_INT128
        template<> class complex<i128>;
    #endif
    #ifdef COMPLEX_UINT8
        template<> class complex<u8>;
    #endif
    #ifdef COMPLEX_UINT16
        template<> class complex<u16>;
    #endif
    #ifdef COMPLEX_UINT32
        template<> class complex<u32>;
    #endif
    #ifdef COMPLEX_UINT64
        template<> class complex<u64>;
    #endif
    #ifdef COMPLEX_UINT128
        template<> class complex<u128>;
    #endif
    #ifdef COMPLEX_FLOAT16
        template<> class complex<f16>;
    #endif
    #ifdef COMPLEX_FLOAT32
        template<> class complex<f32>;
    #endif
    #ifdef COMPLEX_FLOAT64
        template<> class complex<f64>;
    #endif
    #ifdef COMPLEX_FLOAT128
        template<> class complex<f128>;
    #endif
  template<> class complex<float>;
  template<> class complex<double>;
  template<> class complex<long double>;

  ///  Return magnitude of @a z.
  template<typename _Tp> _Tp abs(const complex<_Tp>&);
  ///  Return phase angle of @a z.
  template<typename _Tp> _Tp arg(const complex<_Tp>&);
  ///  Return @a z magnitude squared.
  template<typename _Tp> _Tp _GLIBCXX20_CONSTEXPR norm(const complex<_Tp>&);

  ///  Return complex conjugate of @a z.
  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR complex<_Tp> conj(const complex<_Tp>&);
  ///  Return complex with magnitude @a rho and angle @a theta.
  template<typename _Tp> complex<_Tp> polar(const _Tp&, const _Tp& = 0);

  // Transcendentals:
  /// Return complex cosine of @a z.
  template<typename _Tp> complex<_Tp> cos(const complex<_Tp>&);
  /// Return complex hyperbolic cosine of @a z.
  template<typename _Tp> complex<_Tp> cosh(const complex<_Tp>&);
  /// Return complex base e exponential of @a z.
  template<typename _Tp> complex<_Tp> exp(const complex<_Tp>&);
  /// Return complex natural logarithm of @a z.
  template<typename _Tp> complex<_Tp> log(const complex<_Tp>&);
  /// Return complex base 10 logarithm of @a z.
  template<typename _Tp> complex<_Tp> log10(const complex<_Tp>&);
  /// Return @a x to the @a y'th power.
  template<typename _Tp> complex<_Tp> pow(const complex<_Tp>&, int);
  /// Return @a x to the @a y'th power.
  template<typename _Tp> complex<_Tp> pow(const complex<_Tp>&, const _Tp&);
  /// Return @a x to the @a y'th power.
  template<typename _Tp> complex<_Tp> pow(const complex<_Tp>&,
                                          const complex<_Tp>&);
  /// Return @a x to the @a y'th power.
  template<typename _Tp> complex<_Tp> pow(const _Tp&, const complex<_Tp>&);
  /// Return complex sine of @a z.
  template<typename _Tp> complex<_Tp> sin(const complex<_Tp>&);
  /// Return complex hyperbolic sine of @a z.
  template<typename _Tp> complex<_Tp> sinh(const complex<_Tp>&);
  /// Return complex square root of @a z.
  template<typename _Tp> complex<_Tp> sqrt(const complex<_Tp>&);
  /// Return complex tangent of @a z.
  template<typename _Tp> complex<_Tp> tan(const complex<_Tp>&);
  /// Return complex hyperbolic tangent of @a z.
  template<typename _Tp> complex<_Tp> tanh(const complex<_Tp>&);


  // 26.2.2  Primary template class complex
  /**
   *  Template to represent complex numbers.
   *
   *  Specializations for float, double, and long double are part of the
   *  library.  Results with any other type are not guaranteed.
   *
   *  @param  Tp  Type of real and imaginary values.
  */
  template<typename _Tp>
    class complex
    {
    public:
      /// Value typedef.
      typedef _Tp value_type;

      ///  Default constructor.  First parameter is x, second parameter is y.
      ///  Unspecified parameters default to 0.
      _GLIBCXX_CONSTEXPR complex(const _Tp& __r = _Tp(), const _Tp& __i = _Tp())
      : _M_real(__r), _M_imag(__i) { }

      // Let the compiler synthesize the copy constructor
#if __cplusplus >= 201103L
      constexpr complex(const complex&) = default;
#endif

      ///  Converting constructor.
      template<typename _Up>
#if __cplusplus > 202002L
	explicit(!requires(_Up __u) { _Tp{__u}; })
#endif
	_GLIBCXX_CONSTEXPR complex(const complex<_Up>& __z)
	: _M_real(_Tp(__z.real())), _M_imag(_Tp(__z.imag())) { }

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      _GLIBCXX_ABI_TAG_CXX11
      constexpr _Tp
      real() const { return _M_real; }

      _GLIBCXX_ABI_TAG_CXX11
      constexpr _Tp
      imag() const { return _M_imag; }
#else
      ///  Return real part of complex number.
      _Tp&
      real() { return _M_real; }

      ///  Return real part of complex number.
      const _Tp&
      real() const { return _M_real; }

      ///  Return imaginary part of complex number.
      _Tp&
      imag() { return _M_imag; }

      ///  Return imaginary part of complex number.
      const _Tp&
      imag() const { return _M_imag; }
#endif

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      _GLIBCXX20_CONSTEXPR void
      real(_Tp __val) { _M_real = __val; }

      _GLIBCXX20_CONSTEXPR void
      imag(_Tp __val) { _M_imag = __val; }

      /// Assign a scalar to this complex number.
      _GLIBCXX20_CONSTEXPR complex<_Tp>& operator=(const _Tp&);

      /// Add a scalar to this complex number.
      // 26.2.5/1
      _GLIBCXX20_CONSTEXPR complex<_Tp>&
      operator+=(const _Tp& __t)
      {
	_M_real += __t;
	return *this;
      }

      /// Subtract a scalar from this complex number.
      // 26.2.5/3
      _GLIBCXX20_CONSTEXPR complex<_Tp>&
      operator-=(const _Tp& __t)
      {
	_M_real -= __t;
	return *this;
      }

      /// Multiply this complex number by a scalar.
      _GLIBCXX20_CONSTEXPR complex<_Tp>& operator*=(const _Tp&);
      /// Divide this complex number by a scalar.
      _GLIBCXX20_CONSTEXPR complex<_Tp>& operator/=(const _Tp&);

      // Let the compiler synthesize the copy assignment operator
#if __cplusplus >= 201103L
      _GLIBCXX20_CONSTEXPR complex& operator=(const complex&) = default;
#endif

      /// Assign another complex number to this one.
      template<typename _Up>
        _GLIBCXX20_CONSTEXPR complex<_Tp>& operator=(const complex<_Up>&);
      /// Add another complex number to this one.
      template<typename _Up>
        _GLIBCXX20_CONSTEXPR complex<_Tp>& operator+=(const complex<_Up>&);
      /// Subtract another complex number from this one.
      template<typename _Up>
        _GLIBCXX20_CONSTEXPR complex<_Tp>& operator-=(const complex<_Up>&);
      /// Multiply this complex number by another.
      template<typename _Up>
        _GLIBCXX20_CONSTEXPR complex<_Tp>& operator*=(const complex<_Up>&);
      /// Divide this complex number by another.
      template<typename _Up>
        _GLIBCXX20_CONSTEXPR complex<_Tp>& operator/=(const complex<_Up>&);

      _GLIBCXX_CONSTEXPR complex __rep() const
      { return *this; }

    private:
      _Tp _M_real;
      _Tp _M_imag;
    };

  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator=(const _Tp& __t)
    {
     _M_real = __t;
     _M_imag = _Tp();
     return *this;
    }

  // 26.2.5/5
  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator*=(const _Tp& __t)
    {
      _M_real *= __t;
      _M_imag *= __t;
      return *this;
    }

  // 26.2.5/7
  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator/=(const _Tp& __t)
    {
      _M_real /= __t;
      _M_imag /= __t;
      return *this;
    }

  template<typename _Tp>
    template<typename _Up>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator=(const complex<_Up>& __z)
    {
      _M_real = __z.real();
      _M_imag = __z.imag();
      return *this;
    }

  // 26.2.5/9
  template<typename _Tp>
    template<typename _Up>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator+=(const complex<_Up>& __z)
    {
      _M_real += __z.real();
      _M_imag += __z.imag();
      return *this;
    }

  // 26.2.5/11
  template<typename _Tp>
    template<typename _Up>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator-=(const complex<_Up>& __z)
    {
      _M_real -= __z.real();
      _M_imag -= __z.imag();
      return *this;
    }

  // 26.2.5/13
  // XXX: This is a grammar school implementation.
  template<typename _Tp>
    template<typename _Up>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator*=(const complex<_Up>& __z)
    {
      const _Tp __r = _M_real * __z.real() - _M_imag * __z.imag();
      _M_imag = _M_real * __z.imag() + _M_imag * __z.real();
      _M_real = __r;
      return *this;
    }

  // 26.2.5/15
  // XXX: This is a grammar school implementation.
  template<typename _Tp>
    template<typename _Up>
    _GLIBCXX20_CONSTEXPR complex<_Tp>&
    complex<_Tp>::operator/=(const complex<_Up>& __z)
    {
      const _Tp __r =  _M_real * __z.real() + _M_imag * __z.imag();
      const _Tp __n = std::norm(__z);
      _M_imag = (_M_imag * __z.real() - _M_real * __z.imag()) / __n;
      _M_real = __r / __n;
      return *this;
    }

  // Operators:
  ///@{
  ///  Return new complex value @a x plus @a y.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __x;
      __r += __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator+(const complex<_Tp>& __x, const _Tp& __y)
    {
      complex<_Tp> __r = __x;
      __r += __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator+(const _Tp& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __y;
      __r += __x;
      return __r;
    }
  ///@}

  ///@{
  ///  Return new complex value @a x minus @a y.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __x;
      __r -= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator-(const complex<_Tp>& __x, const _Tp& __y)
    {
      complex<_Tp> __r = __x;
      __r -= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator-(const _Tp& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = -__y;
      __r += __x;
      return __r;
    }
  ///@}

  ///@{
  ///  Return new complex value @a x times @a y.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator*(const complex<_Tp>& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __x;
      __r *= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator*(const complex<_Tp>& __x, const _Tp& __y)
    {
      complex<_Tp> __r = __x;
      __r *= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator*(const _Tp& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __y;
      __r *= __x;
      return __r;
    }
  ///@}

  ///@{
  ///  Return new complex value @a x divided by @a y.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator/(const complex<_Tp>& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __x;
      __r /= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator/(const complex<_Tp>& __x, const _Tp& __y)
    {
      complex<_Tp> __r = __x;
      __r /= __y;
      return __r;
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator/(const _Tp& __x, const complex<_Tp>& __y)
    {
      complex<_Tp> __r = __x;
      __r /= __y;
      return __r;
    }
  ///@}

  ///  Return @a x.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator+(const complex<_Tp>& __x)
    { return __x; }

  ///  Return complex negation of @a x.
  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    operator-(const complex<_Tp>& __x)
    { return complex<_Tp>(-__x.real(), -__x.imag()); }

  ///@{
  ///  Return true if @a x is equal to @a y.
  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
    { return __x.real() == __y.real() && __x.imag() == __y.imag(); }

  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator==(const complex<_Tp>& __x, const _Tp& __y)
    { return __x.real() == __y && __x.imag() == _Tp(); }

#if !(__cpp_impl_three_way_comparison >= 201907L)
  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator==(const _Tp& __x, const complex<_Tp>& __y)
    { return __x == __y.real() && _Tp() == __y.imag(); }
  ///@}

  ///@{
  ///  Return false if @a x is equal to @a y.
  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
    { return __x.real() != __y.real() || __x.imag() != __y.imag(); }

  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator!=(const complex<_Tp>& __x, const _Tp& __y)
    { return __x.real() != __y || __x.imag() != _Tp(); }

  template<typename _Tp>
    inline _GLIBCXX_CONSTEXPR bool
    operator!=(const _Tp& __x, const complex<_Tp>& __y)
    { return __x != __y.real() || _Tp() != __y.imag(); }
#endif
  ///@}

  ///  Extraction operator for complex values.
  template<typename _Tp, typename _CharT, class _Traits>
    basic_istream<_CharT, _Traits>&
    operator>>(basic_istream<_CharT, _Traits>& __is, complex<_Tp>& __x)
    {
      bool __fail = true;
      _CharT __ch;
      if (__is >> __ch)
	{
	  if (_Traits::eq(__ch, __is.widen('(')))
	    {
	      _Tp __u;
	      if (__is >> __u >> __ch)
		{
		  const _CharT __rparen = __is.widen(')');
		  if (_Traits::eq(__ch, __rparen))
		    {
		      __x = __u;
		      __fail = false;
		    }
		  else if (_Traits::eq(__ch, __is.widen(',')))
		    {
		      _Tp __v;
		      if (__is >> __v >> __ch)
			{
			  if (_Traits::eq(__ch, __rparen))
			    {
			      __x = complex<_Tp>(__u, __v);
			      __fail = false;
			    }
			  else
			    __is.putback(__ch);
			}
		    }
		  else
		    __is.putback(__ch);
		}
	    }
	  else
	    {
	      __is.putback(__ch);
	      _Tp __u;
	      if (__is >> __u)
		{
		  __x = __u;
		  __fail = false;
		}
	    }
	}
      if (__fail)
	__is.setstate(ios_base::failbit);
      return __is;
    }

  ///  Insertion operator for complex values.
  template<typename _Tp, typename _CharT, class _Traits>
    basic_ostream<_CharT, _Traits>&
    operator<<(basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
    {
      basic_ostringstream<_CharT, _Traits> __s;
      __s.flags(__os.flags());
      __s.imbue(__os.getloc());
      __s.precision(__os.precision());
      __s << '(' << __x.real() << ',' << __x.imag() << ')';
      return __os << __s.str();
    }

  // Values
#if __cplusplus >= 201103L
  template<typename _Tp>
    constexpr _Tp
    real(const complex<_Tp>& __z)
    { return __z.real(); }

  template<typename _Tp>
    constexpr _Tp
    imag(const complex<_Tp>& __z)
    { return __z.imag(); }
#else
  template<typename _Tp>
    inline _Tp&
    real(complex<_Tp>& __z)
    { return __z.real(); }

  template<typename _Tp>
    inline const _Tp&
    real(const complex<_Tp>& __z)
    { return __z.real(); }

  template<typename _Tp>
    inline _Tp&
    imag(complex<_Tp>& __z)
    { return __z.imag(); }

  template<typename _Tp>
    inline const _Tp&
    imag(const complex<_Tp>& __z)
    { return __z.imag(); }
#endif

#if _GLIBCXX_USE_C99_COMPLEX
#if defined(__STDCPP_FLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline _Float16
  __complex_abs(__complex__ _Float16 __z)
  { return _Float16(__builtin_cabsf(__z)); }

  inline _Float16
  __complex_arg(__complex__ _Float16 __z)
  { return _Float16(__builtin_cargf(__z)); }

  inline __complex__ _Float16
  __complex_cos(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_ccosf(__z)); }

  inline __complex__ _Float16
  __complex_cosh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_ccoshf(__z)); }

  inline __complex__ _Float16
  __complex_exp(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_cexpf(__z)); }

  inline __complex__ _Float16
  __complex_log(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_clogf(__z)); }

  inline __complex__ _Float16
  __complex_sin(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_csinf(__z)); }

  inline __complex__ _Float16
  __complex_sinh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_csinhf(__z)); }

  inline __complex__ _Float16
  __complex_sqrt(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_csqrtf(__z)); }

  inline __complex__ _Float16
  __complex_tan(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_ctanf(__z)); }

  inline __complex__ _Float16
  __complex_tanh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_ctanhf(__z)); }

  inline __complex__ _Float16
  __complex_pow(__complex__ _Float16 __x, __complex__ _Float16 __y)
  { return static_cast<__complex__ _Float16>(__builtin_cpowf(__x, __y)); }
#endif

#if defined(__STDCPP_FLOAT32_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline _Float32
  __complex_abs(__complex__ _Float32 __z) { return __builtin_cabsf(__z); }

  inline _Float32
  __complex_arg(__complex__ _Float32 __z) { return __builtin_cargf(__z); }

  inline __complex__ _Float32
  __complex_cos(__complex__ _Float32 __z) { return __builtin_ccosf(__z); }

  inline __complex__ _Float32
  __complex_cosh(__complex__ _Float32 __z) { return __builtin_ccoshf(__z); }

  inline __complex__ _Float32
  __complex_exp(__complex__ _Float32 __z) { return __builtin_cexpf(__z); }

  inline __complex__ _Float32
  __complex_log(__complex__ _Float32 __z) { return __builtin_clogf(__z); }

  inline __complex__ _Float32
  __complex_sin(__complex__ _Float32 __z) { return __builtin_csinf(__z); }

  inline __complex__ _Float32
  __complex_sinh(__complex__ _Float32 __z) { return __builtin_csinhf(__z); }

  inline __complex__ _Float32
  __complex_sqrt(__complex__ _Float32 __z) { return __builtin_csqrtf(__z); }

  inline __complex__ _Float32
  __complex_tan(__complex__ _Float32 __z) { return __builtin_ctanf(__z); }

  inline __complex__ _Float32
  __complex_tanh(__complex__ _Float32 __z) { return __builtin_ctanhf(__z); }

  inline __complex__ _Float32
  __complex_pow(__complex__ _Float32 __x, __complex__ _Float32 __y)
  { return __builtin_cpowf(__x, __y); }
#endif

#if defined(__STDCPP_FLOAT64_T__) && defined(_GLIBCXX_DOUBLE_IS_IEEE_BINARY64)
  inline _Float64
  __complex_abs(__complex__ _Float64 __z) { return __builtin_cabs(__z); }

  inline _Float64
  __complex_arg(__complex__ _Float64 __z) { return __builtin_carg(__z); }

  inline __complex__ _Float64
  __complex_cos(__complex__ _Float64 __z) { return __builtin_ccos(__z); }

  inline __complex__ _Float64
  __complex_cosh(__complex__ _Float64 __z) { return __builtin_ccosh(__z); }

  inline __complex__ _Float64
  __complex_exp(__complex__ _Float64 __z) { return __builtin_cexp(__z); }

  inline __complex__ _Float64
  __complex_log(__complex__ _Float64 __z) { return __builtin_clog(__z); }

  inline __complex__ _Float64
  __complex_sin(__complex__ _Float64 __z) { return __builtin_csin(__z); }

  inline __complex__ _Float64
  __complex_sinh(__complex__ _Float64 __z) { return __builtin_csinh(__z); }

  inline __complex__ _Float64
  __complex_sqrt(__complex__ _Float64 __z) { return __builtin_csqrt(__z); }

  inline __complex__ _Float64
  __complex_tan(__complex__ _Float64 __z) { return __builtin_ctan(__z); }

  inline __complex__ _Float64
  __complex_tanh(__complex__ _Float64 __z) { return __builtin_ctanh(__z); }

  inline __complex__ _Float64
  __complex_pow(__complex__ _Float64 __x, __complex__ _Float64 __y)
  { return __builtin_cpow(__x, __y); }
#endif

#if defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_LDOUBLE_IS_IEEE_BINARY128)
  inline _Float128
  __complex_abs(__complex__ _Float128 __z) { return __builtin_cabsl(__z); }

  inline _Float128
  __complex_arg(__complex__ _Float128 __z) { return __builtin_cargl(__z); }

  inline __complex__ _Float128
  __complex_cos(__complex__ _Float128 __z) { return __builtin_ccosl(__z); }

  inline __complex__ _Float128
  __complex_cosh(__complex__ _Float128 __z) { return __builtin_ccoshl(__z); }

  inline __complex__ _Float128
  __complex_exp(__complex__ _Float128 __z) { return __builtin_cexpl(__z); }

  inline __complex__ _Float128
  __complex_log(__complex__ _Float128 __z) { return __builtin_clogl(__z); }

  inline __complex__ _Float128
  __complex_sin(__complex__ _Float128 __z) { return __builtin_csinl(__z); }

  inline __complex__ _Float128
  __complex_sinh(__complex__ _Float128 __z) { return __builtin_csinhl(__z); }

  inline __complex__ _Float128
  __complex_sqrt(__complex__ _Float128 __z) { return __builtin_csqrtl(__z); }

  inline __complex__ _Float128
  __complex_tan(__complex__ _Float128 __z) { return __builtin_ctanl(__z); }

  inline __complex__ _Float128
  __complex_tanh(__complex__ _Float128 __z) { return __builtin_ctanhl(__z); }

  inline __complex__ _Float128
  __complex_pow(__complex__ _Float128 __x, __complex__ _Float128 __y)
  { return __builtin_cpowl(__x, __y); }
#elif defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_HAVE_FLOAT128_MATH)
  inline _Float128
  __complex_abs(__complex__ _Float128 __z) { return __builtin_cabsf128(__z); }

  inline _Float128
  __complex_arg(__complex__ _Float128 __z) { return __builtin_cargf128(__z); }

  inline __complex__ _Float128
  __complex_cos(__complex__ _Float128 __z) { return __builtin_ccosf128(__z); }

  inline __complex__ _Float128
  __complex_cosh(__complex__ _Float128 __z) { return __builtin_ccoshf128(__z); }

  inline __complex__ _Float128
  __complex_exp(__complex__ _Float128 __z) { return __builtin_cexpf128(__z); }

  inline __complex__ _Float128
  __complex_log(__complex__ _Float128 __z) { return __builtin_clogf128(__z); }

  inline __complex__ _Float128
  __complex_sin(__complex__ _Float128 __z) { return __builtin_csinf128(__z); }

  inline __complex__ _Float128
  __complex_sinh(__complex__ _Float128 __z) { return __builtin_csinhf128(__z); }

  inline __complex__ _Float128
  __complex_sqrt(__complex__ _Float128 __z) { return __builtin_csqrtf128(__z); }

  inline __complex__ _Float128
  __complex_tan(__complex__ _Float128 __z) { return __builtin_ctanf128(__z); }

  inline __complex__ _Float128
  __complex_tanh(__complex__ _Float128 __z) { return __builtin_ctanhf128(__z); }

  inline __complex__ _Float128
  __complex_pow(__complex__ _Float128 __x, __complex__ _Float128 __y)
  { return __builtin_cpowf128(__x, __y); }
#endif

#if defined(__STDCPP_BFLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __gnu_cxx::__bfloat16_t
  __complex_abs(__complex__ decltype(0.0bf16) __z)
  { return __gnu_cxx::__bfloat16_t(__builtin_cabsf(__z)); }

  inline __gnu_cxx::__bfloat16_t
  __complex_arg(__complex__ decltype(0.0bf16) __z)
  { return __gnu_cxx::__bfloat16_t(__builtin_cargf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_cos(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_ccosf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_cosh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_ccoshf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_exp(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_cexpf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_log(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_clogf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_sin(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_csinf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_sinh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_csinhf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_sqrt(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_csqrtf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_tan(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_ctanf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_tanh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_ctanhf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_pow(__complex__ decltype(0.0bf16) __x,
		__complex__ decltype(0.0bf16) __y)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_cpowf(__x,
								      __y)); }
#endif
#endif

  // 26.2.7/3 abs(__z):  Returns the magnitude of __z.
  template<typename _Tp>
    inline _Tp
    __complex_abs(const complex<_Tp>& __z)
    {
      _Tp __x = __z.real();
      _Tp __y = __z.imag();
      const _Tp __s = std::max(abs(__x), abs(__y));
      if (__s == _Tp())  // well ...
        return __s;
      __x /= __s;
      __y /= __s;
      return __s * sqrt(__x * __x + __y * __y);
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline float
  __complex_abs(__complex__ float __z) { return __builtin_cabsf(__z); }

  inline double
  __complex_abs(__complex__ double __z) { return __builtin_cabs(__z); }

  inline long double
  __complex_abs(const __complex__ long double& __z)
  { return __builtin_cabsl(__z); }

  template<typename _Tp>
    inline _Tp
    abs(const complex<_Tp>& __z) { return __complex_abs(__z.__rep()); }
#else
  template<typename _Tp>
    inline _Tp
    abs(const complex<_Tp>& __z) { return __complex_abs(__z); }
#endif


  // 26.2.7/4: arg(__z): Returns the phase angle of __z.
  template<typename _Tp>
    inline _Tp
    __complex_arg(const complex<_Tp>& __z)
    { return  atan2(__z.imag(), __z.real()); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline float
  __complex_arg(__complex__ float __z) { return __builtin_cargf(__z); }

  inline double
  __complex_arg(__complex__ double __z) { return __builtin_carg(__z); }

  inline long double
  __complex_arg(const __complex__ long double& __z)
  { return __builtin_cargl(__z); }

  template<typename _Tp>
    inline _Tp
    arg(const complex<_Tp>& __z) { return __complex_arg(__z.__rep()); }
#else
  template<typename _Tp>
    inline _Tp
    arg(const complex<_Tp>& __z) { return __complex_arg(__z); }
#endif

  // 26.2.7/5: norm(__z) returns the squared magnitude of __z.
  //     As defined, norm() is -not- a norm is the common mathematical
  //     sense used in numerics.  The helper class _Norm_helper<> tries to
  //     distinguish between builtin floating point and the rest, so as
  //     to deliver an answer as close as possible to the real value.
  template<bool>
    struct _Norm_helper
    {
      template<typename _Tp>
        static inline _GLIBCXX20_CONSTEXPR _Tp _S_do_it(const complex<_Tp>& __z)
        {
          const _Tp __x = __z.real();
          const _Tp __y = __z.imag();
          return __x * __x + __y * __y;
        }
    };

  template<>
    struct _Norm_helper<true>
    {
      template<typename _Tp>
        static inline _GLIBCXX20_CONSTEXPR _Tp _S_do_it(const complex<_Tp>& __z)
        {
          //_Tp __res = std::abs(__z);
          //return __res * __res;
          const _Tp __x = __z.real();
          const _Tp __y = __z.imag();
          return __x * __x + __y * __y;
        }
    };

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR _Tp
    norm(const complex<_Tp>& __z)
    {
      return _Norm_helper<__is_floating<_Tp>::__value
	&& !_GLIBCXX_FAST_MATH>::_S_do_it(__z);
    }

  template<typename _Tp>
    inline complex<_Tp>
    polar(const _Tp& __rho, const _Tp& __theta)
    {
      __glibcxx_assert( __rho >= 0 );
      return complex<_Tp>(__rho * cos(__theta), __rho * sin(__theta));
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR complex<_Tp>
    conj(const complex<_Tp>& __z)
    { return complex<_Tp>(__z.real(), -__z.imag()); }

  // Transcendentals

  // 26.2.8/1 cos(__z):  Returns the cosine of __z.
  template<typename _Tp>
    inline complex<_Tp>
    __complex_cos(const complex<_Tp>& __z)
    {
      const _Tp __x = __z.real();
      const _Tp __y = __z.imag();
      return complex<_Tp>(cos(__x) * cosh(__y), -sin(__x) * sinh(__y));
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_cos(__complex__ float __z) { return __builtin_ccosf(__z); }

  inline __complex__ double
  __complex_cos(__complex__ double __z) { return __builtin_ccos(__z); }

  inline __complex__ long double
  __complex_cos(const __complex__ long double& __z)
  { return __builtin_ccosl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    cos(const complex<_Tp>& __z) { return __complex_cos(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    cos(const complex<_Tp>& __z) { return __complex_cos(__z); }
#endif

  // 26.2.8/2 cosh(__z): Returns the hyperbolic cosine of __z.
  template<typename _Tp>
    inline complex<_Tp>
    __complex_cosh(const complex<_Tp>& __z)
    {
      const _Tp __x = __z.real();
      const _Tp __y = __z.imag();
      return complex<_Tp>(cosh(__x) * cos(__y), sinh(__x) * sin(__y));
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_cosh(__complex__ float __z) { return __builtin_ccoshf(__z); }

  inline __complex__ double
  __complex_cosh(__complex__ double __z) { return __builtin_ccosh(__z); }

  inline __complex__ long double
  __complex_cosh(const __complex__ long double& __z)
  { return __builtin_ccoshl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    cosh(const complex<_Tp>& __z) { return __complex_cosh(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    cosh(const complex<_Tp>& __z) { return __complex_cosh(__z); }
#endif

  // 26.2.8/3 exp(__z): Returns the complex base e exponential of x
  template<typename _Tp>
    inline complex<_Tp>
    __complex_exp(const complex<_Tp>& __z)
    { return std::polar<_Tp>(exp(__z.real()), __z.imag()); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_exp(__complex__ float __z) { return __builtin_cexpf(__z); }

  inline __complex__ double
  __complex_exp(__complex__ double __z) { return __builtin_cexp(__z); }

  inline __complex__ long double
  __complex_exp(const __complex__ long double& __z)
  { return __builtin_cexpl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    exp(const complex<_Tp>& __z) { return __complex_exp(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    exp(const complex<_Tp>& __z) { return __complex_exp(__z); }
#endif

  // 26.2.8/5 log(__z): Returns the natural complex logarithm of __z.
  //                    The branch cut is along the negative axis.
  template<typename _Tp>
    inline complex<_Tp>
    __complex_log(const complex<_Tp>& __z)
    { return complex<_Tp>(log(std::abs(__z)), std::arg(__z)); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_log(__complex__ float __z) { return __builtin_clogf(__z); }

  inline __complex__ double
  __complex_log(__complex__ double __z) { return __builtin_clog(__z); }

  inline __complex__ long double
  __complex_log(const __complex__ long double& __z)
  { return __builtin_clogl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    log(const complex<_Tp>& __z) { return __complex_log(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    log(const complex<_Tp>& __z) { return __complex_log(__z); }
#endif

  template<typename _Tp>
    inline complex<_Tp>
    log10(const complex<_Tp>& __z)
    { return std::log(__z) / log(_Tp(10.0)); }

  // 26.2.8/10 sin(__z): Returns the sine of __z.
  template<typename _Tp>
    inline complex<_Tp>
    __complex_sin(const complex<_Tp>& __z)
    {
      const _Tp __x = __z.real();
      const _Tp __y = __z.imag();
      return complex<_Tp>(sin(__x) * cosh(__y), cos(__x) * sinh(__y));
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_sin(__complex__ float __z) { return __builtin_csinf(__z); }

  inline __complex__ double
  __complex_sin(__complex__ double __z) { return __builtin_csin(__z); }

  inline __complex__ long double
  __complex_sin(const __complex__ long double& __z)
  { return __builtin_csinl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    sin(const complex<_Tp>& __z) { return __complex_sin(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    sin(const complex<_Tp>& __z) { return __complex_sin(__z); }
#endif

  // 26.2.8/11 sinh(__z): Returns the hyperbolic sine of __z.
  template<typename _Tp>
    inline complex<_Tp>
    __complex_sinh(const complex<_Tp>& __z)
    {
      const _Tp __x = __z.real();
      const _Tp  __y = __z.imag();
      return complex<_Tp>(sinh(__x) * cos(__y), cosh(__x) * sin(__y));
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_sinh(__complex__ float __z) { return __builtin_csinhf(__z); }

  inline __complex__ double
  __complex_sinh(__complex__ double __z) { return __builtin_csinh(__z); }

  inline __complex__ long double
  __complex_sinh(const __complex__ long double& __z)
  { return __builtin_csinhl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    sinh(const complex<_Tp>& __z) { return __complex_sinh(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    sinh(const complex<_Tp>& __z) { return __complex_sinh(__z); }
#endif

  // 26.2.8/13 sqrt(__z): Returns the complex square root of __z.
  //                     The branch cut is on the negative axis.
  template<typename _Tp>
    complex<_Tp>
    __complex_sqrt(const complex<_Tp>& __z)
    {
      _Tp __x = __z.real();
      _Tp __y = __z.imag();

      if (__x == _Tp())
        {
          _Tp __t = sqrt(abs(__y) / 2);
          return complex<_Tp>(__t, __y < _Tp() ? -__t : __t);
        }
      else
        {
          _Tp __t = sqrt(2 * (std::abs(__z) + abs(__x)));
          _Tp __u = __t / 2;
          return __x > _Tp()
            ? complex<_Tp>(__u, __y / __t)
            : complex<_Tp>(abs(__y) / __t, __y < _Tp() ? -__u : __u);
        }
    }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_sqrt(__complex__ float __z) { return __builtin_csqrtf(__z); }

  inline __complex__ double
  __complex_sqrt(__complex__ double __z) { return __builtin_csqrt(__z); }

  inline __complex__ long double
  __complex_sqrt(const __complex__ long double& __z)
  { return __builtin_csqrtl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    sqrt(const complex<_Tp>& __z) { return __complex_sqrt(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    sqrt(const complex<_Tp>& __z) { return __complex_sqrt(__z); }
#endif

  // 26.2.8/14 tan(__z):  Return the complex tangent of __z.

  template<typename _Tp>
    inline complex<_Tp>
    __complex_tan(const complex<_Tp>& __z)
    { return std::sin(__z) / std::cos(__z); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_tan(__complex__ float __z) { return __builtin_ctanf(__z); }

  inline __complex__ double
  __complex_tan(__complex__ double __z) { return __builtin_ctan(__z); }

  inline __complex__ long double
  __complex_tan(const __complex__ long double& __z)
  { return __builtin_ctanl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    tan(const complex<_Tp>& __z) { return __complex_tan(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    tan(const complex<_Tp>& __z) { return __complex_tan(__z); }
#endif


  // 26.2.8/15 tanh(__z):  Returns the hyperbolic tangent of __z.

  template<typename _Tp>
    inline complex<_Tp>
    __complex_tanh(const complex<_Tp>& __z)
    { return std::sinh(__z) / std::cosh(__z); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_tanh(__complex__ float __z) { return __builtin_ctanhf(__z); }

  inline __complex__ double
  __complex_tanh(__complex__ double __z) { return __builtin_ctanh(__z); }

  inline __complex__ long double
  __complex_tanh(const __complex__ long double& __z)
  { return __builtin_ctanhl(__z); }

  template<typename _Tp>
    inline complex<_Tp>
    tanh(const complex<_Tp>& __z) { return __complex_tanh(__z.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    tanh(const complex<_Tp>& __z) { return __complex_tanh(__z); }
#endif


  // 26.2.8/9  pow(__x, __y): Returns the complex power base of __x
  //                          raised to the __y-th power.  The branch
  //                          cut is on the negative axis.
  template<typename _Tp>
    complex<_Tp>
    __complex_pow_unsigned(complex<_Tp> __x, unsigned __n)
    {
      complex<_Tp> __y = __n % 2 ? __x : complex<_Tp>(1);

      while (__n >>= 1)
        {
          __x *= __x;
          if (__n % 2)
            __y *= __x;
        }

      return __y;
    }

  // In C++11 mode we used to implement the resolution of
  // DR 844. complex pow return type is ambiguous.
  // thus the following overload was disabled in that mode.  However, doing
  // that causes all sorts of issues, see, for example:
  //   http://gcc.gnu.org/ml/libstdc++/2013-01/msg00058.html
  // and also PR57974.
  template<typename _Tp>
    inline complex<_Tp>
    pow(const complex<_Tp>& __z, int __n)
    {
      return __n < 0
	? complex<_Tp>(1) / std::__complex_pow_unsigned(__z, -(unsigned)__n)
        : std::__complex_pow_unsigned(__z, __n);
    }

  template<typename _Tp>
    complex<_Tp>
    pow(const complex<_Tp>& __x, const _Tp& __y)
    {
#if ! _GLIBCXX_USE_C99_COMPLEX
      if (__x == _Tp())
	return _Tp();
#endif
      if (__x.imag() == _Tp() && __x.real() > _Tp())
        return pow(__x.real(), __y);

      complex<_Tp> __t = std::log(__x);
      return std::polar<_Tp>(exp(__y * __t.real()), __y * __t.imag());
    }

  template<typename _Tp>
    inline complex<_Tp>
    __complex_pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
    { return __x == _Tp() ? _Tp() : std::exp(__y * std::log(__x)); }

#if _GLIBCXX_USE_C99_COMPLEX
  inline __complex__ float
  __complex_pow(__complex__ float __x, __complex__ float __y)
  { return __builtin_cpowf(__x, __y); }

  inline __complex__ double
  __complex_pow(__complex__ double __x, __complex__ double __y)
  { return __builtin_cpow(__x, __y); }

  inline __complex__ long double
  __complex_pow(const __complex__ long double& __x,
		const __complex__ long double& __y)
  { return __builtin_cpowl(__x, __y); }

  template<typename _Tp>
    inline complex<_Tp>
    pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
    { return __complex_pow(__x.__rep(), __y.__rep()); }
#else
  template<typename _Tp>
    inline complex<_Tp>
    pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
    { return __complex_pow(__x, __y); }
#endif

  template<typename _Tp>
    inline complex<_Tp>
    pow(const _Tp& __x, const complex<_Tp>& __y)
    {
      return __x > _Tp() ? std::polar<_Tp>(pow(__x, __y.real()),
					   __y.imag() * log(__x))
	                 : std::pow(complex<_Tp>(__x), __y);
    }

  /// 26.2.3  complex specializations
  /// complex<float> specialization
  template<>
    class complex<float>
    {
    public:
      typedef float value_type;
      typedef __complex__ float _ComplexT;

      _GLIBCXX_CONSTEXPR complex(_ComplexT __z) : _M_value(__z) { }

      _GLIBCXX_CONSTEXPR complex(float __r = 0.0f, float __i = 0.0f)
#if __cplusplus >= 201103L
      : _M_value{ __r, __i } { }
#else
      {
	__real__ _M_value = __r;
	__imag__ _M_value = __i;
      }
#endif

#if __cplusplus > 202002L
      template<typename _Up>
	explicit(!requires(_Up __u) { value_type{__u}; })
	constexpr complex(const complex<_Up>& __z)
	: _M_value{ value_type(__z.real()), value_type(__z.imag()) } { }
#else
      explicit _GLIBCXX_CONSTEXPR complex(const complex<double>&);
      explicit _GLIBCXX_CONSTEXPR complex(const complex<long double>&);
#endif

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      __attribute ((__abi_tag__ ("cxx11")))
      constexpr float
      real() const { return __real__ _M_value; }

      __attribute ((__abi_tag__ ("cxx11")))
      constexpr float
      imag() const { return __imag__ _M_value; }
#else
      float&
      real() { return __real__ _M_value; }

      const float&
      real() const { return __real__ _M_value; }

      float&
      imag() { return __imag__ _M_value; }

      const float&
      imag() const { return __imag__ _M_value; }
#endif

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      _GLIBCXX20_CONSTEXPR void
      real(float __val) { __real__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR void
      imag(float __val) { __imag__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR complex&
      operator=(float __f)
      {
	_M_value = __f;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator+=(float __f)
      {
	_M_value += __f;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator-=(float __f)
      {
	_M_value -= __f;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator*=(float __f)
      {
	_M_value *= __f;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator/=(float __f)
      {
	_M_value /= __f;
	return *this;
      }

      // Let the compiler synthesize the copy and assignment
      // operator.  It always does a pretty good job.
#if __cplusplus >= 201103L
      _GLIBCXX14_CONSTEXPR complex& operator=(const complex&) = default;
#endif

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator=(const complex<_Tp>&  __z)
	{
	  __real__ _M_value = __z.real();
	  __imag__ _M_value = __z.imag();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator+=(const complex<_Tp>& __z)
	{
	  _M_value += __z.__rep();
	  return *this;
	}

      template<class _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator-=(const complex<_Tp>& __z)
	{
	  _M_value -= __z.__rep();
	  return *this;
	}

      template<class _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator*=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value *= __t;
	  return *this;
	}

      template<class _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator/=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value /= __t;
	  return *this;
	}

      _GLIBCXX_CONSTEXPR _ComplexT __rep() const { return _M_value; }

    private:
      _ComplexT _M_value;
    };

  /// 26.2.3  complex specializations
  /// complex<double> specialization
  template<>
    class complex<double>
    {
    public:
      typedef double value_type;
      typedef __complex__ double _ComplexT;

      _GLIBCXX_CONSTEXPR complex(_ComplexT __z) : _M_value(__z) { }

      _GLIBCXX_CONSTEXPR complex(double __r = 0.0, double __i = 0.0)
#if __cplusplus >= 201103L
      : _M_value{ __r, __i } { }
#else
      {
	__real__ _M_value = __r;
	__imag__ _M_value = __i;
      }
#endif

#if __cplusplus > 202002L
      template<typename _Up>
	explicit(!requires(_Up __u) { value_type{__u}; })
	constexpr complex(const complex<_Up>& __z)
	: _M_value{ value_type(__z.real()), value_type(__z.imag()) } { }
#else
      _GLIBCXX_CONSTEXPR complex(const complex<float>& __z)
      : _M_value(__z.__rep()) { }

      explicit _GLIBCXX_CONSTEXPR complex(const complex<long double>&);
#endif

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      __attribute ((__abi_tag__ ("cxx11")))
      constexpr double
      real() const { return __real__ _M_value; }

      __attribute ((__abi_tag__ ("cxx11")))
      constexpr double
      imag() const { return __imag__ _M_value; }
#else
      double&
      real() { return __real__ _M_value; }

      const double&
      real() const { return __real__ _M_value; }

      double&
      imag() { return __imag__ _M_value; }

      const double&
      imag() const { return __imag__ _M_value; }
#endif

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      _GLIBCXX20_CONSTEXPR void
      real(double __val) { __real__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR void
      imag(double __val) { __imag__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR complex&
      operator=(double __d)
      {
	_M_value = __d;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator+=(double __d)
      {
	_M_value += __d;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator-=(double __d)
      {
	_M_value -= __d;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator*=(double __d)
      {
	_M_value *= __d;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator/=(double __d)
      {
	_M_value /= __d;
	return *this;
      }

      // The compiler will synthesize this, efficiently.
#if __cplusplus >= 201103L
      _GLIBCXX14_CONSTEXPR complex& operator=(const complex&) = default;
#endif

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator=(const complex<_Tp>& __z)
	{
	  _M_value = __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator+=(const complex<_Tp>& __z)
	{
	  _M_value += __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator-=(const complex<_Tp>& __z)
	{
	  _M_value -= __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator*=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value *= __t;
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator/=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value /= __t;
	  return *this;
	}

      _GLIBCXX_CONSTEXPR _ComplexT __rep() const { return _M_value; }

    private:
      _ComplexT _M_value;
    };

  /// 26.2.3  complex specializations
  /// complex<long double> specialization
  template<>
    class complex<long double>
    {
    public:
      typedef long double value_type;
      typedef __complex__ long double _ComplexT;

      _GLIBCXX_CONSTEXPR complex(_ComplexT __z) : _M_value(__z) { }

      _GLIBCXX_CONSTEXPR complex(long double __r = 0.0L,
				 long double __i = 0.0L)
#if __cplusplus >= 201103L
      : _M_value{ __r, __i } { }
#else
      {
	__real__ _M_value = __r;
	__imag__ _M_value = __i;
      }
#endif

#if __cplusplus > 202002L
      template<typename _Up>
	explicit(!requires(_Up __u) { value_type{__u}; })
	constexpr complex(const complex<_Up>& __z)
	: _M_value{ value_type(__z.real()), value_type(__z.imag()) } { }
#else
      _GLIBCXX_CONSTEXPR complex(const complex<float>& __z)
      : _M_value(__z.__rep()) { }

      _GLIBCXX_CONSTEXPR complex(const complex<double>& __z)
      : _M_value(__z.__rep()) { }
#endif

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      __attribute ((__abi_tag__ ("cxx11")))
      constexpr long double
      real() const { return __real__ _M_value; }

      __attribute ((__abi_tag__ ("cxx11")))
      constexpr long double
      imag() const { return __imag__ _M_value; }
#else
      long double&
      real() { return __real__ _M_value; }

      const long double&
      real() const { return __real__ _M_value; }

      long double&
      imag() { return __imag__ _M_value; }

      const long double&
      imag() const { return __imag__ _M_value; }
#endif

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // DR 387. std::complex over-encapsulated.
      _GLIBCXX20_CONSTEXPR void
      real(long double __val) { __real__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR void
      imag(long double __val) { __imag__ _M_value = __val; }

      _GLIBCXX20_CONSTEXPR complex&
      operator=(long double __r)
      {
	_M_value = __r;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator+=(long double __r)
      {
	_M_value += __r;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator-=(long double __r)
      {
	_M_value -= __r;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator*=(long double __r)
      {
	_M_value *= __r;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR complex&
      operator/=(long double __r)
      {
	_M_value /= __r;
	return *this;
      }

      // The compiler knows how to do this efficiently
#if __cplusplus >= 201103L
      _GLIBCXX14_CONSTEXPR complex& operator=(const complex&) = default;
#endif

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator=(const complex<_Tp>& __z)
	{
	  _M_value = __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
	operator+=(const complex<_Tp>& __z)
	{
	  _M_value += __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
	operator-=(const complex<_Tp>& __z)
	{
	  _M_value -= __z.__rep();
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
	operator*=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value *= __t;
	  return *this;
	}

      template<typename _Tp>
        _GLIBCXX20_CONSTEXPR complex&
	operator/=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value /= __t;
	  return *this;
	}

      _GLIBCXX_CONSTEXPR _ComplexT __rep() const { return _M_value; }

    private:
      _ComplexT _M_value;
    };

#if __cplusplus > 202002L
  template<typename _Tp>
    struct __complex_type
    { };

#ifdef __STDCPP_FLOAT16_T__
  template<>
    struct __complex_type<_Float16>
    { typedef __complex__ _Float16 type; };
#endif

#ifdef __STDCPP_FLOAT32_T__
  template<>
    struct __complex_type<_Float32>
    { typedef __complex__ _Float32 type; };
#endif

#ifdef __STDCPP_FLOAT64_T__
  template<>
    struct __complex_type<_Float64>
    { typedef __complex__ _Float64 type; };
#endif

#ifdef __STDCPP_FLOAT128_T__
  template<>
    struct __complex_type<_Float128>
    { typedef __complex__ _Float128 type; };
#endif

#ifdef __STDCPP_BFLOAT16_T__
  template<>
    struct __complex_type<__gnu_cxx::__bfloat16_t>
    { typedef __complex__ decltype(0.0bf16) type; };
#endif

  template<typename _Tp>
    requires requires { typename __complex_type<_Tp>::type; }
    class complex<_Tp>
    {
    public:
      typedef _Tp value_type;
      typedef typename std::__complex_type<_Tp>::type _ComplexT;

      constexpr complex(_ComplexT __z) : _M_value(__z) { }

      constexpr complex(_Tp __r = _Tp(), _Tp __i = _Tp())
      : _M_value{ __r, __i } { }

      template<typename _Up>
	explicit(!requires(_Up __u) { value_type{__u}; })
	constexpr complex(const complex<_Up>& __z)
	: _M_value{ value_type(__z.real()), value_type(__z.imag()) } { }

      constexpr _Tp
      real() const { return __real__ _M_value; }

      constexpr _Tp
      imag() const { return __imag__ _M_value; }

      constexpr void
      real(_Tp __val) { __real__ _M_value = __val; }

      constexpr void
      imag(_Tp __val) { __imag__ _M_value = __val; }

      constexpr complex&
      operator=(_Tp __f)
      {
	_M_value = __f;
	return *this;
      }

      constexpr complex&
      operator+=(_Tp __f)
      {
	_M_value += __f;
	return *this;
      }

      constexpr complex&
      operator-=(_Tp __f)
      {
	_M_value -= __f;
	return *this;
      }

      constexpr complex&
      operator*=(_Tp __f)
      {
	_M_value *= __f;
	return *this;
      }

      constexpr complex&
      operator/=(_Tp __f)
      {
	_M_value /= __f;
	return *this;
      }

      // Let the compiler synthesize the copy and assignment
      // operator.  It always does a pretty good job.
      constexpr complex& operator=(const complex&) = default;

      template<typename _Up>
	constexpr complex&
	operator=(const complex<_Up>&  __z)
	{
	  __real__ _M_value = __z.real();
	  __imag__ _M_value = __z.imag();
	  return *this;
	}

      template<typename _Up>
	constexpr complex&
	operator+=(const complex<_Up>& __z)
	{
	  _M_value += __z.__rep();
	  return *this;
	}

      template<class _Up>
	constexpr complex&
	operator-=(const complex<_Up>& __z)
	{
	  _M_value -= __z.__rep();
	  return *this;
	}

      template<class _Up>
	constexpr complex&
	operator*=(const complex<_Up>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value *= __t;
	  return *this;
	}

      template<class _Up>
	constexpr complex&
	operator/=(const complex<_Up>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value /= __t;
	  return *this;
	}

      constexpr _ComplexT __rep() const { return _M_value; }

    private:
      _ComplexT _M_value;
    };
#endif

#if __cplusplus <= 202002L
  // These bits have to be at the end of this file, so that the
  // specializations have all been defined.
  inline _GLIBCXX_CONSTEXPR
  complex<float>::complex(const complex<double>& __z)
  : _M_value(__z.__rep()) { }

  inline _GLIBCXX_CONSTEXPR
  complex<float>::complex(const complex<long double>& __z)
  : _M_value(__z.__rep()) { }

  inline _GLIBCXX_CONSTEXPR
  complex<double>::complex(const complex<long double>& __z)
  : _M_value(__z.__rep()) { }
#endif

  // Inhibit implicit instantiations for required instantiations,
  // which are defined via explicit instantiations elsewhere.
  // NB:  This syntax is a GNU extension.
#if _GLIBCXX_EXTERN_TEMPLATE
  extern template istream& operator>>(istream&, complex<float>&);
  extern template ostream& operator<<(ostream&, const complex<float>&);
  extern template istream& operator>>(istream&, complex<double>&);
  extern template ostream& operator<<(ostream&, const complex<double>&);
  extern template istream& operator>>(istream&, complex<long double>&);
  extern template ostream& operator<<(ostream&, const complex<long double>&);

#ifdef _GLIBCXX_USE_WCHAR_T
  extern template wistream& operator>>(wistream&, complex<float>&);
  extern template wostream& operator<<(wostream&, const complex<float>&);
  extern template wistream& operator>>(wistream&, complex<double>&);
  extern template wostream& operator<<(wostream&, const complex<double>&);
  extern template wistream& operator>>(wistream&, complex<long double>&);
  extern template wostream& operator<<(wostream&, const complex<long double>&);
#endif
#endif

  /// @} group complex_numbers

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace

#if __cplusplus >= 201103L

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  // Forward declarations.
  template<typename _Tp> std::complex<_Tp> acos(const std::complex<_Tp>&);
  template<typename _Tp> std::complex<_Tp> asin(const std::complex<_Tp>&);
  template<typename _Tp> std::complex<_Tp> atan(const std::complex<_Tp>&);

  template<typename _Tp> std::complex<_Tp> acosh(const std::complex<_Tp>&);
  template<typename _Tp> std::complex<_Tp> asinh(const std::complex<_Tp>&);
  template<typename _Tp> std::complex<_Tp> atanh(const std::complex<_Tp>&);
  // DR 595.
  template<typename _Tp> _Tp               fabs(const std::complex<_Tp>&);

  template<typename _Tp>
    inline std::complex<_Tp>
    __complex_acos(const std::complex<_Tp>& __z)
    {
      const std::complex<_Tp> __t = std::asin(__z);
      const _Tp __pi_2 = 1.5707963267948966192313216916397514L;
      return std::complex<_Tp>(__pi_2 - __t.real(), -__t.imag());
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
#if defined(__STDCPP_FLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ _Float16
  __complex_acos(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_cacosf(__z)); }

  inline __complex__ _Float16
  __complex_asin(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_casinf(__z)); }

  inline __complex__ _Float16
  __complex_atan(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_catanf(__z)); }

  inline __complex__ _Float16
  __complex_acosh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_cacoshf(__z)); }

  inline __complex__ _Float16
  __complex_asinh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_casinhf(__z)); }

  inline __complex__ _Float16
  __complex_atanh(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_catanhf(__z)); }
#endif

#if defined(__STDCPP_FLOAT32_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ _Float32
  __complex_acos(__complex__ _Float32 __z)
  { return __builtin_cacosf(__z); }

  inline __complex__ _Float32
  __complex_asin(__complex__ _Float32 __z)
  { return __builtin_casinf(__z); }

  inline __complex__ _Float32
  __complex_atan(__complex__ _Float32 __z)
  { return __builtin_catanf(__z); }

  inline __complex__ _Float32
  __complex_acosh(__complex__ _Float32 __z)
  { return __builtin_cacoshf(__z); }

  inline __complex__ _Float32
  __complex_asinh(__complex__ _Float32 __z)
  { return __builtin_casinhf(__z); }

  inline __complex__ _Float32
  __complex_atanh(__complex__ _Float32 __z)
  { return __builtin_catanhf(__z); }
#endif

#if defined(__STDCPP_FLOAT64_T__) && defined(_GLIBCXX_DOUBLE_IS_IEEE_BINARY64)
  inline __complex__ _Float64
  __complex_acos(__complex__ _Float64 __z)
  { return __builtin_cacos(__z); }

  inline __complex__ _Float64
  __complex_asin(__complex__ _Float64 __z)
  { return __builtin_casin(__z); }

  inline __complex__ _Float64
  __complex_atan(__complex__ _Float64 __z)
  { return __builtin_catan(__z); }

  inline __complex__ _Float64
  __complex_acosh(__complex__ _Float64 __z)
  { return __builtin_cacosh(__z); }

  inline __complex__ _Float64
  __complex_asinh(__complex__ _Float64 __z)
  { return __builtin_casinh(__z); }

  inline __complex__ _Float64
  __complex_atanh(__complex__ _Float64 __z)
  { return __builtin_catanh(__z); }
#endif

#if defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_LDOUBLE_IS_IEEE_BINARY128)
  inline __complex__ _Float128
  __complex_acos(__complex__ _Float128 __z)
  { return __builtin_cacosl(__z); }

  inline __complex__ _Float128
  __complex_asin(__complex__ _Float128 __z)
  { return __builtin_casinl(__z); }

  inline __complex__ _Float128
  __complex_atan(__complex__ _Float128 __z)
  { return __builtin_catanl(__z); }

  inline __complex__ _Float128
  __complex_acosh(__complex__ _Float128 __z)
  { return __builtin_cacoshl(__z); }

  inline __complex__ _Float128
  __complex_asinh(__complex__ _Float128 __z)
  { return __builtin_casinhl(__z); }

  inline __complex__ _Float128
  __complex_atanh(__complex__ _Float128 __z)
  { return __builtin_catanhl(__z); }
#elif defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_HAVE_FLOAT128_MATH)
  inline __complex__ _Float128
  __complex_acos(__complex__ _Float128 __z)
  { return __builtin_cacosf128(__z); }

  inline __complex__ _Float128
  __complex_asin(__complex__ _Float128 __z)
  { return __builtin_casinf128(__z); }

  inline __complex__ _Float128
  __complex_atan(__complex__ _Float128 __z)
  { return __builtin_catanf128(__z); }

  inline __complex__ _Float128
  __complex_acosh(__complex__ _Float128 __z)
  { return __builtin_cacoshf128(__z); }

  inline __complex__ _Float128
  __complex_asinh(__complex__ _Float128 __z)
  { return __builtin_casinhf128(__z); }

  inline __complex__ _Float128
  __complex_atanh(__complex__ _Float128 __z)
  { return __builtin_catanhf128(__z); }
#endif

#if defined(__STDCPP_BFLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ decltype(0.0bf16)
  __complex_acos(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_cacosf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_asin(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_casinf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_atan(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_catanf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_acosh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_cacoshf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_asinh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_casinhf(__z)); }

  inline __complex__ decltype(0.0bf16)
  __complex_atanh(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_catanhf(__z)); }
#endif
#endif

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_acos(__complex__ float __z)
  { return __builtin_cacosf(__z); }

  inline __complex__ double
  __complex_acos(__complex__ double __z)
  { return __builtin_cacos(__z); }

  inline __complex__ long double
  __complex_acos(const __complex__ long double& __z)
  { return __builtin_cacosl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    acos(const std::complex<_Tp>& __z)
    { return __complex_acos(__z.__rep()); }
#else
  /// acos(__z) [8.1.2].
  //  Effects:  Behaves the same as C99 function cacos, defined
  //            in subclause 7.3.5.1.
  template<typename _Tp>
    inline std::complex<_Tp>
    acos(const std::complex<_Tp>& __z)
    { return __complex_acos(__z); }
#endif

  template<typename _Tp>
    inline std::complex<_Tp>
    __complex_asin(const std::complex<_Tp>& __z)
    {
      std::complex<_Tp> __t(-__z.imag(), __z.real());
      __t = std::asinh(__t);
      return std::complex<_Tp>(__t.imag(), -__t.real());
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_asin(__complex__ float __z)
  { return __builtin_casinf(__z); }

  inline __complex__ double
  __complex_asin(__complex__ double __z)
  { return __builtin_casin(__z); }

  inline __complex__ long double
  __complex_asin(const __complex__ long double& __z)
  { return __builtin_casinl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    asin(const std::complex<_Tp>& __z)
    { return __complex_asin(__z.__rep()); }
#else
  /// asin(__z) [8.1.3].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.5.2.
  template<typename _Tp>
    inline std::complex<_Tp>
    asin(const std::complex<_Tp>& __z)
    { return __complex_asin(__z); }
#endif

  template<typename _Tp>
    std::complex<_Tp>
    __complex_atan(const std::complex<_Tp>& __z)
    {
      const _Tp __r2 = __z.real() * __z.real();
      const _Tp __x = _Tp(1.0) - __r2 - __z.imag() * __z.imag();

      _Tp __num = __z.imag() + _Tp(1.0);
      _Tp __den = __z.imag() - _Tp(1.0);

      __num = __r2 + __num * __num;
      __den = __r2 + __den * __den;

      return std::complex<_Tp>(_Tp(0.5) * atan2(_Tp(2.0) * __z.real(), __x),
			       _Tp(0.25) * log(__num / __den));
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_atan(__complex__ float __z)
  { return __builtin_catanf(__z); }

  inline __complex__ double
  __complex_atan(__complex__ double __z)
  { return __builtin_catan(__z); }

  inline __complex__ long double
  __complex_atan(const __complex__ long double& __z)
  { return __builtin_catanl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    atan(const std::complex<_Tp>& __z)
    { return __complex_atan(__z.__rep()); }
#else
  /// atan(__z) [8.1.4].
  //  Effects:  Behaves the same as C99 function catan, defined
  //            in subclause 7.3.5.3.
  template<typename _Tp>
    inline std::complex<_Tp>
    atan(const std::complex<_Tp>& __z)
    { return __complex_atan(__z); }
#endif

  template<typename _Tp>
    std::complex<_Tp>
    __complex_acosh(const std::complex<_Tp>& __z)
    {
      // Kahan's formula.
      return _Tp(2.0) * std::log(std::sqrt(_Tp(0.5) * (__z + _Tp(1.0)))
				 + std::sqrt(_Tp(0.5) * (__z - _Tp(1.0))));
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_acosh(__complex__ float __z)
  { return __builtin_cacoshf(__z); }

  inline __complex__ double
  __complex_acosh(__complex__ double __z)
  { return __builtin_cacosh(__z); }

  inline __complex__ long double
  __complex_acosh(const __complex__ long double& __z)
  { return __builtin_cacoshl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    acosh(const std::complex<_Tp>& __z)
    { return __complex_acosh(__z.__rep()); }
#else
  /// acosh(__z) [8.1.5].
  //  Effects:  Behaves the same as C99 function cacosh, defined
  //            in subclause 7.3.6.1.
  template<typename _Tp>
    inline std::complex<_Tp>
    acosh(const std::complex<_Tp>& __z)
    { return __complex_acosh(__z); }
#endif

  template<typename _Tp>
    std::complex<_Tp>
    __complex_asinh(const std::complex<_Tp>& __z)
    {
      std::complex<_Tp> __t((__z.real() - __z.imag())
			    * (__z.real() + __z.imag()) + _Tp(1.0),
			    _Tp(2.0) * __z.real() * __z.imag());
      __t = std::sqrt(__t);

      return std::log(__t + __z);
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_asinh(__complex__ float __z)
  { return __builtin_casinhf(__z); }

  inline __complex__ double
  __complex_asinh(__complex__ double __z)
  { return __builtin_casinh(__z); }

  inline __complex__ long double
  __complex_asinh(const __complex__ long double& __z)
  { return __builtin_casinhl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    asinh(const std::complex<_Tp>& __z)
    { return __complex_asinh(__z.__rep()); }
#else
  /// asinh(__z) [8.1.6].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.6.2.
  template<typename _Tp>
    inline std::complex<_Tp>
    asinh(const std::complex<_Tp>& __z)
    { return __complex_asinh(__z); }
#endif

  template<typename _Tp>
    std::complex<_Tp>
    __complex_atanh(const std::complex<_Tp>& __z)
    {
      const _Tp __i2 = __z.imag() * __z.imag();
      const _Tp __x = _Tp(1.0) - __i2 - __z.real() * __z.real();

      _Tp __num = _Tp(1.0) + __z.real();
      _Tp __den = _Tp(1.0) - __z.real();

      __num = __i2 + __num * __num;
      __den = __i2 + __den * __den;

      return std::complex<_Tp>(_Tp(0.25) * (log(__num) - log(__den)),
			       _Tp(0.5) * atan2(_Tp(2.0) * __z.imag(), __x));
    }

#if _GLIBCXX_USE_C99_COMPLEX_TR1
  inline __complex__ float
  __complex_atanh(__complex__ float __z)
  { return __builtin_catanhf(__z); }

  inline __complex__ double
  __complex_atanh(__complex__ double __z)
  { return __builtin_catanh(__z); }

  inline __complex__ long double
  __complex_atanh(const __complex__ long double& __z)
  { return __builtin_catanhl(__z); }

  template<typename _Tp>
    inline std::complex<_Tp>
    atanh(const std::complex<_Tp>& __z)
    { return __complex_atanh(__z.__rep()); }
#else
  /// atanh(__z) [8.1.7].
  //  Effects:  Behaves the same as C99 function catanh, defined
  //            in subclause 7.3.6.3.
  template<typename _Tp>
    inline std::complex<_Tp>
    atanh(const std::complex<_Tp>& __z)
    { return __complex_atanh(__z); }
#endif

  template<typename _Tp>
    inline _Tp
    /// fabs(__z) [8.1.8].
    //  Effects:  Behaves the same as C99 function cabs, defined
    //            in subclause 7.3.8.1.
    fabs(const std::complex<_Tp>& __z)
    { return std::abs(__z); }

  /// Additional overloads [8.1.9].
  template<typename _Tp>
    inline typename __gnu_cxx::__promote<_Tp>::__type
    arg(_Tp __x)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
#if (_GLIBCXX11_USE_C99_MATH && !_GLIBCXX_USE_C99_FP_MACROS_DYNAMIC)
      return std::signbit(__x) ? __type(3.1415926535897932384626433832795029L)
	                       : __type();
#else
      return std::arg(std::complex<__type>(__x));
#endif
    }

  template<typename _Tp>
    _GLIBCXX_CONSTEXPR inline typename __gnu_cxx::__promote<_Tp>::__type
    imag(_Tp)
    { return _Tp(); }

  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR inline typename __gnu_cxx::__promote<_Tp>::__type
    norm(_Tp __x)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return __type(__x) * __type(__x);
    }

  template<typename _Tp>
    _GLIBCXX_CONSTEXPR inline typename __gnu_cxx::__promote<_Tp>::__type
    real(_Tp __x)
    { return __x; }

  template<typename _Tp, typename _Up>
    inline std::complex<typename __gnu_cxx::__promote_2<_Tp, _Up>::__type>
    pow(const std::complex<_Tp>& __x, const _Up& __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return std::pow(std::complex<__type>(__x), __type(__y));
    }

  template<typename _Tp, typename _Up>
    inline std::complex<typename __gnu_cxx::__promote_2<_Tp, _Up>::__type>
    pow(const _Tp& __x, const std::complex<_Up>& __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return std::pow(__type(__x), std::complex<__type>(__y));
    }

  template<typename _Tp, typename _Up>
    inline std::complex<typename __gnu_cxx::__promote_2<_Tp, _Up>::__type>
    pow(const std::complex<_Tp>& __x, const std::complex<_Up>& __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return std::pow(std::complex<__type>(__x),
		      std::complex<__type>(__y));
    }

  // Forward declarations.
  // DR 781.
  template<typename _Tp>
    std::complex<_Tp> proj(const std::complex<_Tp>&);

  // Generic implementation of std::proj, does not work for infinities.
  template<typename _Tp>
    inline std::complex<_Tp>
    __complex_proj(const std::complex<_Tp>& __z)
    { return __z; }

#if _GLIBCXX_USE_C99_COMPLEX
  inline complex<float>
  __complex_proj(const complex<float>& __z)
  { return __builtin_cprojf(__z.__rep()); }

  inline complex<double>
  __complex_proj(const complex<double>& __z)
  { return __builtin_cproj(__z.__rep()); }

  inline complex<long double>
  __complex_proj(const complex<long double>& __z)
  { return __builtin_cprojl(__z.__rep()); }

#if __cplusplus > 202002L
#if defined(__STDCPP_FLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ _Float16
  __complex_proj(__complex__ _Float16 __z)
  { return static_cast<__complex__ _Float16>(__builtin_cprojf(__z)); }
#endif

#if defined(__STDCPP_FLOAT32_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ _Float32
  __complex_proj(__complex__ _Float32 __z)
  { return __builtin_cprojf(__z); }
#endif

#if defined(__STDCPP_FLOAT64_T__) && defined(_GLIBCXX_DOUBLE_IS_IEEE_BINARY64)
  inline __complex__ _Float64
  __complex_proj(__complex__ _Float64 __z)
  { return __builtin_cproj(__z); }
#endif

#if defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_LDOUBLE_IS_IEEE_BINARY128)
  inline __complex__ _Float128
  __complex_proj(__complex__ _Float128 __z)
  { return __builtin_cprojl(__z); }
#elif defined(__STDCPP_FLOAT128_T__) && defined(_GLIBCXX_HAVE_FLOAT128_MATH)
  inline __complex__ _Float128
  __complex_proj(__complex__ _Float128 __z)
  { return __builtin_cprojf128(__z); }
#endif

#if defined(__STDCPP_BFLOAT16_T__) && defined(_GLIBCXX_FLOAT_IS_IEEE_BINARY32)
  inline __complex__ decltype(0.0bf16)
  __complex_proj(__complex__ decltype(0.0bf16) __z)
  { return static_cast<__complex__ decltype(0.0bf16)>(__builtin_cprojf(__z)); }
#endif

  template<typename _Tp>
    requires requires { typename __complex_type<_Tp>::type; }
    inline complex<_Tp>
    __complex_proj(const complex<_Tp>& __z)
    { return __complex_proj(__z.__rep()); }
#endif

#elif defined _GLIBCXX_USE_C99_MATH_TR1
  inline complex<float>
  __complex_proj(const complex<float>& __z)
  {
    if (__builtin_isinf(__z.real()) || __builtin_isinf(__z.imag()))
      return complex<float>(__builtin_inff(),
			    __builtin_copysignf(0.0f, __z.imag()));
    return __z;
  }

  inline complex<double>
  __complex_proj(const complex<double>& __z)
  {
    if (__builtin_isinf(__z.real()) || __builtin_isinf(__z.imag()))
      return complex<double>(__builtin_inf(),
			     __builtin_copysign(0.0, __z.imag()));
    return __z;
  }

  inline complex<long double>
  __complex_proj(const complex<long double>& __z)
  {
    if (__builtin_isinf(__z.real()) || __builtin_isinf(__z.imag()))
      return complex<long double>(__builtin_infl(),
				  __builtin_copysignl(0.0l, __z.imag()));
    return __z;
  }
#endif

  template<typename _Tp>
    inline std::complex<_Tp>
    proj(const std::complex<_Tp>& __z)
    { return __complex_proj(__z); }

  // Overload for scalars
  template<typename _Tp>
    inline std::complex<typename __gnu_cxx::__promote<_Tp>::__type>
    proj(_Tp __x)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return std::proj(std::complex<__type>(__x));
    }

  template<typename _Tp>
    inline _GLIBCXX20_CONSTEXPR
	std::complex<typename __gnu_cxx::__promote<_Tp>::__type>
    conj(_Tp __x)
    {
      typedef typename __gnu_cxx::__promote<_Tp>::__type __type;
      return std::complex<__type>(__x, -__type());
    }

#if __cplusplus > 201103L

inline namespace literals {
inline namespace complex_literals {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wliteral-suffix"
#define __cpp_lib_complex_udls 201309L

  constexpr std::complex<float>
  operator""if(long double __num)
  { return std::complex<float>{0.0F, static_cast<float>(__num)}; }

  constexpr std::complex<float>
  operator""if(unsigned long long __num)
  { return std::complex<float>{0.0F, static_cast<float>(__num)}; }

  constexpr std::complex<double>
  operator""i(long double __num)
  { return std::complex<double>{0.0, static_cast<double>(__num)}; }

  constexpr std::complex<double>
  operator""i(unsigned long long __num)
  { return std::complex<double>{0.0, static_cast<double>(__num)}; }

  constexpr std::complex<long double>
  operator""il(long double __num)
  { return std::complex<long double>{0.0L, __num}; }

  constexpr std::complex<long double>
  operator""il(unsigned long long __num)
  { return std::complex<long double>{0.0L, static_cast<long double>(__num)}; }

#pragma GCC diagnostic pop
} // inline namespace complex_literals
} // inline namespace literals

#endif // C++14

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace

#endif  // C++11

#ifdef COMPLEX_INT8
    typedef std::complex<i8>    ci8;
#endif
#ifdef COMPLEX_INT16
    typedef std::complex<i16>   ci16;
#endif
#ifdef COMPLEX_INT32
    typedef std::complex<i32>   ci32;
#endif
#ifdef COMPLEX_INT64
    typedef std::complex<i64>   ci64;
#endif
#ifdef COMPLEX_INT128
    typedef std::complex<i128>  ci128;
#endif
#ifdef COMPLEX_UINT8
    typedef std::complex<u8>    cu8;
#endif
#ifdef COMPLEX_UINT16
    typedef std::complex<u16>   cu16;
#endif
#ifdef COMPLEX_UINT32
    typedef std::complex<u32>   cu32;
#endif
#ifdef COMPLEX_UINT64
    typedef std::complex<u64>   cu64;
#endif
#ifdef COMPLEX_UINT128
    typedef std::complex<u128>  cu128;
#endif
#ifdef COMPLEX_FLOAT16
    typedef std::complex<f16>   cf16;
#endif
#ifdef COMPLEX_FLOAT32
    typedef std::complex<f32>   cf32;
#endif
#ifdef COMPLEX_FLOAT64
    typedef std::complex<f64>   cf64;
#endif
#ifdef COMPLEX_FLOAT128
    typedef std::complex<f128>  cf128;
#endif

template<typename T> inline T&              operator+=  (T& lhs, const std::complex<T> rhs);
template<typename T> inline T&              operator-=  (T& lhs, const std::complex<T> rhs);
template<typename T> inline T&              operator*=  (T& lhs, const std::complex<T> rhs);
template<typename T> inline T&              operator/=  (T& lhs, const std::complex<T> rhs);

template<typename T> inline T& operator+=(T& lhs, const std::complex<T> rhs)
{
    lhs += rhs.real();
    return lhs;
}

template<typename T> inline T& operator-=(T& lhs, const std::complex<T> rhs)
{
    lhs -= rhs.real();
    return lhs;
}

template<typename T> inline T& operator*=(T& lhs, const std::complex<T> rhs)
{
    std::complex<T> c = lhs * rhs;
    lhs = c.real();
    return lhs;
}

template<typename T> inline T& operator/=(T& lhs, const std::complex<T> rhs)
{
    std::complex<T> c = lhs / rhs;
    lhs = c.real();
    return lhs;
}

#endif  /* _GLIBCXX_COMPLEX */
