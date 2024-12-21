// Copyright 2024 Bewusstsein Labs

use std::{
    ops::{ Add, Sub, Mul, AddAssign },
    cmp::PartialOrd
};
use num::traits::{ Num, Pow, FromPrimitive };

/// Izhikevich neuron model
///
pub struct Izhikevich<T> {
    /// Membrane potential
    ///
    v: T,
    /// Recovery variable
    ///
    u: T
}

impl<T> Izhikevich<T> {
    pub const fn new_const( v: T, u: T ) -> Self {
        Self { v, u }
    }

    pub fn new( v: T, u: T ) -> Self {
        Self { v, u }
    }

    pub fn v( &self ) -> T
    where
        T: Copy
    {
        self.v
    }

    pub fn u( &self ) -> T
    where
        T: Copy
    {
        self.u
    }

    #[inline]
    fn potential( e: T, f: T, g: T, v: T, u: T, i: T ) -> T
    where
        T: Num + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Pow<i32, Output = T>
    {
        ( e * v.pow( 2 ) ) + ( f * v ) + g - u + i
    }

    #[inline]
    fn recovery( a: T, b: T, v: T, u: T ) -> T
    where
        T: Num + Copy + Sub<Output = T> + Mul<Output = T>
    {
        a * ( ( b * v ) - u )
    }

    pub fn update( &mut self, step: T, a: T, b: T, c: T, d: T, e: T, f: T, g: T, vt: T, i: T ) -> T
    where
        T: Num + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign + PartialOrd + Pow<i32, Output = T> + FromPrimitive
    {
        if self.v < vt {
            self.v += step * Self::potential( e, f, g, self.v, self.u, i );
            self.u += step * Self::recovery( a, b, self.v, self.u );
            self.v
        } else {
            self.v = c;
            self.u += d;
            self.v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich() {
        let step = 0.1;
        let a = 0.02;
        let b = 0.2;
        let c = -65.0;
        let d = 8.0;
        let e = 0.04;
        let f = 5.0;
        let g = 140.0;
        let vt = 30.0;
        let i = 14.0;
        let mut izhikevich = Izhikevich::new( c, b * c );

        for _ in 0..1000 {
            let v = izhikevich.update( step, a, b, c, d, e, f, g, vt, i );
            println!( "{}", v );
        }
    }
}
