
pub struct Izhikevich<T> {
    v: T,
    u: T,
    i: T,
    t: T
}

impl Izhikevich<f64> {
    pub const fn new_const( v: f64, u: f64, i: f64, t: f64 ) -> Self {
        Self { v, u, i, t }
    }

    pub fn new( v: f64, u: f64, i: f64, t: f64 ) -> Self {
        Self { v, u, i, t }
    }
}
