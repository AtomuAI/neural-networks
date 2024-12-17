// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Add, Sub, Mul, Div, AddAssign }
};
use num::traits::{ Pow, FromPrimitive };

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::node::NodeLayer;

//pub mod activation;
pub mod bias;
//pub mod convolution;
//pub mod cost;
pub mod dense;
//pub mod dropout;
//pub mod mask;
pub mod node;
//pub mod normalization;
//pub mod pooling;
//pub mod softmax;

pub trait Inference<Input, Output> {
    fn inference( &self, input: &Input, output: &mut Output );
}

pub trait InferenceAssign<Input> {
    fn inference_assign( &self, input: &mut Input );
}

pub trait Backprop<Input, Output, InputDelta, OutputDelta, Gradient> {
    fn backprop( &self, input: &Input, output: &Output, input_delta: &InputDelta, output_delta: &mut OutputDelta, grad: &mut Gradient );
}

pub trait BackpropAssign<Output, OutputDelta, Gradient> {
    fn backprop_assign( &self, output: &Output, output_delta: &mut OutputDelta, grad: &mut Gradient );
}

pub trait GradientDecent<T, Gradient> {
    fn grad_descent( &mut self, step: T, grad: &Gradient );
}

pub trait GradientDescentMomentum<T, Gradient> {
    fn grad_descent_momentum( &mut self, step: T, grad: &Gradient, momentum: &mut Gradient );
}

pub trait GradientDescentAdam<T, Time, Gradient>
{
    fn grad_descent_adam( &mut self, step: T, time: Time, beta_m: T, beta_v: T, epsilon: T, grad: &Gradient, momentum: &mut Gradient, velocity: &mut Gradient );
}

impl<T, const DIM: usize, M> GradientDecent<T, Tensor<T, DIM, M>> for Tensor<T, DIM, M>
where
    T: Default + Debug + Copy + Mul<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>
{
    fn grad_descent( &mut self, step: T, grad: &Tensor<T, DIM, M> ) {
        self.iter_mut().zip( grad.iter() ).for_each( |( param, &grad )| {
            *param += step * grad;
        });
    }
}

impl<T, const DIM: usize, M> GradientDescentMomentum<T, Tensor<T, DIM, M>> for Tensor<T, DIM, M>
where
    T: Default + Debug + Copy + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>
{
    fn grad_descent_momentum( &mut self, step: T, grad: &Tensor<T, DIM, M>, momentum: &mut Tensor<T, DIM, M> ) {
        self.iter_mut().zip( grad.iter() ).zip( momentum.iter_mut() ).for_each( |( ( param, &grad ), momentum )| {
            *param += step * ( *momentum + grad );
            *momentum = grad;
        });
    }
}

impl<T, Time, const DIM: usize, M> GradientDescentAdam<T, Time, Tensor<T, DIM, M>> for Tensor<T, DIM, M>
where
    T: Default + Debug + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + AddAssign + Pow<T, Output = T> + Pow<Time, Output = T> + Pow<i32, Output = T> + Pow<f32, Output = T> + FromPrimitive,
    Time: Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>
{
    fn grad_descent_adam( &mut self, step: T, time: Time, beta_m: T, beta_v: T, epsilon: T, grad: &Tensor<T, DIM, M>, momentum: &mut Tensor<T, DIM, M>, velocity: &mut Tensor<T, DIM, M> ) {
        let beta_m_subpow = T::from_i32( 1 ).unwrap() - beta_m.pow( time );
        let beta_v_subpow = T::from_i32( 1 ).unwrap() - beta_v.pow( time );
        let beta_m_sub = T::from_i32( 1 ).unwrap() - beta_m;
        let beta_v_sub = T::from_i32( 1 ).unwrap() - beta_v;

        self.iter_mut().zip( grad.iter() ).zip( momentum.iter_mut() ).zip( velocity.iter_mut() ).for_each( |( ( ( param, &grad ), momentum ), velocity ) | {
            *momentum = ( beta_m * *momentum ) + ( beta_m_sub * grad );
            *velocity = ( beta_v * *velocity ) + ( beta_v_sub * grad.pow( 2 ) );

            let momentum_hat = *momentum / beta_m_subpow;
            let velocity_hat = *velocity / beta_v_subpow;

            *param += step * ( momentum_hat / ( velocity_hat.pow( 0.5 ) + epsilon ) );
        });
    }
}

pub trait Layer {}
