// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Add, Sub, Div, AddAssign, Deref, DerefMut, Mul }
};
use num::traits::{ Pow, FromPrimitive };

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits, contract }, shape::Shape };
use crate::layer::{
    Layer,
    Inference,
    Backprop,
    GradientDecent, GradientDescentMomentum, GradientDescentAdam,
    node::NodeLayer
};

#[derive( Clone, Default, Debug )]
pub struct DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    weights: Tensor<T, 2, M>,
}

impl<T, M> DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>
{
    pub fn new( shape: Shape<2>, memory: <Memory<T, M> as MemoryTraits>::New ) -> Self {
        Self {
            weights: Tensor::<T, 2, M>::new( shape, memory ),
        }
    }

    pub fn iter( &self ) -> impl Iterator<Item = &T> {
        self.weights.iter()
    }

    pub fn iter_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.weights.iter_mut()
    }
}

impl<T, M> Deref for DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Target = Tensor<T, 2, M>;

    fn deref( &self ) -> &Self::Target {
        &self.weights
    }
}

impl<T, M> DerefMut for DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.weights
    }
}

impl<T, M> Layer for DenseLayer<T, M>
where
    T: Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{}

impl<T, M, O, P> Inference<NodeLayer<T, O>, NodeLayer<T, P>> for DenseLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Add<Output = T> + Mul<Output = T> + AddAssign,
    M: MemoryType,
    O: MemoryType,
    P: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
    Memory<T, P>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>,
    Tensor<T, 5, O>: TensorTraits<T, 5, O>,
    Tensor<T, 5, P>: TensorTraits<T, 5, P>
{
    fn inference( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> ) {
        contract( &self.weights, & **input, &mut **output, &[1], &[0] );
    }
}

impl<T, M, P, O> Backprop<NodeLayer<T, P>, NodeLayer<T, O>, Tensor<T, 5, P>, Tensor<T, 5, O>, Tensor<T, 2, M>> for DenseLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Add<Output = T> + Mul<Output = T> + AddAssign,
    M: MemoryType,
    O: MemoryType,
    P: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
    Memory<T, P>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>,
    Tensor<T, 5, O>: TensorTraits<T, 5, O>,
    Tensor<T, 5, P>: TensorTraits<T, 5, P>
{
    fn backprop( &self, input: &NodeLayer<T, P>, _: &NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, grad: &mut Tensor<T, 2, M> ) {
        contract( output_delta, & **input, grad, &[1], &[0] );
        contract( &self.weights, output_delta, input_delta, &[1], &[0] );
    }
}

impl<T, M> GradientDecent<T, Tensor<T, 2, M>> for DenseLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Mul<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>
{
    fn grad_descent( &mut self, step: T, grad: &Tensor<T, 2, M> ) {
        self.weights.grad_descent( step, grad );
    }
}

impl<T, M> GradientDescentMomentum<T, Tensor<T, 2, M>> for DenseLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>
{
    fn grad_descent_momentum( &mut self, step: T, grad: &Tensor<T, 2, M>, momentum: &mut Tensor<T, 2, M> ) {
        self.weights.grad_descent_momentum( step, grad, momentum );
    }
}

impl<T, Time, M> GradientDescentAdam<T, Time, Tensor<T, 2, M>> for DenseLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + AddAssign + Pow<T, Output = T> + Pow<Time, Output = T> + Pow<i32, Output = T> + Pow<f32, Output = T> + FromPrimitive,
    Time: Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>
{
    fn grad_descent_adam( &mut self, step: T, time: Time, beta_m: T, beta_v: T, epsilon: T, grad: &Tensor<T, 2, M>, momentum: &mut Tensor<T, 2, M>, velocity: &mut Tensor<T, 2, M> ) {
        self.weights.grad_descent_adam( step, time, beta_m, beta_v, epsilon, grad, momentum, velocity );
    }
}

#[cfg(test)]
mod tests {
    use crate::layer::{Backprop, GradientDecent};

    #[test]
    fn inference_test() {
        use memory::{ Memory, stack::Stack };
        use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
        use crate::layer::{
            Layer,
            Inference,
            Backprop,
            node::NodeLayer
        };
        use super::DenseLayer;

        let input = NodeLayer::<f32, Stack<4>>::new(
            [ 4, 1, 1, 1, 1 ].into(),
            [
                1.0, 1.0, 1.0, 1.0
            ],
        );

        let mut output = NodeLayer::<f32, Stack<4>>::new(
            [ 4, 1, 1, 1, 1 ].into(),
            [
                0.0, 0.0, 0.0, 0.0
            ],
        );

        let mut dense = DenseLayer::<f32, Stack<16>>::new(
            [ 4, 4 ].into(),
            [
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0
            ],
        );

        let mut input_delta = Tensor::<f32, 5, Stack<4>>::new(
            [ 4, 1, 1, 1, 1 ].into(),
            [ 0.1; 4 ]
        );

        let output_delta = Tensor::<f32, 5, Stack<4>>::new(
            [ 4, 1, 1, 1, 1 ].into(),
            [ 0.1; 4 ]
        );

        let mut grad = Tensor::<f32, 2, Stack<16>>::new(
            [ 4, 4 ].into(),
            [ 0.1; 16 ]
        );

        println!( "Before:");
        println!( "{:?}", input );
        println!( "{:?}", output );

        dense.inference( &input, &mut output );

        println!( "After:");
        println!( "{:?}", input );
        println!( "{:?}", output );

        dense.backprop( &output, &input, &output_delta, &mut input_delta, &mut grad );
        dense.grad_descent( 0.1, &grad );

        println!( "Before:");
        println!( "{:?}", input );
        println!( "{:?}", output );

        dense.inference( &input, &mut output );

        println!( "After:");
        println!( "{:?}", input );
        println!( "{:?}", output );
    }
}
