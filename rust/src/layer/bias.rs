// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut, Add, Sub, Mul, Div, AddAssign }
};
use num::traits::{ Pow, FromPrimitive };

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::{
    Layer,
    Inference, InferenceAssign,
    Backprop, BackpropAssign,
    GradientDecent, GradientDescentMomentum, GradientDescentAdam,
    node::NodeLayer
};

#[derive( Clone, Default, Debug )]
pub struct BiasLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
{
    nodes: Tensor<T, 4, M>,
}

impl<T, M> BiasLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>
{
    pub fn new( shape: Shape<4>, memory: <Memory<T, M> as MemoryTraits>::New ) -> Self {
        Self {
            nodes: Tensor::<T, 4, M>::new( shape, memory ),
        }
    }

    pub fn iter( &self ) -> impl Iterator<Item = &T> {
        self.nodes.iter()
    }

    pub fn iter_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.nodes.iter_mut()
    }
}

impl<T, M> Deref for BiasLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Target = Tensor<T, 4, M>;

    fn deref( &self ) -> &Self::Target {
        &self.nodes
    }
}

impl<T, M> DerefMut for BiasLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.nodes
    }
}

impl<T, M> Layer for BiasLayer<T, M>
where
    T: Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{}

impl<T, M, O, P> Inference<NodeLayer<T, O>, NodeLayer<T, P>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Add<Output = T>,
    M: MemoryType,
    O: MemoryType,
    P: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
    Memory<T, P>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>,
    Tensor<T, 5, O>: TensorTraits<T, 5, O>,
    Tensor<T, 5, P>: TensorTraits<T, 5, P>
{
    fn inference( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> ) {
        self.iter().zip( input.iter() ).zip( output.iter_mut() )
            .for_each( |( ( &bias, &input ), output )| *output = bias + input );
    }
}

impl<T, M, O> InferenceAssign<NodeLayer<T, O>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + AddAssign,
    M: MemoryType,
    O: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>,
    Tensor<T, 5, O>: TensorTraits<T, 5, O>,
{
    fn inference_assign( &self, input: &mut NodeLayer<T, O> ) {
        self.iter().zip( input.iter_mut() )
            .for_each( |( &bias, input )| *input += bias );
    }
}

impl<T, M, P, O> Backprop<NodeLayer<T, P>, NodeLayer<T, O>, Tensor<T, 5, P>, Tensor<T, 5, O>, Tensor<T, 4, M>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + AddAssign,
    M: MemoryType,
    O: MemoryType,
    P: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
    Memory<T, P>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>,
    Tensor<T, 5, O>: TensorTraits<T, 5, O>,
    Tensor<T, 5, P>: TensorTraits<T, 5, P>
{
    fn backprop( &self, _: &NodeLayer<T, P>, _: &NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, grad: &mut Tensor<T, 4, M> ) {
        input_delta.iter_mut().zip( output_delta.iter() ).zip( grad.iter_mut() )
            .for_each( |( ( input_delta, &output_delta ), grad )| {
                *grad += output_delta;
                *input_delta = output_delta;
            });
    }
}

impl<T, M, P> BackpropAssign<NodeLayer<T, P>, Tensor<T, 5, P>, Tensor<T, 4, M>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + AddAssign,
    M: MemoryType,
    P: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, P>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>,
    Tensor<T, 5, P>: TensorTraits<T, 5, P>
{
    fn backprop_assign( &self, _: &NodeLayer<T, P>, output_delta: &mut Tensor<T, 5, P>, grad: &mut Tensor<T, 4, M> ) {
        output_delta.iter_mut().zip( grad.iter_mut() )
            .for_each( |( &mut output_delta, grad )| {
                *grad += output_delta;
            });
    }
}

impl<T, M> GradientDecent<T, Tensor<T, 4, M>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Mul<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>
{
    fn grad_descent( &mut self, step: T, grad: &Tensor<T, 4, M> ) {
        self.nodes.grad_descent( step, grad );
    }
}

impl<T, M> GradientDescentMomentum<T, Tensor<T, 4, M>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>
{
    fn grad_descent_momentum( &mut self, step: T, grad: &Tensor<T, 4, M>, momentum: &mut Tensor<T, 4, M> ) {
        self.nodes.grad_descent_momentum( step, grad, momentum );
    }
}

impl<T, Time, M> GradientDescentAdam<T, Time, Tensor<T, 4, M>> for BiasLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + AddAssign + Pow<T, Output = T> + Pow<Time, Output = T> + Pow<i32, Output = T> + Pow<f32, Output = T> + FromPrimitive,
    Time: Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>
{
    fn grad_descent_adam( &mut self, step: T, time: Time, beta_m: T, beta_v: T, epsilon: T, grad: &Tensor<T, 4, M>, momentum: &mut Tensor<T, 4, M>, velocity: &mut Tensor<T, 4, M> ) {
        self.nodes.grad_descent_adam( step, time, beta_m, beta_v, epsilon, grad, momentum, velocity );
    }
}

#[cfg(test)]
mod tests {
use super::*;

    #[test]
    fn test_inference() {
        let input = NodeLayer::<f32, Heap>::new(
            [1, 5, 1, 1, 1].into(),
            [1.0; 5].into()
        );

        let mut output = NodeLayer::<f32, Heap>::new(
            [1, 5, 1, 1, 1].into(),
            [1.0; 5].into()
        );

        let mut bias = BiasLayer::<f32, Heap>::new(
            [1, 5, 1, 1].into(),
            [1.0; 5].into()
        );

        let mut input_delta = Tensor::<f32, 5, Heap>::new(
            [ 1, 5, 1, 1, 1 ].into(),
            [ 0.0; 5 ].into()
        );

        let output_delta = Tensor::<f32, 5, Heap>::new(
            [ 1, 5, 1, 1, 1 ].into(),
            [ 0.1; 5 ].into()
        );

        let mut grad = Tensor::<f32, 4, Heap>::new(
            [ 1, 5, 1, 1 ].into(),
            [ 0.0; 5 ].into()
        );

        bias.inference( &input, &mut output );

        println!( "Before:");
        println!( "{:?}", input );
        println!( "{:?}", output );

        bias.backprop( &output, &input, &output_delta, &mut input_delta, &mut grad );
        bias.grad_descent( 0.1, &grad );

        bias.inference( &input, &mut output );

        println!( "After:");
        println!( "{:?}", input );
        println!( "{:?}", output );


    }
}
