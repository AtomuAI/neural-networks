// Copyright 2024 Bewusstsein Labs

use std::fmt::Debug;

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::node::NodeLayer;

pub mod activation;
pub mod bias;
pub mod convolution;
pub mod cost;
pub mod dense;
pub mod dropout;
pub mod mask;
pub mod node;
pub mod normalization;
pub mod pooling;
pub mod softmax;

pub trait Layer<T, const N: usize, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn new( shape: Shape<N> ) -> Self;
    fn take( shape: Shape<N>, memory: <Memory<T, M> as MemoryTraits>::Take ) -> Self;

    fn inference<O, P>( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> )
    where
        O: MemoryType,
        P: MemoryType,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>,
        Tensor<T, 5, O>: TensorTraits<T, 5, O>,
        Tensor<T, 5, P>: TensorTraits<T, 5, P>;

    fn backprop<O, P>( &self, output: &NodeLayer<T, P>, input: &mut NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, grad: &mut Tensor<T, N, M> )
    where
        O: MemoryType,
        P: MemoryType,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>,
        Tensor<T, 5, O>: TensorTraits<T, 5, O>,
        Tensor<T, 5, P>: TensorTraits<T, 5, P>;

    fn grad_descent( &mut self, grad: &Tensor<T, N, M> );
    fn grad_descent_momentum( &mut self, grad: &Tensor<T, N, M>, momentum: &mut Tensor<T, N, M> );
    fn grad_descent_adam( &mut self, grad: &Tensor<T, N, M>, momentum: &mut Tensor<T, N, M>, velocity: &mut Tensor<T, N, M> );
}
