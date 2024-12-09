// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut }
};

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::{
    Layer,
    node::NodeLayer
};

#[derive( Clone, Default, Debug )]
pub struct SoftmaxLayer;

impl<T> Layer<T, 0, Stack<0>> for SoftmaxLayer
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
{
    fn new( _: Shape<0> ) -> Self {
        Self {}
    }

    fn take( _: Shape<0>, _: [T; 0] ) -> Self {
        Self {}
    }

    fn inference<O, P>( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> )
    where
        T: 'static + Default + Debug + Clone + Copy + PartialEq,
        O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        let _ = input;
        let _ = output;
    }

    fn backprop<O, P>( &self, output: &NodeLayer<T, P>, input: &mut NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, _: &mut Tensor<T, 0, Stack<0>> )
    where
        T: 'static + Default + Debug + Clone + Copy + PartialEq,
        O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        let _ = output;
        let _ = input;
        let _ = output_delta;
        let _ = input_delta;
    }

    fn grad_descent( &mut self, _: &Tensor<T, 0, Stack<0>> ) {
        panic!( "Not implemented" );
    }

    fn grad_descent_momentum( &mut self, _: &Tensor<T, 0, Stack<0>>, _: &mut Tensor<T, 0, Stack<0>> ) {
        panic!( "Not implemented" );
    }

    fn grad_descent_adam( &mut self, _: &Tensor<T, 0, Stack<0>>, _: &mut Tensor<T, 0, Stack<0>>, _: &mut Tensor<T, 0, Stack<0>> ) {
        panic!( "Not implemented" );
    }
}
