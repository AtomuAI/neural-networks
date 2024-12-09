// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut }
};

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::Layer;

#[derive( Clone, Default, Debug )]
pub struct NodeLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    nodes: Tensor<T, 5, M>,
}

impl<T, M> Deref for NodeLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Target = Tensor<T, 5, M>;

    fn deref( &self ) -> &Self::Target {
        &self.nodes
    }
}

impl<T, M> DerefMut for NodeLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.nodes
    }
}

impl<T, M> Layer<T, 5, M> for NodeLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 5, M>: TensorTraits<T, 5, M>
{
    fn new( shape: Shape<5> ) -> Self {
        Self {
            nodes: Tensor::<T, 5, M>::new( shape ),
        }
    }

    fn take( shape: Shape<5>, memory: <Memory<T, M> as MemoryTraits>::Take ) -> Self {
        Self {
            nodes: Tensor::<T, 5, M>::take( shape, memory ),
        }
    }

    fn inference<O, P>( &self, _: &NodeLayer<T, O>, _: &mut NodeLayer<T, P> )
    where
        O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        panic!( "Not implemented" );
    }

    fn backprop<O, P>( &self, _: &NodeLayer<T, P>, _: &mut NodeLayer<T, O>, _: &Tensor<T, 5, P>, _: &mut Tensor<T, 5, O>, _: &mut Tensor<T, 5, M> )
    where
    O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        panic!( "Not implemented" );
    }

    fn grad_descent( &mut self, _: &Tensor<T, 5, M> ) {
        panic!( "Not implemented" );
    }

    fn grad_descent_momentum( &mut self, _: &Tensor<T, 5, M>, _: &mut Tensor<T, 5, M> ) {
        panic!( "Not implemented" );
    }

    fn grad_descent_adam( &mut self, _: &Tensor<T, 5, M>, _: &mut Tensor<T, 5, M>, _: &mut Tensor<T, 5, M> ) {
        panic!( "Not implemented" );
    }
}
