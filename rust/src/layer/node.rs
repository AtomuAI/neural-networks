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

impl<T, M> NodeLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 5, M>: TensorTraits<T, 5, M>
{
    pub fn new( shape: Shape<5>, memory: <Memory<T, M> as MemoryTraits>::New ) -> Self {
        Self {
            nodes: Tensor::<T, 5, M>::new( shape, memory ),
        }
    }

    pub fn iter( &self ) -> impl Iterator<Item = &T> {
        self.nodes.iter()
    }

    pub fn iter_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.nodes.iter_mut()
    }
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

impl<T, M> Layer for NodeLayer<T, M>
where
    T: Default + Debug + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{}
