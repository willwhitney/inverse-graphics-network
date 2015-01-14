reverse-graphics-network
========================

An implementation of (parts of) [Tijmen Tieleman's PhD thesis](http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf) in Theano.

## Optimization to do

- ensure all datatypes match (apparently Theano uses Python code for dot products of non-matching structures)
    + this happens in ACR.output_value_at
- reduce # of inc_subtensor commands in the ACR scan
    + use set_value
- rerun optimizer profile
- run more `train` iterations and see how performance scales

## done
- ifelse apparently runs in Python, but only computes one branch. faster for really big branches of execution, slower otherwise. should remove them.
