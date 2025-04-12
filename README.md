# Bulirsch-Stoer Method in Rust

[![Crates.io](https://img.shields.io/crates/v/bulirsch.svg)](https://crates.io/crates/bulirsch)
[![Docs](https://docs.rs/bulirsch/badge.svg)](https://docs.rs/bulirsch)

Implementation of the Bulirsch-Stoer method for stepping ordinary differential equations.

The [(Gragg-)Bulirsch-Stoer](https://en.wikipedia.org/wiki/Bulirsch%E2%80%93Stoer_algorithm)
algorithm combines the (modified) midpoint method with Richardson extrapolation to accelerate
convergence. It is an explicit method that does not require Jacobians.

This crate's implementation, which follows ch. 17.3.2 of Numerical Recipes (Third Edition), does not
contain adaptive step size routines. It can be useful in situations where an ODE is being integrated
step by step with a prescribed step size that is not too large relative to the dynamical timescale,
for example in simulations of electromechanical control systems with a fixed control cycle period.
Each integration step is stateless, aside from the integration state vector which the caller must
maintain. Only time-independent ODEs are supported, but without loss of generality (since the state
vector can be augmented with a time variable if needed).
