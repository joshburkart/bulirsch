# Bulirsch-Stoer Method in Rust

This crate contains a simple implementation of the Bulirsch-Stoer method for solving ordinary
differential equations.

This crate may be useful in situations where an ODE is being integrated step by step with a
prescribed step size that is not too large relative to the dynamical timescale, for example in
cyber-physical system simulations with a fixed control cycle period. By contrast, routines from the
`ode_solvers` crate can require many more system evaluations to produce the same performance.
