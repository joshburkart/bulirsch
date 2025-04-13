Implementation of the Bulirsch-Stoer method for stepping ordinary differential equations.

The [(Gragg-)Bulirsch-Stoer](https://en.wikipedia.org/wiki/Bulirsch%E2%80%93Stoer_algorithm)
algorithm combines the (modified) midpoint method with Richardson extrapolation to accelerate
convergence. It is an explicit method that does not require Jacobians.

This crate's implementation contains a simplistic adaptive step size routine without order
estimation. Its API is designed to be useful in situations where an ODE is being integrated step by
step with a prescribed time step, for example in simulations of electromechanical control systems
with a fixed control cycle period. Only time-independent ODEs are supported, but without loss of
generality (since the state vector can be augmented with a time variable if needed).

The implementation follows:
* Press, William H. Numerical Recipes 3rd Edition: The Art of Scientific Computing. Cambridge
  University Press, 2007. Ch. 17.3.2.
* Deuflhard, Peter. "Order and stepsize control in extrapolation methods." Numerische Mathematik
  41 (1983): 399-422.
