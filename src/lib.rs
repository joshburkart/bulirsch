//! Implementation of the Bulirsch-Stoer method for stepping ordinary differential equations.
//!
//! The [(Gragg-)Bulirsch-Stoer](https://en.wikipedia.org/wiki/Bulirsch%E2%80%93Stoer_algorithm)
//! algorithm combines the (modified) midpoint method with Richardson extrapolation to accelerate
//! convergence. It is an explicit method that does not require Jacobians.
//!
//! This crate's implementation, which follows ch. 17.3.2 of Numerical Recipes (Third Edition), does
//! not contain adaptive step size routines. It can be useful in situations where an ODE is being
//! integrated step by step with a prescribed step size that is not too large relative to the
//! dynamical timescale, for example in simulations of electromechanical control systems with a
//! fixed control cycle period. Each integration step is stateless, aside from the integration state
//! vector which the caller must maintain. Only time-independent ODEs are supported, but without
//! loss of generality (since the state vector can be augmented with a time variable if needed).
//!
//! As an example, consider a simple exponential system:
//!
//! ```
//! // Define ODE.
//! struct ExpSystem {}
//!
//! impl bulirsch::System for ExpSystem {
//!     type Float = f64;
//!
//!     fn system(
//!         &self,
//!         y: bulirsch::ArrayView1<Self::Float>,
//!         mut dydt: bulirsch::ArrayViewMut1<Self::Float>,
//!     ) {
//!         dydt.assign(&y);
//!     }
//! }
//!
//! let system = ExpSystem {};
//!
//! // Set up the integrator.
//! let integrator = bulirsch::Integrator::<ExpSystem>::default()
//!     .with_abs_tol(1e-4)
//!     .with_rel_tol(1e-4)
//!     .with_max_iterations(10)
//!     .with_step_size_policy(bulirsch::StepSizePolicy::Linear);
//!
//! // Define initial conditions and provide solution storage.
//! let t_final: f64 = 0.2;
//! let y = ndarray::array![1.];
//! let mut y_final = ndarray::Array::zeros([1]);
//!
//! // Integrate.
//! let stats = integrator
//!     .step(&system, t_final, y.view(), y_final.view_mut())
//!     .unwrap();
//!
//! // Ensure result matches analytic solution.
//! approx::assert_relative_eq!(
//!     t_final.exp(),
//!     y_final[[0]],
//!     epsilon = 1e-4,
//!     max_relative = 1e-4,
//! );
//!
//! // Check integration performance.
//! assert_eq!(stats.num_system_evals, 7);
//! assert_eq!(stats.num_iterations, 1);
//! assert_eq!(stats.num_substeps, 4);
//! approx::assert_relative_eq!(stats.substep_size, t_final / 4.);
//! assert!(stats.scaled_truncation_error < 1.);
//! ```
//!
//! Note that only a handful of system evaluations have been used. By contrast, the `ode_solvers`
//! crate uses several times more system evaluations for the same small timestep, in part because
//! its adaptive timestep routines need to be initialized:
//!
//! ```
//! struct ExpSystem {}
//!
//! impl ode_solvers::System<f64, ode_solvers::SVector<f64, 1>> for ExpSystem {
//!     fn system(
//!         &self,
//!         _x: f64,
//!         y: &ode_solvers::SVector<f64, 1>,
//!         dy: &mut ode_solvers::SVector<f64, 1>,
//!     ) {
//!         dy[0] = y[0];
//!     }
//! }
//!
//! let t_final = 0.2;
//! let system = ExpSystem {};
//! let mut solver = ode_solvers::Dop853::new(
//!     system,
//!     0.,
//!     t_final,
//!     t_final,
//!     ode_solvers::Vector1::new(1.),
//!     1e-4,
//!     1e-4,
//! );
//! let stats = solver.integrate().unwrap();
//! assert_eq!(stats.num_eval, 33);
//! ```

#![expect(
    non_snake_case,
    reason = "Used for math symbols to match notation in Numerical Recipes"
)]

pub use nd::ArrayView1;
pub use nd::ArrayViewMut1;
use ndarray as nd;
use num_traits::cast;

pub trait Float:
    num_traits::Float
    + core::iter::Sum
    + core::ops::AddAssign
    + core::ops::MulAssign
    + core::fmt::Debug
    + nd::ScalarOperand
{
}

impl Float for f32 {}
impl Float for f64 {}

/// Trait for defining an ordinary differential equation system.
pub trait System {
    /// The floating point type.
    type Float: Float;

    /// Evaluate the ordinary differential equation and store the derivative in `dydt`.
    fn system(&self, y: ArrayView1<Self::Float>, dydt: ArrayViewMut1<Self::Float>);
}

/// Statistics from taking an integration step.
#[must_use]
#[derive(Debug)]
pub struct Stats<F: Float> {
    /// The total number of ODE system evaluations used.
    pub num_system_evals: usize,

    /// The number of iterations used.
    pub num_iterations: usize,
    /// The number of substeps used on the final iteration.
    pub num_substeps: usize,
    /// The substep size on the final iteration.
    pub substep_size: F,

    /// The scaled (including absolute and relative tolerances) truncation error.
    ///
    /// Will be <= 1 if convergence was achieved, and > 1 if convergence was not achieved.
    pub scaled_truncation_error: F,
}

/// Error produced when the integration step failed to converge.
#[must_use]
#[derive(Debug)]
pub struct FailedToConverge<F: Float> {
    /// Statistics from the failed step.
    pub stats: Stats<F>,
}

/// Possible step size policies, which determine how the algorithm increases the number of substeps
/// on every iteration.
#[derive(Default)]
pub enum StepSizePolicy {
    /// Increase the number of substeps linearly with each subsequent iteration.
    ///
    /// This is appropriate if the step size is already relatively small, and few substeps are
    /// likely needed for convergence.
    #[default]
    Linear,
    /// Increase the number of substeps exponentially with each subsequent iteration.
    ///
    /// This is appropriate if the step size may be relatively large, so many substeps might be
    /// required for convergence.
    Exponential,
}

/// Possible extrapolation policies, which determine how the algorithm extrapolates.
///
/// This affects how convergence is assessed, as well as how the integration result is produced.
#[derive(Default)]
pub enum ConvergencePolicy {
    /// Use all previous iterations to extrapolate.
    ///
    /// This is appropriate for smooth system functions where the step size is relatively small.
    #[default]
    AllIterations,
    /// Use only the last `num_iterations` to extrapolate.
    ///
    /// This is appropriate for larger step sizes, or if the system function is not smooth, so that
    /// early iterations should be ignored when extrapolating. It can also be useful if NaNs or
    /// infinities are produced when the substep size is too large (for early iterations), so that
    /// they these non-finite numbers are eventually ignored.
    Window {
        /// The number of previous iterations' results to use for extrapolation.
        num_iterations: core::num::NonZero<usize>,
    },
}

impl ConvergencePolicy {
    fn get_extrap_pair<'a, F: Float>(&self, Tk: &'a [nd::Array1<F>]) -> &'a [nd::Array1<F>] {
        match self {
            Self::AllIterations => Tk.last_chunk::<2>().unwrap(),
            Self::Window { num_iterations } => Tk
                .get(num_iterations.get()..num_iterations.get() + 2)
                .unwrap_or_else(|| Tk.last_chunk::<2>().unwrap()),
        }
    }
}

/// An explicit ODE integrator using the Bulirsch-Stoer algorithm.
pub struct Integrator<S: System> {
    /// The absolute tolerance.
    abs_tol: S::Float,
    /// The relative tolerance.
    rel_tol: S::Float,

    /// The step size policy to use.
    step_size_policy: StepSizePolicy,
    /// The convergence policy to use.
    convergence_policy: ConvergencePolicy,
    /// The maximum number of iterations to use.
    max_iterations: usize,
}

impl<S: System> Default for Integrator<S>
where
    S::Float: Float,
{
    fn default() -> Self {
        Self {
            abs_tol: cast(1e-5).unwrap(),
            rel_tol: cast(1e-5).unwrap(),
            step_size_policy: StepSizePolicy::default(),
            convergence_policy: ConvergencePolicy::default(),
            max_iterations: 20,
        }
    }
}

impl<S: System> Integrator<S>
where
    S::Float: Float,
{
    /// Set the absolute tolerance.
    pub fn with_abs_tol(self, abs_tol: S::Float) -> Self {
        Self { abs_tol, ..self }
    }
    /// Set the relative tolerance.
    pub fn with_rel_tol(self, rel_tol: S::Float) -> Self {
        Self { rel_tol, ..self }
    }

    /// Set the step size policy.
    pub fn with_step_size_policy(self, step_size_policy: StepSizePolicy) -> Self {
        Self {
            step_size_policy,
            ..self
        }
    }
    /// Set the convergence policy.
    pub fn with_convergence_policy(self, convergence_policy: ConvergencePolicy) -> Self {
        Self {
            convergence_policy,
            ..self
        }
    }
    /// Set the maximum allowed number of iterations per step.
    pub fn with_max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// Take a step using the Bulirsch-Stoer method.
    ///
    /// # Arguments
    ///
    /// * `system`: The ODE system.
    /// * `delta_t`: The step size to take.
    /// * `y_init`: The initial state vector at the start of the step.
    /// * `y_final`: The vector into which to store the final computed state at the end of the step.
    ///
    /// # Result
    ///
    /// Stats providing information about integration performance, or an error if integration
    /// failed.
    ///
    /// # Examples
    ///
    /// Note that if you're using e.g. [`nalgebra`] to define your ODE, you can bridge to
    /// [`ndarray`] vectors using slices, as long as you're using [`nalgebra`]'s dynamically sized
    /// vectors. The same applies to using [`Vec`]s, etc. For example, consider a simple
    /// trigonometric system defined using [`nalgebra`]:
    ///
    /// ```
    /// // Define trigonometric ODE.
    /// #[derive(Clone, Copy)]
    /// struct TrigSystem {
    ///     omega: f32,
    /// }
    ///
    /// fn compute_dydt(
    ///     omega: f32,
    ///     y: nalgebra::DVectorView<f32>,
    ///     mut dydt: nalgebra::DVectorViewMut<f32>,
    /// ) {
    ///     dydt[0] = y[1];
    ///     dydt[1] = -omega.powi(2) * y[0];
    /// }
    ///
    /// impl bulirsch::System for TrigSystem {
    ///     type Float = f32;
    ///
    ///     fn system(
    ///         &self,
    ///         y: bulirsch::ArrayView1<Self::Float>,
    ///         mut dydt: bulirsch::ArrayViewMut1<Self::Float>,
    ///     ) {
    ///         let y_nalgebra = nalgebra::DVectorView::from_slice(
    ///             y.as_slice().unwrap(),
    ///             y.len(),
    ///         );
    ///         let dydt_nalgebra = nalgebra::DVectorViewMut::from_slice(
    ///             dydt.as_slice_mut().unwrap(),
    ///             y.len(),
    ///         );
    ///         compute_dydt(self.omega, y_nalgebra, dydt_nalgebra);
    ///     }
    /// }
    ///
    /// // Instantiate system and integrator.
    /// let system = TrigSystem { omega: 1.2 };
    /// let integrator =
    ///     bulirsch::Integrator::default()
    ///         .with_abs_tol(1e-6)
    ///         .with_rel_tol(0.)
    ///         .with_step_size_policy(bulirsch::StepSizePolicy::Exponential);
    ///
    /// // Define initial conditions and integrate.
    /// let y = ndarray::array![1., 0.];
    /// let mut y_next = ndarray::Array1::zeros(y.raw_dim());
    /// let t_final = 0.6;
    /// let stats = integrator
    ///     .step(&system, t_final, y.view(), y_next.view_mut())
    ///     .unwrap();
    ///
    /// // Check against analytic solution.
    /// let (sin, cos) = (t_final * system.omega).sin_cos();
    /// approx::assert_relative_eq!(y_next[0], cos, epsilon = 1e-2);
    /// approx::assert_relative_eq!(
    ///     y_next[1],
    ///     -system.omega * sin,
    ///     epsilon = 1e-2
    /// );
    ///
    /// // Check integrator performance.
    /// assert_eq!(stats.num_system_evals, 31);
    ///
    /// // Check against `ode_solvers`.
    /// impl ode_solvers::System<f32, ode_solvers::SVector<f32, 2>> for TrigSystem {
    ///     fn system(
    ///         &self,
    ///         _x: f32,
    ///         y: &ode_solvers::SVector<f32, 2>,
    ///         dy: &mut ode_solvers::SVector<f32, 2>,
    ///     ) {
    ///         <Self as bulirsch::System>::system(
    ///             self,
    ///             bulirsch::ArrayView1::from_shape([2], y.as_slice()).unwrap(),
    ///             bulirsch::ArrayViewMut1::from_shape([2], dy.as_mut_slice()).unwrap(),
    ///         );
    ///     }
    /// }
    ///
    /// let mut solver = ode_solvers::Dop853::new(
    ///     system,
    ///     0.,
    ///     t_final,
    ///     t_final,
    ///     ode_solvers::Vector2::new(1., 0.),
    ///     0.,
    ///     1e-6,
    /// );
    /// let ode_solvers_stats = solver.integrate().unwrap();
    /// assert_eq!(ode_solvers_stats.num_eval, 48);
    /// ```
    pub fn step(
        &self,
        system: &S,
        delta_t: S::Float,
        y_init: nd::ArrayView1<S::Float>,
        mut y_final: nd::ArrayViewMut1<S::Float>,
    ) -> Result<Stats<S::Float>, FailedToConverge<S::Float>> {
        let mut evaluation_counter = SystemEvaluationCounter {
            system,
            num_system_evals: 0,
        };

        let f_init = {
            let mut f_init = nd::Array1::zeros(y_init.raw_dim());
            evaluation_counter.system(y_init, f_init.view_mut());
            f_init
        };

        // Step size policy.
        let compute_n = match self.step_size_policy {
            StepSizePolicy::Linear => |k: usize| -> usize { 2 * (k + 1) },
            StepSizePolicy::Exponential => {
                |k: usize| -> usize { 2u32.pow((k + 1) as u32) as usize }
            }
        };

        // Build up integration tableau.
        let mut T = Vec::<Vec<nd::Array1<S::Float>>>::new();
        for k in 0..self.max_iterations {
            let nk = compute_n(k);
            let mut Tk = Vec::with_capacity(k + 1);
            Tk.push(self.midpoint_step(&mut evaluation_counter, delta_t, nk, &f_init, y_init));
            for j in 0..k {
                // There is a mistake in eq. 17.3.8. See
                // https://www.numerical.recipes/forumarchive/index.php/t-2256.html.
                let denominator = <S::Float as num_traits::Float>::powi(
                    cast::<_, S::Float>(nk).unwrap() / cast(compute_n(k - j - 1)).unwrap(),
                    2,
                ) - <S::Float as num_traits::One>::one();
                Tk.push(&Tk[j] + (&Tk[j] - &T[k - 1][j]) / denominator);
            }

            if k > 0 {
                let extrap_pair = self.convergence_policy.get_extrap_pair(&Tk);
                let scaled_truncation_error = compute_scaled_truncation_error(
                    extrap_pair[0].view(),
                    extrap_pair[1].view(),
                    self.abs_tol,
                    self.rel_tol,
                );
                if scaled_truncation_error <= <S::Float as num_traits::One>::one() {
                    y_final.assign(&extrap_pair[1]);
                    return Ok(Stats {
                        num_system_evals: evaluation_counter.num_system_evals,
                        num_iterations: k,
                        num_substeps: nk,
                        substep_size: delta_t / cast::<_, S::Float>(nk).unwrap(),
                        scaled_truncation_error,
                    });
                }
            }

            T.push(Tk);
        }

        // Failed to converge. Compute stats and return.
        let last_Tk = T.last().unwrap();
        let extrap_pair = self.convergence_policy.get_extrap_pair(last_Tk);
        y_final.assign(&extrap_pair[1]);
        let scaled_truncation_error = compute_scaled_truncation_error(
            extrap_pair[0].view(),
            extrap_pair[1].view(),
            self.abs_tol,
            self.rel_tol,
        );

        let n = compute_n(self.max_iterations);
        Err(FailedToConverge {
            stats: Stats {
                num_system_evals: evaluation_counter.num_system_evals,
                num_iterations: self.max_iterations,
                num_substeps: n,
                substep_size: delta_t / cast(n).unwrap(),
                scaled_truncation_error,
            },
        })
    }

    fn midpoint_step(
        &self,
        evaluation_counter: &mut SystemEvaluationCounter<S>,
        delta_t: S::Float,
        n: usize,
        f_init: &nd::Array1<S::Float>,
        y_init: nd::ArrayView1<S::Float>,
    ) -> nd::Array1<S::Float> {
        let substep_size = delta_t / cast(n).unwrap();
        let two_substep_size = cast::<_, S::Float>(2).unwrap() * substep_size;

        // 0    1    2    3    4    5    6    n
        //                  ..
        //           zi  zip1
        //           zip1 zi
        //                zi zip1
        //                  ..
        //                               zi  zip1
        let mut zi = y_init.to_owned();
        let mut zip1 = &zi + f_init * substep_size;
        let mut fi = f_init.clone();

        for _i in 1..n {
            core::mem::swap(&mut zi, &mut zip1);
            evaluation_counter.system(zi.view(), fi.view_mut());
            fi *= two_substep_size;
            zip1 += &fi;
        }

        evaluation_counter.system(zip1.view(), fi.view_mut());
        fi *= substep_size;
        let mut result = zi;
        result += &zip1;
        result += &fi;
        result *= cast::<_, S::Float>(0.5).unwrap();
        result
    }
}

fn compute_scaled_truncation_error<F: Float + core::iter::Sum>(
    y: nd::ArrayView1<F>,
    y_alt: nd::ArrayView1<F>,
    abs_tol: F,
    rel_tol: F,
) -> F {
    (y.iter()
        .zip(y_alt.iter())
        .map(|(&yi, &yi_alt)| {
            let scale = abs_tol + rel_tol * yi_alt.abs().max(yi.abs());
            (yi - yi_alt).powi(2) / scale.powi(2)
        })
        .sum::<F>()
        / cast(y.len()).unwrap())
    .sqrt()
}

struct SystemEvaluationCounter<'a, S: System> {
    system: &'a S,
    num_system_evals: usize,
}

impl<'a, S: System> SystemEvaluationCounter<'a, S> {
    fn system(&mut self, y: nd::ArrayView1<S::Float>, dydt: nd::ArrayViewMut1<S::Float>) {
        self.num_system_evals += 1;
        <S as System>::system(&self.system, y, dydt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_system_high_precision() {
        struct ExpSystem {}

        impl System for ExpSystem {
            type Float = f64;

            fn system(&self, y: ArrayView1<Self::Float>, mut dydt: ArrayViewMut1<Self::Float>) {
                dydt.assign(&y);
            }
        }

        let system = ExpSystem {};

        // Set up integrator with tolerance parameters.
        let integrator = Integrator::<ExpSystem>::default()
            .with_abs_tol(0.)
            .with_rel_tol(1e-14);

        // Define initial conditions and provide solution storage.
        let t_final = 0.2;
        let y = ndarray::array![1.];
        let mut y_final = ndarray::Array::zeros([1]);

        // Integrate.
        let stats = integrator
            .step(&system, t_final, y.view(), y_final.view_mut())
            .unwrap();

        // Ensure result matches analytic solution to high precision.
        approx::assert_relative_eq!(t_final.exp(), y_final[[0]], max_relative = 1e-14);

        // Check integration performance.
        assert_eq!(stats.num_system_evals, 43);
        assert_eq!(stats.num_iterations, 5);
        assert_eq!(stats.num_substeps, 12);
        approx::assert_relative_eq!(stats.substep_size, t_final / 12.);
        assert!(stats.scaled_truncation_error < 1.);
    }

    #[test]
    fn exp_system_handle_nans() {
        struct ExpSystem {
            hit_a_nan: core::cell::RefCell<bool>,
        }

        impl System for ExpSystem {
            type Float = f64;

            fn system(&self, y: ArrayView1<Self::Float>, mut dydt: ArrayViewMut1<Self::Float>) {
                if y[0].abs() > 10. {
                    *self.hit_a_nan.borrow_mut() = true;
                    dydt[0] = core::f64::NAN;
                } else {
                    dydt.assign(&(-&y));
                }
            }
        }

        let system = ExpSystem {
            hit_a_nan: false.into(),
        };

        // Set up integrator with tolerance parameters.
        let integrator = Integrator::<ExpSystem>::default()
            .with_abs_tol(0.)
            .with_rel_tol(1e-10)
            .with_step_size_policy(StepSizePolicy::Exponential)
            .with_convergence_policy(ConvergencePolicy::Window {
                num_iterations: core::num::NonZero::new(3).unwrap(),
            });

        // Define initial conditions and provide solution storage.
        let t_final = 5.;
        let y = ndarray::array![1.];
        let mut y_final = ndarray::Array::zeros([1]);

        // Integrate.
        let _stats = integrator
            .step(&system, t_final, y.view(), y_final.view_mut())
            .unwrap();

        // Ensure result matches analytic solution.
        approx::assert_relative_eq!((-t_final).exp(), y_final[[0]], max_relative = 1e-8);

        // Ensure we hit at least one NaN.
        assert!(*system.hit_a_nan.borrow());
    }
}
