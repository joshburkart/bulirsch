//! Implementation of the Bulirsch-Stoer method for stepping ordinary differential equations.
//!
//! The [(Gragg-)Bulirsch-Stoer](https://en.wikipedia.org/wiki/Bulirsch%E2%80%93Stoer_algorithm)
//! algorithm combines the (modified) midpoint method with Richardson extrapolation to accelerate
//! convergence. It is an explicit method that does not require Jacobians.
//!
//! This crate's implementation contains a simplistic adaptive step size routine without order
//! estimation. Its API is designed to be useful in situations where an ODE is being integrated step
//! by step with a prescribed time step, for example in simulations of electromechanical control
//! systems with a fixed control cycle period. Only time-independent ODEs are supported, but without
//! loss of generality (since the state vector can be augmented with a time variable if needed).
//!
//! The implementation follows:
//! * Press, William H. Numerical Recipes 3rd Edition: The Art of Scientific Computing. Cambridge
//!   University Press, 2007. Ch. 17.3.2.
//! * Deuflhard, Peter. "Order and stepsize control in extrapolation methods." Numerische Mathematik
//!   41 (1983): 399-422.
//!
//! As an example, consider a simple trigonometric system:
//!
//! ```
//! // Define ODE.
//! struct TrigSystem {
//!     omega: f64,
//! }
//!
//! impl bulirsch::System for TrigSystem {
//!     type Float = f64;
//!
//!     fn system(
//!         &self,
//!         y: bulirsch::ArrayView1<Self::Float>,
//!         mut dydt: bulirsch::ArrayViewMut1<Self::Float>,
//!     ) {
//!         dydt[[0]] = y[[1]];
//!         dydt[[1]] = -self.omega.powi(2) * y[[0]];
//!     }
//! }
//!
//! let system = TrigSystem { omega: 1.2 };
//!
//! // Set up the integrator. In a future version, we hope to automatically estimate the target
//! // number of iterations online, but for now it has to be set manually, and should be tuned to
//! // minimize the number of function evaluations for a given system. Here using 7 target
//! // iterations instead of the default of 3 decreases the number of function evaluations by ~35%
//! // (see below).
//! let mut integrator = bulirsch::Integrator::default()
//!     .with_abs_tol(1e-8)
//!     .with_rel_tol(1e-8)
//!     .into_adaptive()
//!     .with_target_num_iterations(7);
//!
//! // Define initial conditions and provide solution storage.
//! let delta_t: f64 = 10.2;
//! let mut y = ndarray::array![1., 0.];
//! let mut y_next = ndarray::Array::zeros(y.raw_dim());
//!
//! // Integrate for 10 steps.
//! let num_steps = 10;
//! for _ in 0..num_steps {
//!     integrator
//!         .step(&system, delta_t, y.view(), y_next.view_mut())
//!         .unwrap();
//!     y.assign(&y_next);
//! }
//!
//! // Ensure result matches analytic solution.
//! approx::assert_relative_eq!(
//!     (system.omega * delta_t * num_steps as f64).cos(),
//!     y_next[[0]],
//!     epsilon = 5e-7,
//!     max_relative = 5e-7,
//! );
//!
//! // Check integration performance.
//! assert_eq!(integrator.overall_stats().num_system_evals, 3862);
//! approx::assert_relative_eq!(integrator.step_size().unwrap(), 2.74, epsilon = 1e-2);
//! ```
//!
//! Note that 3.9k system evaluations have been used. By contrast, the `ode_solvers::Dopri5`
//! algorithm uses more:
//!
//! ```
//! struct TrigSystem {
//!     omega: f64,
//! }
//!
//! impl ode_solvers::System<f64, ode_solvers::SVector<f64, 2>> for TrigSystem {
//!     fn system(
//!         &self,
//!         _x: f64,
//!         y: &ode_solvers::SVector<f64, 2>,
//!         dy: &mut ode_solvers::SVector<f64, 2>,
//!     ) {
//!         dy[0] = y[1];
//!         dy[1] = -self.omega.powi(2) * y[0];
//!     }
//! }
//!
//! let omega = 1.2;
//! let delta_t: f64 = 10.2;
//! let mut num_system_eval = 0;
//! let mut y = ode_solvers::Vector2::new(1., 0.);
//! let num_steps = 10;
//! for _ in 0..num_steps {
//!     let system = TrigSystem { omega };
//!     let mut solver = ode_solvers::Dopri5::new(
//!         system,
//!         0.,
//!         delta_t,
//!         delta_t,
//!         y,
//!         1e-8,
//!         1e-8,
//!     );
//!     num_system_eval += solver.integrate().unwrap().num_eval;
//!     y = *solver.y_out().get(1).unwrap();
//! }
//! assert_eq!(num_system_eval, 7476);
//!
//! // Ensure result matches analytic solution.
//! approx::assert_relative_eq!(
//!     (omega * delta_t * num_steps as f64).cos(),
//!     y[0],
//!     epsilon = 5e-7,
//!     max_relative = 5e-7,
//! );
//! ```
//!
//! As of writing this, the latest version of `ode_solvers`, 0.6.1, also gives a dramatically
//! incorrect answer likely due to a regression. As a result we use version 0.5 as a dev dependency.

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

/// Error produced when integration produced a step size smaller than the minimum allowed step size.
#[derive(Debug)]
pub struct StepSizeUnderflow<F: Float>(F);

/// Statistics from taking an integration step.
#[derive(Clone, Debug)]
pub struct Stats {
    /// Number of system function evaluations.
    pub num_system_evals: usize,
}

#[derive(Clone)]
pub struct AdaptiveStepSizeIntegrator<F: Float> {
    /// The underlying non-adaptive integrator.
    integrator: Integrator<F>,

    /// The current step size.
    step_size: Option<F>,
    /// The minimum step size to allow before returning [`StepSizeUnderflow`].
    minimum_step_size: F,

    /// The target number of iterations to use. Ideally this would be estimated using order control,
    /// but is instead an input parameter for simplicity.
    target_num_iterations: usize,

    /// Overall stats
    overall_stats: Stats,
}

impl<F: Float> AdaptiveStepSizeIntegrator<F> {
    /// Take a step using the Bulirsch-Stoer method.
    ///
    /// # Arguments
    ///
    /// * `system`: The ODE system.
    /// * `delta_t`: The size of the prescribed time step to take.
    /// * `y_init`: The initial state vector at the start of the time step.
    /// * `y_final`: The vector into which to store the final computed state at the end of the time
    ///   step.
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
    /// let mut integrator =
    ///     bulirsch::Integrator::default()
    ///         .with_abs_tol(1e-6)
    ///         .with_rel_tol(0.)
    ///         .into_adaptive()
    ///         .with_target_num_iterations(3);
    ///
    /// // Define initial conditions and integrate.
    /// let mut y = ndarray::array![1., 0.];
    /// let mut y_next = ndarray::Array1::zeros(y.raw_dim());
    /// let delta_t = 0.6;
    /// let num_steps = 10;
    /// let mut num_system_evals = 0;
    /// for _ in 0..num_steps {
    ///     num_system_evals += integrator
    ///         .step(&system, delta_t, y.view(), y_next.view_mut())
    ///         .unwrap()
    ///         .num_system_evals;
    ///     y.assign(&y_next);
    /// }
    ///
    /// // Check against analytic solution.
    /// let (sin, cos) = (delta_t * num_steps as f32 * system.omega).sin_cos();
    /// approx::assert_relative_eq!(y_next[0], cos, epsilon = 1e-2);
    /// approx::assert_relative_eq!(
    ///     y_next[1],
    ///     -system.omega * sin,
    ///     epsilon = 1e-2
    /// );
    ///
    /// // Check integrator performance.
    /// assert_eq!(num_system_evals, 310);
    /// ```
    pub fn step<S: System<Float = F>>(
        &mut self,
        system: &S,
        delta_t: S::Float,
        y_init: nd::ArrayView1<S::Float>,
        mut y_final: nd::ArrayViewMut1<S::Float>,
    ) -> Result<Stats, StepSizeUnderflow<F>> {
        let mut step_size = if let Some(step_size) = self.step_size {
            step_size
        } else {
            delta_t
        };

        let mut system = SystemEvaluationCounter {
            system,
            num_system_evals: 0,
        };

        // Iteratively take steps until taking a step would put us past the input `delta_t`. At that
        // point, take an exact step to finish `delta_t`. Dynamically adjust the step size to
        // control truncation error as we go.
        let mut y_before_step = y_init.to_owned();
        let mut y_after_step = y_init.to_owned();
        let mut t = F::zero();
        loop {
            if step_size < self.minimum_step_size || !step_size.is_finite() {
                return Err(StepSizeUnderflow(step_size));
            }

            // `next_t` is `None` if we're at the tail end of `delta_t` and are taking a smaller
            // step than is optimal so we don't overshoot.
            let next_t = if t < delta_t - step_size {
                Some((t + step_size).min(delta_t))
            } else {
                None
            };

            let step_result = self.integrator.step(
                &mut system,
                step_size.min(delta_t - t),
                y_before_step.view(),
                y_after_step.view_mut(),
            );

            let adjustment_factor = match &step_result {
                Ok(stats) | Err(stats) => {
                    let scaled_truncation_error = *stats
                        .scaled_truncation_errors
                        .get(self.target_num_iterations)
                        .unwrap();
                    self.compute_step_size_adjustment_factor(scaled_truncation_error)
                }
            };
            match (step_result.is_ok(), next_t) {
                // The step was successful, and we're at the end of `delta_t`. Done. Don't adjust
                // the step size since we took a smaller step than was possible, so our truncation
                // error is not useful for step size adjustment.
                (true, None) => {
                    break;
                }
                // The step was successful, and we're not at the end of `delta_t`. Adjust step size
                // and continue.
                (true, Some(next_t)) => {
                    t = next_t;
                    step_size *= adjustment_factor;
                    y_before_step.assign(&y_after_step);
                }
                // The step failed. Adjust step size and try again.
                (false, _) => {
                    step_size *= adjustment_factor;
                }
            }
        }

        self.step_size = Some(step_size);
        y_final.assign(&y_after_step);
        self.overall_stats.num_system_evals += system.num_system_evals;

        Ok(Stats {
            num_system_evals: system.num_system_evals,
        })
    }

    /// Set the target number of iterations.
    ///
    /// A step size will be continuously attempted to be produced so that convergence is achieved at
    /// this number of iterations. Ideally this would be estimated automatically to minimize
    /// computational cost, but it is instead currently an input parameter for simplicity.
    pub fn with_target_num_iterations(self, target_num_iterations: usize) -> Self {
        assert!(target_num_iterations < self.integrator.max_iterations);
        Self {
            target_num_iterations,
            integrator: Integrator {
                min_iterations: target_num_iterations,
                ..self.integrator
            },
            ..self
        }
    }

    /// Set the minimum step size to allow before returning [`StepSizeUnderflow`].
    pub fn with_minimum_step_size(self, minimum_step_size: F) -> Self {
        Self {
            minimum_step_size,
            ..self
        }
    }

    /// Get the overall stats across all steps taken so far.
    pub fn overall_stats(&self) -> &Stats {
        &self.overall_stats
    }
    /// Get the current step size.
    pub fn step_size(&self) -> Option<F> {
        self.step_size
    }

    fn compute_step_size_adjustment_factor(&self, scaled_truncation_error: F) -> F {
        let safety_factor: F = cast(0.95).unwrap();
        let min_step_size_decrease_factor: F = cast(0.01).unwrap();
        let max_step_size_increase_factor = min_step_size_decrease_factor.recip();

        if scaled_truncation_error > F::zero() {
            // Eq. 2.14, Deuflhard, Peter. "Order and stepsize control in
            // extrapolation methods." Numerische Mathematik 41 (1983): 399-422.
            (safety_factor
                / scaled_truncation_error
                    .powf(F::one() / cast(2 * self.target_num_iterations + 1).unwrap()))
            .max(min_step_size_decrease_factor)
            .min(max_step_size_increase_factor)
        } else if scaled_truncation_error.is_finite() {
            F::one()
        } else {
            cast(0.5).unwrap()
        }
    }
}

/// An ODE integrator using the Bulirsch-Stoer algorithm with a fixed step size.
#[derive(Clone)]
pub struct Integrator<F: Float> {
    /// The absolute tolerance.
    abs_tol: F,
    /// The relative tolerance.
    rel_tol: F,

    /// The minimum number of iterations to use.
    min_iterations: usize,
    /// The maximum number of iterations to use.
    max_iterations: usize,
}

impl<F: Float> Default for Integrator<F> {
    fn default() -> Self {
        Self {
            abs_tol: cast(1e-5).unwrap(),
            rel_tol: cast(1e-5).unwrap(),
            min_iterations: 2,
            max_iterations: 10,
        }
    }
}

impl<F: Float> Integrator<F> {
    /// Make an [`AdaptiveStepSizeIntegrator`].
    pub fn into_adaptive(self) -> AdaptiveStepSizeIntegrator<F> {
        let target_num_iterations = 3;
        AdaptiveStepSizeIntegrator {
            integrator: Self {
                min_iterations: target_num_iterations,
                ..self
            },
            step_size: None,
            minimum_step_size: cast(1e-6).unwrap(),
            target_num_iterations,
            overall_stats: Stats {
                num_system_evals: 0,
            },
        }
    }

    /// Set the absolute tolerance.
    pub fn with_abs_tol(self, abs_tol: F) -> Self {
        Self { abs_tol, ..self }
    }
    /// Set the relative tolerance.
    pub fn with_rel_tol(self, rel_tol: F) -> Self {
        Self { rel_tol, ..self }
    }

    /// Set the maximum allowed number of iterations per step.
    pub fn with_max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// Take a single extrapolating step, iteratively subdividing in order to extrapolate.
    fn step<S: System<Float = F>>(
        &self,
        system: &mut SystemEvaluationCounter<S>,
        step_size: F,
        y_init: nd::ArrayView1<F>,
        mut y_final: nd::ArrayViewMut1<F>,
    ) -> Result<ExtrapolationStats<F>, ExtrapolationStats<F>> {
        let f_init = {
            let mut f_init = nd::Array1::zeros(y_init.raw_dim());
            system.system(y_init, f_init.view_mut());
            f_init
        };

        // Step size policy.
        let compute_n = |k: usize| -> usize { 2 * (k + 1) };

        // Build up an extrapolation tableau.
        let mut tableau = ExtrapolationTableau(Vec::<ExtrapolationTableauRow<_>>::new());
        for k in 0..self.max_iterations {
            let nk = compute_n(k);
            let tableau_row = {
                let mut Tk = Vec::with_capacity(k + 1);
                Tk.push(self.midpoint_step(system, step_size, nk, &f_init, y_init));
                for j in 0..k {
                    // There is a mistake in eq. 17.3.8. See
                    // https://www.numerical.recipes/forumarchive/index.php/t-2256.html.
                    let denominator = <F as num_traits::Float>::powi(
                        cast::<_, F>(nk).unwrap() / cast(compute_n(k - j - 1)).unwrap(),
                        2,
                    ) - <F as num_traits::One>::one();
                    Tk.push(&Tk[j] + (&Tk[j] - &tableau.0[k - 1].0[j]) / denominator);
                }
                ExtrapolationTableauRow(Tk)
            };
            tableau.0.push(tableau_row);

            if k > 0 {
                let scaled_truncation_error = tableau
                    .0
                    .last()
                    .unwrap()
                    .compute_scaled_truncation_error(self.abs_tol, self.rel_tol);
                if k > self.min_iterations
                    && scaled_truncation_error <= <F as num_traits::One>::one()
                {
                    y_final.assign(&tableau.0.last().unwrap().estimate());
                    return Ok(ExtrapolationStats {
                        scaled_truncation_errors: tableau
                            .compute_scaled_truncation_errors(self.abs_tol, self.rel_tol),
                    });
                }
            }
        }

        // Failed to converge.
        Err(ExtrapolationStats {
            scaled_truncation_errors: tableau
                .compute_scaled_truncation_errors(self.abs_tol, self.rel_tol),
        })
    }

    fn midpoint_step<S: System<Float = F>>(
        &self,
        evaluation_counter: &mut SystemEvaluationCounter<S>,
        step_size: F,
        n: usize,
        f_init: &nd::Array1<F>,
        y_init: nd::ArrayView1<F>,
    ) -> nd::Array1<F> {
        let substep_size = step_size / cast(n).unwrap();
        let two_substep_size = cast::<_, F>(2).unwrap() * substep_size;

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
        result *= cast::<_, F>(0.5).unwrap();
        result
    }
}

/// Statistics from taking an integration step.
#[derive(Debug)]
struct ExtrapolationStats<F: Float> {
    /// The scaled (including absolute and relative tolerances) truncation errors for each
    /// iteration.
    ///
    /// Each will be <= 1 if convergence was achieved or > 1 if convergence was not achieved.
    scaled_truncation_errors: Vec<F>,
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

struct ExtrapolationTableau<F: Float>(Vec<ExtrapolationTableauRow<F>>);

impl<F: Float> ExtrapolationTableau<F> {
    fn compute_scaled_truncation_errors(&self, abs_tol: F, rel_tol: F) -> Vec<F> {
        self.0
            .iter()
            .skip(1)
            .map(|row| row.compute_scaled_truncation_error(abs_tol, rel_tol))
            .collect()
    }
}

struct ExtrapolationTableauRow<F: Float>(Vec<nd::Array1<F>>);

impl<F: Float> ExtrapolationTableauRow<F> {
    fn compute_scaled_truncation_error(&self, abs_tol: F, rel_tol: F) -> F {
        let extrap_pair = self.0.last_chunk::<2>().unwrap();
        let y = &extrap_pair[0];
        let y_alt = &extrap_pair[1];
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

    fn estimate(&self) -> &nd::Array1<F> {
        self.0.last().unwrap()
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
        let mut integrator = Integrator::default()
            .with_abs_tol(0.)
            .with_rel_tol(1e-14)
            .into_adaptive()
            .with_target_num_iterations(4);

        // Define initial conditions and provide solution storage.
        let t_final = 3.5;
        let y = ndarray::array![1.];
        let mut y_final = ndarray::Array::zeros([1]);

        // Integrate.
        let stats = integrator
            .step(&system, t_final, y.view(), y_final.view_mut())
            .unwrap();

        // Ensure result matches analytic solution to high precision.
        approx::assert_relative_eq!(t_final.exp(), y_final[[0]], max_relative = 5e-13);

        // Check integration performance.
        assert_eq!(stats.num_system_evals, 541);
        approx::assert_relative_eq!(integrator.step_size().unwrap(), 0.385, epsilon = 1e-3);
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
        let mut integrator = Integrator::default()
            .with_abs_tol(0.)
            .with_rel_tol(1e-10)
            .into_adaptive()
            .with_target_num_iterations(4);

        // Define initial conditions and provide solution storage.
        let t_final = 20.;
        let y = ndarray::array![1.];
        let mut y_final = ndarray::Array::zeros([1]);

        // Integrate.
        let stats = integrator
            .step(&system, t_final, y.view(), y_final.view_mut())
            .unwrap();

        // Ensure result matches analytic solution.
        approx::assert_relative_eq!((-t_final).exp(), y_final[[0]], max_relative = 1e-8);

        // Ensure we hit at least one NaN.
        assert!(*system.hit_a_nan.borrow());

        assert_eq!(stats.num_system_evals, 1404);
    }
}
