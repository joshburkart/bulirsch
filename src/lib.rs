//! Implementation of the Bulirsch-Stoer method for solving ordinary differential equations
//!
//! This crate may be useful in situations where an ODE is being integrated step by step with a
//! prescribed step size that is not too large relative to the dynamical timescale, for example in
//! cyber-physical system simulations with a fixed control cycle period. Each integration step is
//! stateless, aside from the integration vector which the caller must maintain.
//!
//! As an example, consider a simple exponential system:
//!
//! ```
//! use ndarray as nd;
//!
//! struct ExpSystem {}
//!
//! impl bulirsch::System for ExpSystem {
//!     type Float = f64;
//!
//!     fn system<
//!         S: nd::RawDataMut<Elem = Self::Float> + nd::Data + nd::DataMut,
//!     >(
//!         &self,
//!         y: nd::ArrayView1<Self::Float>,
//!         mut dydt: nd::ArrayBase<S, nd::Ix1>,
//!     ) {
//!         dydt.assign(&y);
//!     }
//! }
//!
//! let system = ExpSystem {};
//! let integrator = bulirsch::Integrator::<ExpSystem>::default()
//!     .with_abs_tol(1e-4)
//!     .with_rel_tol(1e-4);
//!
//! let t_final = 0.2;
//! let y = nd::array![1.];
//! let mut y_final = nd::Array::zeros([1]);
//! let stats = integrator
//!     .step(&system, t_final, y.view(), y_final.view_mut())
//!     .unwrap();
//! approx::assert_relative_eq!(
//!     t_final.exp(),
//!     y_final[[0]],
//!     epsilon = 1e-4,
//!     max_relative = 1e-4,
//! );
//! assert_eq!(stats.num_system_evals, 7);
//! assert_eq!(stats.num_iterations, 1);
//! assert_eq!(stats.num_midpoint_substeps, 4);
//! approx::assert_relative_eq!(stats.midpoint_substep_size, t_final / 4.);
//! assert!(stats.scaled_truncation_error < 1.);
//! ```
//!
//! Note that only a handful of system evaluations have been used. By contrast, the `ode_solvers`
//! crate uses several times more system evaluations for the same small timestep, in part because
//! the adaptive timestep routines need to be initialized:
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
//!
//! The implementation follows ch. 17.3.2 of Numerical Recipes (Third Edition).

#![expect(
    non_snake_case,
    reason = "Used for math symbols to match notation in Numerical Recipes"
)]

pub use nd::ArrayView1;
pub use nd::ArrayViewMut1;
use ndarray as nd;
use num_traits::cast;

pub trait Float:
    num_traits::Float + core::iter::Sum + core::ops::AddAssign
{
}

impl Float for f32 {}
impl Float for f64 {}

pub trait System {
    type Float: Float;

    fn system<
        Storage: nd::RawDataMut<Elem = Self::Float> + nd::Data + nd::DataMut,
    >(
        &self,
        y: nd::ArrayView1<Self::Float>,
        dydt: nd::ArrayBase<Storage, nd::Ix1>,
    );
}

/// Statistics from taking an integration step
#[must_use]
#[derive(Debug)]
pub struct Stats<F: Float> {
    /// The total number of ODE system evaluations used to achieve convergence
    pub num_system_evals: usize,

    /// The number of iterations used when convergence was achieved
    pub num_iterations: usize,
    /// The number of midpoint substeps used when convergence was achieved
    pub num_midpoint_substeps: usize,

    /// The substep size when convergence was achieved
    pub midpoint_substep_size: F,

    /// The scaled (including absolute and relative tolerances) truncation error
    ///
    /// Will be <= 1 if convergence was achieved, and > 1 if convergence was not achieved.
    pub scaled_truncation_error: F,
}

/// The integration step failed to converge
#[must_use]
#[derive(Debug)]
pub struct FailedToConverge<F: Float> {
    /// Statistics from the failed step
    pub stats: Stats<F>,
}

/// An explicit ODE integrator using the Bulirsch-Stoer algorithm.
pub struct Integrator<S: System> {
    /// The absolute tolerance
    abs_tol: S::Float,
    /// The relative tolerance
    rel_tol: S::Float,

    /// The maximum number of iterations to use
    max_iterations: usize,
}

impl<S: System> Default for Integrator<S>
where
    S::Float: Float + nd::ScalarOperand,
{
    fn default() -> Self {
        Self {
            abs_tol: cast(1e-5).unwrap(),
            rel_tol: cast(1e-5).unwrap(),
            max_iterations: 20,
        }
    }
}

impl<S: System> Integrator<S>
where
    S::Float: Float + nd::ScalarOperand,
{
    /// Set the absolute tolerance
    pub fn with_abs_tol(self, abs_tol: S::Float) -> Self {
        Self { abs_tol, ..self }
    }
    /// Set the relative tolerance
    pub fn with_rel_tol(self, rel_tol: S::Float) -> Self {
        Self { rel_tol, ..self }
    }

    /// Set the maximum number of iterations per step
    pub fn with_max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// Take a step using the Bulirsch-Stoer method
    ///
    /// # Arguments
    ///
    /// * `system`: The ODE system.
    /// * `delta_t`: The step size to take.
    /// * `y_init`: The initial state vector at the start of the step. Note that if
    /// * `y_final`: The vector into which to store the final computed state at the end of the step.
    ///
    /// Note that if you're using e.g. `nalgebra`, you can bridge to `ndarray` vectors using slices.
    /// See the `test_trig` test.
    ///
    /// # Result
    ///
    /// Stats providing information about integration performance, or an error if integration
    /// failed.
    pub fn step(
        &self,
        system: &S,
        delta_t: S::Float,
        y_init: nd::ArrayView1<S::Float>,
        mut y_final: nd::ArrayViewMut1<S::Float>,
    ) -> Result<Stats<S::Float>, FailedToConverge<S::Float>> {
        let mut evaluation_counter = EvaluationCounter {
            system,
            num_system_evals: 0,
        };

        let f_init = {
            let mut f_init = nd::Array1::zeros(y_init.raw_dim());
            evaluation_counter.system(y_init, f_init.view_mut());
            f_init
        };

        // Step size policy.
        let compute_n = |k: usize| -> usize { 2 * (k + 1) };

        // Build up integration tableau.
        let mut T = Vec::<Vec<nd::Array1<S::Float>>>::new();
        for k in 0..self.max_iterations {
            let n = compute_n(k);
            let mut Tk = Vec::with_capacity(k + 1);
            Tk.push(self.midpoint_step(
                &mut evaluation_counter,
                delta_t,
                n,
                &f_init,
                y_init,
            ));
            for j in 0..k {
                let denominator = <S::Float as num_traits::Float>::powi(
                    cast::<_, S::Float>(n).unwrap()
                        / cast(compute_n(k - j - 1)).unwrap(),
                    2,
                ) - <S::Float as num_traits::One>::one();
                Tk.push(&Tk[j] + (&Tk[j] - &T[k - 1][j]) / denominator);
            }

            if k > 0 {
                let last_two = Tk.last_chunk::<2>().unwrap();
                let scaled_truncation_error = compute_scaled_truncation_error(
                    last_two[0].view(),
                    last_two[1].view(),
                    self.abs_tol,
                    self.rel_tol,
                );
                if scaled_truncation_error
                    <= <S::Float as num_traits::One>::one()
                {
                    y_final.assign(&last_two[1]);
                    return Ok(Stats {
                        num_system_evals: evaluation_counter.num_system_evals,
                        num_iterations: k,
                        num_midpoint_substeps: n,
                        midpoint_substep_size: delta_t
                            / cast::<_, S::Float>(n).unwrap(),
                        scaled_truncation_error,
                    });
                }
            }

            T.push(Tk);
        }

        let last_two = T.last().unwrap().last_chunk::<2>().unwrap();
        let scaled_truncation_error = compute_scaled_truncation_error(
            last_two[0].view(),
            last_two[1].view(),
            self.abs_tol,
            self.rel_tol,
        );

        let n = compute_n(self.max_iterations);
        Err(FailedToConverge {
            stats: Stats {
                num_system_evals: evaluation_counter.num_system_evals,
                num_iterations: self.max_iterations,
                num_midpoint_substeps: n,
                midpoint_substep_size: delta_t / cast(n).unwrap(),
                scaled_truncation_error,
            },
        })
    }

    fn midpoint_step(
        &self,
        evaluation_counter: &mut EvaluationCounter<S>,
        delta_t: S::Float,
        n: usize,
        f_init: &nd::Array1<S::Float>,
        y_init: nd::ArrayView1<S::Float>,
    ) -> nd::Array1<S::Float> {
        let substep_size = delta_t / cast(n).unwrap();

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
            std::mem::swap(&mut zi, &mut zip1);
            evaluation_counter.system(zi.view(), fi.view_mut());
            zip1 += &(&fi * cast::<_, S::Float>(2.).unwrap() * substep_size);
        }

        evaluation_counter.system(zip1.view(), fi.view_mut());
        (&zi + &zip1 + fi * S::Float::from(substep_size))
            * cast::<_, S::Float>(0.5).unwrap()
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

struct EvaluationCounter<'a, S: System> {
    system: &'a S,
    num_system_evals: usize,
}

impl<'a, S: System> EvaluationCounter<'a, S> {
    fn system<
        Storage: nd::RawDataMut<Elem = S::Float> + nd::Data + nd::DataMut,
    >(
        &mut self,
        y: nd::ArrayView1<S::Float>,
        dydt: nd::ArrayBase<Storage, nd::Ix1>,
    ) {
        self.num_system_evals += 1;
        <S as System>::system(&self.system, y, dydt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig() {
        let system = TrigSystem { omega: 1.2 };
        let integrator =
            Integrator::default().with_abs_tol(1e-6).with_rel_tol(0.);

        let y = nd::array![1., 0.];
        let mut y_next = nd::Array1::zeros(y.raw_dim());
        let t_final = 1.4;
        let stats = integrator
            .step(&system, t_final, y.view(), y_next.view_mut())
            .unwrap();

        let (sin, cos) = (t_final * system.omega).sin_cos();
        approx::assert_relative_eq!(y_next[0], cos, epsilon = 1e-2);
        approx::assert_relative_eq!(
            y_next[1],
            -system.omega * sin,
            epsilon = 1e-2
        );

        assert_eq!(stats.num_system_evals, 43);

        let mut solver = ode_solvers::Dop853::new(
            system,
            0.,
            t_final,
            t_final,
            ode_solvers::Vector2::new(1., 0.),
            0.,
            1e-6,
        );
        let ode_solvers_stats = solver.integrate().unwrap();
        assert_eq!(ode_solvers_stats.num_eval, 63);
    }

    struct TrigSystem {
        omega: f32,
    }

    impl System for TrigSystem {
        type Float = f32;

        fn system<
            S: nd::RawDataMut<Elem = Self::Float> + nd::Data + nd::DataMut,
        >(
            &self,
            y: nd::ArrayView1<Self::Float>,
            mut dydt: nd::ArrayBase<S, nd::Ix1>,
        ) {
            dydt[[0]] = y[[1]];
            dydt[[1]] = -self.omega.powi(2) * y[[0]];
        }
    }

    impl ode_solvers::System<f32, ode_solvers::SVector<f32, 2>> for TrigSystem {
        fn system(
            &self,
            _x: f32,
            y: &ode_solvers::SVector<f32, 2>,
            dy: &mut ode_solvers::SVector<f32, 2>,
        ) {
            <Self as System>::system(
                self,
                nd::ArrayView1::from_shape([2], y.as_slice()).unwrap(),
                nd::ArrayViewMut1::from_shape([2], dy.as_mut_slice()).unwrap(),
            );
        }
    }
}
