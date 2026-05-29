# Crank-Nicolson independent cross-check of the H^n heat kernels

An independent confirmation that the library's hyperbolic heat kernels
solve the radial heat equation, using a method that shares none of the
Grigor'yan operator-chain machinery they were derived from.

Run: `uv run python notes/validation/heat_kernel_crank_nicolson.py`

## Method

The radial heat equation on `H^n` is
`∂_t u = ∂_r² u + (n-1) coth(r) ∂_r u`. Take the library kernel
`k^n_{t0}(r)` as initial data on a radial grid, evolve it to `t1` with
Crank-Nicolson (centered second difference + centered first difference;
Dirichlet boundary values read from the kernel at the moving time), and
compare the interior to the library kernel `k^n_{t1}(r)`. A kernel that
solves the PDE is reproduced to the scheme's `O(Δt² + h²)` truncation
error; a wrong kernel is not. This is the operator-residual half of the
C7 dual check, realized with an actual time-stepper.

Grid: `[0.1, 10.0]`, `m = 4000` (`h ≈ 2.5e-3`), evolve `t0 = 0.3 → t1 =
0.6` in `300` Crank-Nicolson steps (`Δt = 1e-3`).

## Results

| n | path | max relative error | at r |
|---:|---|---:|---:|
| 5 | closed form | 7.83e-05 | 4.62 |
| 7 | closed form (this pass) | 1.82e-04 | 4.18 |
| 9 | recursion from n=7 | 3.91e-04 | 3.81 |

(Error measured where the kernel exceeds `1e-6 ×` its peak, to avoid
dividing into the negligible large-`r` tail.)

## Reading

Each library kernel is reproduced by an independent time-stepper to the
Crank-Nicolson `O(Δt² + h²)` floor (~`1e-4`–`1e-3` on this grid). The
error grows mildly with `n` (the `n=9` recursion compounds one
`torch.autograd.grad` step on top of the `n=7` closed form, and the
`coth` advection term stiffens with `n`), but stays at the discretization
floor for all three. A closed form with a wrong coefficient — the kind of
error the original spectral-shift bug (C1) introduced — would show an
`O(1)` mismatch here, not `1e-4`.

This complements the symbolic verification
(`notes/verification/heat_kernel_n5_sympy.py`,
`heat_kernel_n7_sympy.py`): symbolic algebra proves the closed forms
equal the operator chain exactly; the Crank-Nicolson solver confirms,
through a completely separate numerical route, that they evolve as
solutions of the heat equation.
