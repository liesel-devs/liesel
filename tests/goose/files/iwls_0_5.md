# IWLS compatibility fixtures

`iwls_0_5_adaptive.pkl` and `iwls_0_5_untuned.pkl` are genuine
`SamplingResults` saved before the 0.6 migration, from commit
`33cd490ea41bd501a6a026eba860b02c3a043040` (0.5.3-dev0).
They include the legacy IWLS class, kernel states, transition diagnostics,
and samples from a non-Gaussian target with position-dependent precision.

They were generated with `legacy_run` in `tests/goose/kernels/test_mala.py`,
JAX 64-bit mode enabled, and either `IWLSKernel(["x"], initial_step_size=0.01)`
or `IWLSKernel.untuned(["x"])`. Do not regenerate them with the new IWLS
implementation: these fixtures establish the historical compatibility baseline.
