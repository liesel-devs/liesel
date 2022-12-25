# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.1] - 2022-12-25

### What's new?

- :swan: Updated the [experimental PyMC interface](https://docs.liesel-project.org/en/v0.2.1/generated/liesel.experimental.pymc.html) for Goose to PyMC 5.0 (@wiep)
- :swan: Fixed a bug with the column dtype being `object` instead of `float` in the MCMC summary (@wiep)

[All commits since 0.2.0](https://github.com/liesel-devs/liesel/compare/v0.2.0...v0.2.1)

## [0.2.0] - 2022-12-09

For this release, the Liesel modeling library has been rewritten from scratch. We are
currently working on updated tutorials explaining the new concepts introduced in v0.2.0
in full detail.

### What's new?

- :girl: Rewrote the Liesel modeling library from scratch, introducing the
  [`Var`](https://docs.liesel-project.org/en/v0.2.0/generated/liesel.model.nodes.Var.html) and the
  [`GraphBuilder`](https://docs.liesel-project.org/en/v0.2.0/generated/liesel.model.model.GraphBuilder.html)
- :girl: Replaced the
  [`SmoothPrior`](https://docs.liesel-project.org/en/v0.1.4/generated/liesel.tfp.jax.distributions.smooth_prior.SmoothPrior.html) TFP distribution with the more general
  [`MultivariateNormalDegenerate`](https://docs.liesel-project.org/en/v0.2.0/generated/liesel.distributions.mvn_degen.MultivariateNormalDegenerate.html)
- :swan: Removed deprecated functionality from the Goose summary modules
- :warning: The import paths have changed:
  - `import liesel.liesel as lsl` :arrow_right: `import liesel.model as lsl`
  - `import liesel.tfp.jax.distributions as lsld` :arrow_right: `import liesel.distributions as lsld`
  - `import liesel.tfp.jax.bijectors as lslb` :arrow_right: `import liesel.bijectors as lslb`

### Contributors

- @hriebl
- @wiep
- @jobrachem
- @GianmarcoCallegher

[All commits since 0.1.4](https://github.com/liesel-devs/liesel/compare/v0.1.4...v0.2.0)

## [0.1.4] - 2022-10-24

### What's new?

- :earth_africa: We have a new project homepage: <https://liesel-project.org>
- :book: Migrated the docs from pdoc to Sphinx: <https://docs.liesel-project.org> (@jobrachem)
- :swan: An MCMC summary can now be created with `gs.Summary(results)` instead of `gs.Summary.from_results(results)` (#13, @jobrachem)
- :swan: Fixed a bug with the reported quantiles in the per-chain MCMC summary (#14, @jobrachem)

[All commits since 0.1.3](https://github.com/liesel-devs/liesel/compare/v0.1.3...v0.1.4)

## [0.1.3] - 2022-09-01

### What's new?

- :swan: Added a generic Metropolis-Hastings kernel that can wrap a user-defined proposal function (@wiep)
- :swan: Added a column to the MCMC summary table that lists the MCMC kernel for each model parameter (@jobrachem)
- :swan: Added an [experimental PyMC interface](https://github.com/liesel-devs/liesel/blob/v0.1.3/liesel/experimental/pymc.py) that can sample PyMC models with Goose (@wiep)
- Updated dependencies: JAX == 0.3.16, BlackJAX >= 0.8.3 (@jobrachem, @hriebl)
- Improved logging setup using a non-propagating logger (@jobrachem)

[All commits since 0.1.2](https://github.com/liesel-devs/liesel/compare/v0.1.2...v0.1.3)

## [0.1.2] - 2022-07-27

### What's new?

- :swan: Fixed MCMC summary for single chains (@hriebl)

[All commits since 0.1.1](https://github.com/liesel-devs/liesel/compare/v0.1.1...v0.1.2)

## [0.1.1] - 2022-07-20

### What's new?

- :swan: Thinning is now possible in warmup and posterior epochs (@wiep)
- :swan: The NUTS kernel now reports an error when reaching the maximum tree depth (@hriebl)
- :swan: The MCMC error log can now be extracted and summarized more conveniently (@wiep, @hriebl)
- :swan: New functions for scatter and pair plots of MCMC samples (@jobrachem)
- :book: New [chapter on reproducibility](https://liesel-devs.github.io/liesel-tutorials/reproducibility.html) (@GianmarcoCallegher, @hriebl)

[All commits since 0.1.0](https://github.com/liesel-devs/liesel/compare/v0.1.0...v0.1.1)

## [0.1.0] - 2022-06-17

### What's new?

- First release.

### Contributors

- @wiep
- @hriebl
- @joel-beck
- @GianmarcoCallegher
- @jobrachem

[Unreleased]: https://github.com/liesel-devs/liesel/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/liesel-devs/liesel/releases/tag/v0.2.1
[0.2.0]: https://github.com/liesel-devs/liesel/releases/tag/v0.2.0
[0.1.4]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.4
[0.1.3]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.3
[0.1.2]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.2
[0.1.1]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.1
[0.1.0]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.0
