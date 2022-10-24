# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.4] - 2022-10-24

### What's new?

- :earth_africa: We have a new project homepage: <https://liesel-project.org>
- :book: Migrated the docs from pdoc to Sphinx, see [the project homepage](https://docs.liesel-project.org) (@jobrachem)
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

[Unreleased]: https://github.com/liesel-devs/liesel/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.4
[0.1.3]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.3
[0.1.2]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.2
[0.1.1]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.1
[0.1.0]: https://github.com/liesel-devs/liesel/releases/tag/v0.1.0
