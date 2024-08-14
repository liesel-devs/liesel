# Changelog

All notable changes to this project will be documented in this file.

## [0.2.10] - [unreleased]

- :construction: Fixed spurious warning during model creation (#186, @jobrachem)
- :construction: Update logging setup (#187, @jobrachem)
- :sparkles: New method `Var.transform`. (#174, @jobrachem)

## [0.2.9] - 2024-04-04

- :construction: Pinned scipy version to <=1.12 (@jobrachem)
- :construction: Handeled error in `liesel.model.model.transform` for duplicated nodes (#167, @GianmarcoCallegher)
- :sparkles: `lsl.Calc` nodes will now try to evaluate their function upon initialization. This can make it much easier to spot errors early. The behavior can be turned off by passing `update_on_init=False`.  (#92, @jobrachem)
- :construction: Fixed unexpected behaviour `liesel.model.viz.plot_vars` when using an intermediate `Calc`. (#168, @GianmarcoCallegher)

## [0.2.8] - 2023-12-03

- :sparkles: Added `NamedTupleInterface` (#151, @wiep)

## [0.2.7] - 2023-11-10

- :sparkles: Large documentation updates! (#149, #146, #145, #143, #135, #134, @jobrachem; #144, #140, @GianmarcoCallegher)
- :sparkles: The method `gs.Summary.error_df()` is now publicly available. (#137, @jobrachem)
- :sparkles: The class `liesel.goose.engine.SamplingResults` is now exported via `liesel.goose`, which means it can be used as `gs.SamplingResults`. (@jobrachem)
- :sparkles: Improved the efficiency of the `liesel.distributions.mvn_degen.MultivariateNormalDegenerate.from_penalty` constructor (#101, @GianmarcoCallegher)
- :sparkles: Added alternative constructor `.from_penalty_smooth` to `MultivariateNormalDegenerate` (#133, @GianmarcoCallegher)
- :construction: Added `observed=True` to a `pd.DataFrame.groupby()` call in `goose/summary_m.py` to silence a warning due to a deprecation in [pandas v2.1.0](https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#deprecations)
- :construction: Renamed `lsl.Param` to `lsl.param` and `lsl.Obs` to `lsl.obs` to reflect the fact that those are functions, not classes. The old names are deprecated and scheduled for removal in v0.4.0. (#130, @jobrachem)
- :construction: Renamed/moved the following symbols. The old names are deprecated and scheduled for removal in v0.4.0. (#136, @jobrachem)
    - `liesel.model.goose.GooseModel` -> (renamed/moved) `liesel.goose.interface.LieselInterface`
    - `liesel.goose.models.DictModel` -> (renamed/moved) `liesel.goose.interface.DictInterface`
    - `liesel.goose.models.DataClassModel` -> (renamed/moved) `liesel.goose.interface.DataclassInterface`
- :construction: Some maintenance updates to stay up to date with dependencies (#147, #141, #102, @GianmarcoCallegher; #139, @wiep; #109, @jobrachem)


## [0.2.5] - 2023-09-28

- :construction: Updated for compatibility with Blackjax 1.0.0 (#100, @wiep & @hriebl)
- :construction: Updated for compatibility with the latest mypy update (#97, @wiep & @hriebl)
- :sparkles: Added functionality for easy setup and customization of initial value jittering (#72, @GianmarcoCallegher & @hriebl)
- :sparkles: Improved error messages in `lsl.Calc.update()` (#84, @jobrachem)
- :construction: Fixed a bug in `gs.plot_param()` (#81, @viktoriussuwandi)
- :construction: Fixed an error in the tutorial on linear regression (#85, @jobrachem)
- :construction: Fixed the display of the plot title in `gs.plot_scatter()` (#98, @hriebl)

## [0.2.4] - 2023-08-18

### What's new?

- :construction: Removed all references to `jax.numpy.DeviceArray` to make Liesel compatible with Jax 0.4.14 (#73, @jobrachem)
- ✨ Added a visual distinction for edges that represent a connection to a variable's distribution or value (#76, @GianmarcoCallegher)
- ✨ Added `ls.Model.simulate()`, which provides a convenient way to draw random samples from a Liesel model using the specified priors. (#70, @hriebl)
- ✨ Added `liesel.model.goose.finite_discrete_gibbs_kernel`, which helps you to automatically set up a `gs.GibbsKernel` for a discrete variable (#64 & #65, @jobrachem and @hriebl)
- ✨ Added an intialization message to `gs.Engine` (#66, @GianmarcoCallegher)

## [0.2.3] - 2023-03-31

### What's new?

- :book: Improved documentation (#47, #48, @jobrachem, @hriebl)
- :sparkles:  Added convert_dtype to graph builder (#50, @hriebl)
- :sparkles: [New overview page](https://docs.liesel-project.org/en/latest/tutorials_overview.html) for tutorials (#62, @jobrachem)
- ✨ [New tutorial](https://docs.liesel-project.org/en/latest/tutorials/md/07-groups.html) on advanced group usage (#63, @jobrachem)
- :sparkles: Added a method to convert sampling results to arviz's inference data (#49, @wiep)
- :construction: Changes `__repr__` for multiple classes in `liesel.model` (#57, @wiep)

## [0.2.2] - 2023-03-08

### What's new?

- :truck: The [tutorials](https://docs.liesel-project.org/en/latest/#tutorials) have been updated to v.0.2.2 and are now part of the documentation (@jobrachem, @wiep, @hriebl, @GianmarcoCallegher)
- :sparkles:  Added [new tutorial](https://docs.liesel-project.org/en/latest/tutorials/md/06-pymc.html) showcasing the interface to PyMC (@GianmarcoCallegher)
- :sparkles: Added node / variable groups (#28, @jobrachem)
- :sparkles: Sampling from the `MultivariateNormalDegenerate` is now possible (#34, @jobrachem)
- :construction: Fixed undefined behaviour in the distreg module (#20, @hriebl)
- :construction: The distreg module will now use variables names as position keys (#22, @hriebl)

[All commits since 0.2.1](https://github.com/liesel-devs/liesel/compare/v0.2.1...v0.2.2)

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
