<!-- This README is for PyPI. The README for GitHub is in .github/README.md -->

# Liesel: A Probabilistic Programming Framework

<img src="https://raw.githubusercontent.com/liesel-devs/liesel/main/misc/logo/logo-light.png" alt="logo" align="right" width="185">

Liesel is a probabilistic programming framework with a focus on semi-parametric regression. It includes:

- [**Liesel**](https://github.com/liesel-devs/liesel), a library to express statistical models as Probabilistic Graphical Models (PGMs). Through the PGM representation, the user can build and update models in a natural way.
- **Goose**, a library to build custom MCMC algorithms with several parameter blocks and MCMC kernels such as the No U-Turn Sampler (NUTS), the Iteratively Weighted Least Squares (IWLS) sampler, or different Gibbs samplers. Goose also takes care of the MCMC bookkeeping and the chain post-processing.
- [**RLiesel**](https://github.com/liesel-devs/rliesel), an R interface for Liesel which assists the user with the configuration of semi-parametric regression models such as Generalized Additive Models for Location, Scale and Shape (GAMLSS) with different response distributions, spline-based smooth terms and shrinkage priors.

The name "Liesel" is an homage to the [Gänseliesel fountain](https://en.wikipedia.org/wiki/G%C3%A4nseliesel), landmark of Liesel's birth city [Göttingen](https://en.wikipedia.org/wiki/G%C3%B6ttingen).

## Resources

- [Paper on arXiv](https://arxiv.org/abs/2209.10975)
- [Liesel & Goose repo](https://github.com/liesel-devs/liesel)
- [Liesel & Goose API docs](https://docs.liesel-project.org)
- [RLiesel repo](https://github.com/liesel-devs/rliesel)
- [Tutorials](https://docs.liesel-project.org/en/latest/tutorials_overview.html)
