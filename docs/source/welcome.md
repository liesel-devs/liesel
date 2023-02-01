Welcome to the API documentation of Liesel, a probabilistic programming framework with
a focus on semi-parametric regression. It includes:

- [**Liesel**][1], a library to express statistical models as Probabilistic Graphical
  Models (PGMs). Through the PGM representation, the user can build and update models
  in a natural way.
- **Goose**, a library to build custom MCMC algorithms with several parameter blocks
  and MCMC kernels such as the No U-Turn Sampler (NUTS), the Iteratively Weighted Least
  Squares (IWLS) sampler, or different Gibbs samplers. Goose also takes care of the
  MCMC bookkeeping and the chain post-processing.
- [**RLiesel**][2], an R interface for Liesel which assists the user with the
  configuration of semi-parametric regression models such as Generalized Additive
  Models for Location, Scale and Shape (GAMLSS) with different response distributions,
  spline-based smooth terms and shrinkage priors.

The name "Liesel" is an homage to the [Gänseliesel fountain][3], landmark of Liesel's
birth city [Göttingen][4].

# Installation

You can install Liesel via pip:

```
$ pip install liesel
```

For more detailed installation instructions, see the [README][5] in the main repository.
This may be sensible, if you want tot work with the latest development version, or
if you want to install PyGraphviz for nicely ordered display of Liesel model graphs.

# Tutorials

To start working with Liesel, our tutorials might come in handy, starting with
a tutorial on [linear regression](tutorials/md/01-lin-reg.md#linear-regression).

# Further Reading

For a scientific discussion of the software, see our [paper][6] on arXiv.


# Acknowledgements

Liesel is being developed by Paul Wiemann and Hannes Riebl at the
[University of Göttingen][7] with support from Thomas Kneib. Important contributions
were made by Joel Beck, Alex Afanasev, Gianmarco Callegher and Johannes Brachem. We are
grateful to the [German Research Foundation (DFG)][7] for funding the development
through grant 443179956.

<img src="https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg" alt="University of Göttingen" style="height: 4em; margin: 1em 2em 1em 0">
<img src="https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg" alt="Funded by DFG" style="height: 4em; margin: 1em 0 1em 0">

[1]: https://github.com/liesel-devs/liesel
[2]: https://github.com/liesel-devs/rliesel
[3]: https://en.wikipedia.org/wiki/G%C3%A4nseliesel
[4]: https://en.wikipedia.org/wiki/G%C3%B6ttingen
[5]: https://github.com/liesel-devs/liesel#installation
[6]: https://arxiv.org/abs/2209.10975
[7]: https://www.uni-goettingen.de/en
[8]: https://www.dfg.de/en
