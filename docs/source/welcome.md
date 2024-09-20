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

If you want to work with the latest development version of Liesel or use PyGraphviz for
prettier plots of the model graphs, see the [README][5] in the main repository.

Now you can get started. Throughout this documentation, we import Liesel as follows:

```python
import liesel.model as lsl
import liesel.goose as gs
```

We also commonly use the following imports:

```python
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
```

We provide overviews of the most important building blocks provided by `liesel.model`
and `liesel.goose` in [Model Building (liesel.model)](model_overview) and
[MCMC Sampling (liesel.goose)](goose_overview), respectively.

# Tutorials

To start working with Liesel, our tutorials might come in handy, starting with
a tutorial on [linear regression](tutorials/md/01-lin-reg.md#linear-regression). An overview of our tutorials can be found here: [Liesel tutorials](tutorials_overview.rst).


# Further Reading

For a scientific discussion of the software, see our [paper][6] on arXiv.


# Acknowledgements

Liesel is being developed by Paul Wiemann and Hannes Riebl at the
[University of Göttingen][7] with support from Thomas Kneib. Important contributions
were made by Joel Beck, Alex Afanasev, Gianmarco Callegher and Johannes Brachem. We are
grateful to the [German Research Foundation (DFG)][8] for funding the development
through grant 443179956.

<img src="https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg" alt="University of Göttingen">
<img src="https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg" alt="Funded by DFG">

[1]: https://github.com/liesel-devs/liesel
[2]: https://github.com/liesel-devs/rliesel
[3]: https://en.wikipedia.org/wiki/G%C3%A4nseliesel
[4]: https://en.wikipedia.org/wiki/G%C3%B6ttingen
[5]: https://github.com/liesel-devs/liesel#installation
[6]: https://arxiv.org/abs/2209.10975
[7]: https://www.uni-goettingen.de/en
[8]: https://www.dfg.de/en
