
Tutorials
=========

Welcome to the tutorials page for Liesel, a probabilistic programming framework for developing semi-parametric regression models and custom Bayesian inference algorithms. In these tutorials, you will learn how to use Liesel to express statistical models as probabilistic graphical models (PGMs), manipulate and update them with ease, and run efficient and modular Markov chain Monte Carlo (MCMC) procedures with different kernels.

Here are the basics to get you started, ranging from simple linear regression up to
location-scale models:

.. toctree::
   :maxdepth: 1
   :caption: Basics

   Overview <self>
   tutorials/md/01a-lin-reg
   tutorials/md/01b-model
   tutorials/md/01d-gibbs-sampling
   tutorials/md/01c-transform
   tutorials/md/02-ls-reg

Here are some more advanced topics, including generalized extreme value (GEV) models,
a comparison of different samplers, and more:

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   tutorials/md/03-gev
   tutorials/md/04-mcycle
   tutorials/md/05-reproducibility
   tutorials/md/06-pymc
   tutorials/md/07-error-correction
   tutorials/md/08-custom-kernel

For parameter fitting, start with the :doc:`optimization` guide. It links to
two worked examples and short guides for common tasks.

For approximate posteriors, start with :doc:`variational-inference`. Its two
tutorials introduce Gaussian approximations, posterior sampling and models with
multiple observation groups using :class:`liesel.optim.LieselVI`.
