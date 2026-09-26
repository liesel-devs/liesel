# Linear Regression

The introduction now lives in {doc}`../notebooks/11-model-building`.
It covers variables, priors, model graphs, densities, and reactive updates.

<a id="imports"></a>
<a id="generating-the-data"></a>
<a id="building-the-model"></a>
<a id="the-regression-coefficients"></a>
<a id="the-variance-and-standard-deviation"></a>
<a id="design-matrix-fitted-values-and-response"></a>
<a id="bringing-the-model-together"></a>

<a id="generate-data"></a>
<a id="plot-vars"></a>

For a complete posterior fit that samples both coefficients and noise variance,
continue with {doc}`01c-transform`.

<a id="mcmc-inference-with-goose"></a>

The earlier example sampled only the coefficients while holding its variance
fixed. A prior alone does not assign a sampling kernel. See
{doc}`../../goose-kernels` for this distinction and
{doc}`../../model-modification` for deliberately changing a model.
