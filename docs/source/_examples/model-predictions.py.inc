# Python cell fragment: imports are visible in each guide.
x_data = jnp.linspace(-1.0, 1.0, 120)
true_mean = 1.0 + 2.0 * x_data
true_scale = jnp.exp(-0.4 + 0.7 * x_data)
y_data = true_mean + true_scale * jax.random.normal(jax.random.key(8), (120,))

beta = lsl.Var.new_param(
    jnp.zeros(2),
    dist=lsl.Dist(tfd.Normal, loc=0.0, scale=2.5),
    name="beta",
)
gamma = lsl.Var.new_param(
    jnp.zeros(2),
    dist=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    name="gamma",
)
x = lsl.Var.new_obs(x_data, name="x")
mu = lsl.Var.new_calc(
    lambda x, b: b[0] + b[1] * x,
    x,
    beta,
    name="mu",
)
log_sigma = lsl.Var.new_calc(
    lambda x, g: g[0] + g[1] * x,
    x,
    gamma,
    name="log_sigma",
)
sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")
y = lsl.Var.new_obs(
    y_data,
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=sigma),
    name="y",
)
model = lsl.Model(y)

for parameter in model.parameters.values():
    parameter.inference = gs.MCMCSpec(
        gs.NUTSKernel,
        kernel_group="regression",
        jitter_dist=tfd.Normal(loc=0.0, scale=0.1),
    )
results = gs.LieselMCMC(model).run_for_epochs(
    seed=31,
    num_chains=4,
    adaptation=500,
    posterior=500,
    show_progress=False,
)
samples = results.get_posterior_samples()
