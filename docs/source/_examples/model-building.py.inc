# Python cell fragment: imports are visible in each guide.
x_data = jnp.linspace(-1.0, 1.0, 80)
y_data = 1.0 + 2.0 * x_data + 0.7 * jax.random.normal(jax.random.key(42), (80,))

prior_scale = lsl.Var.new_value(2.5, name="prior_scale")
beta = lsl.Var.new_param(
    jnp.zeros(2),
    dist=lsl.Dist(tfd.Normal, loc=0.0, scale=prior_scale),
    name="beta",
)
variance = lsl.Var.new_param(
    1.0,
    dist=lsl.Dist(tfd.LogNormal, loc=0.0, scale=0.5),
    name="variance",
)
x = lsl.Var.new_obs(x_data, name="x")

mu = lsl.Var.new_calc(
    lambda x, b: b[0] + b[1] * x,
    x,
    beta,
    name="mu",
)
sigma = lsl.Var.new_calc(jnp.sqrt, variance, name="sigma")
y = lsl.Var.new_obs(
    y_data,
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=sigma),
    name="y",
)
model = lsl.Model(y)

# Match the tutorial after its reactive update.
beta.value = jnp.array([1.0, 2.0])
