# Python cell fragment: imports are visible in each guide.
# Simulate regression observations.
rng = np.random.default_rng(42)
n = 500
true_beta = np.array([1.0, 2.0])
true_sigma = 1.0
x = rng.uniform(size=n)
X_mat = np.column_stack([np.ones(n), x])
y_vec = X_mat @ true_beta + rng.normal(scale=true_sigma, size=n)

# Define the priors.
beta = lsl.Var.new_param(
    jnp.zeros(2),
    dist=lsl.Dist(tfd.Normal, 0.0, 5.0),
    name="beta",
)
sigma_sq = lsl.Var.new_param(
    1.0,
    dist=lsl.Dist(tfd.InverseGamma, concentration=3.0, scale=2.0),
    name="sigma_sq",
)

# Connect the parameters to the observations.
sigma = lsl.Var.new_calc(jnp.sqrt, sigma_sq, name="sigma")
X = lsl.Var.new_obs(X_mat, name="X")
mu = lsl.Var.new_calc(jnp.dot, X, beta, name="mu")
y = lsl.Var.new_obs(
    y_vec,
    dist=lsl.Dist(tfd.Normal, mu, sigma),
    name="y",
)

sigma_sq.biject(tfb.Exp(), name="log_sigma_sq")
log_sigma_sq = sigma_sq.bijected_var

joint = gs.MCMCSpec(
    gs.NUTSKernel,
    kernel_group="regression",
    jitter_dist=tfd.Normal(0.0, 0.2),
)
beta.inference = joint
log_sigma_sq.inference = joint
model = lsl.Model(y)
