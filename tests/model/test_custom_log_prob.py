import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

import liesel.goose as gs
import liesel.model as lsl


def test_custom_log_prob():
    node = lsl.Var.new_param(0.0, lsl.Dist(tfp.distributions.Normal, 0.0, 1.0), "node")
    log_prob_node = lsl.Calc(lambda lp: lp + 1.0, node.dist_node)
    mb = lsl.GraphBuilder()
    mb.add(node)
    mb.log_prob_node = log_prob_node

    model = mb.build_model()

    state = model.state
    interface = gs.LieselInterface(model)

    updated_state = interface.update_state(gs.Position({"node": 1.0}), state)

    model_log_probs = jnp.array(
        [interface.log_prob(state), interface.log_prob(updated_state)]
    )

    log_probs = tfp.distributions.Normal(0.0, 1.0).log_prob(jnp.array([0.0, 1.0])) + 1.0

    assert jnp.allclose(model_log_probs, log_probs)


def test_custom_log_prior():
    node = lsl.Var.new_param(0.0, lsl.Dist(tfp.distributions.Normal, 0.0, 1.0), "node")
    log_prob_node = lsl.Calc(lambda lp: lp + 1.0, node.dist_node)
    mb = lsl.GraphBuilder()
    mb.add(node)
    mb.log_prior_node = log_prob_node

    model = mb.build_model()

    state = model.state
    interface = gs.LieselInterface(model)

    updated_state = interface.update_state(gs.Position({"node": 1.0}), state)

    # interface does not give access to prior, so it is directly accessed using the
    # reserved node name
    model_log_priors = jnp.array(
        [state["_model_log_prior"].value, updated_state["_model_log_prior"].value]
    )

    log_probs = tfp.distributions.Normal(0.0, 1.0).log_prob(jnp.array([0.0, 1.0])) + 1.0

    assert jnp.allclose(model_log_priors, log_probs)


def test_custom_log_likelihood():
    node = lsl.Var.new_param(0.0, lsl.Dist(tfp.distributions.Normal, 0.0, 1.0), "node")
    log_prob_node = lsl.Calc(lambda lp: lp + 1.0, node.dist_node)
    mb = lsl.GraphBuilder()
    mb.add(node)
    mb.log_lik_node = log_prob_node

    model = mb.build_model()

    state = model.state
    interface = gs.LieselInterface(model)

    updated_state = interface.update_state(gs.Position({"node": 1.0}), state)

    # interface does not give access to likelihood, so it is directly accessed
    # using the reserved node name
    model_log_liks = jnp.array(
        [state["_model_log_lik"].value, updated_state["_model_log_lik"].value]
    )

    log_probs = tfp.distributions.Normal(0.0, 1.0).log_prob(jnp.array([0.0, 1.0])) + 1.0

    assert jnp.allclose(model_log_liks, log_probs)
