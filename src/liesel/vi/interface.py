import copy

import jax.numpy as jnp


class LieselInterface:
    def __init__(self, model):
        self.model = model

    def get_params(self) -> dict[str, jnp.ndarray]:
        """Retrieve model parameters as a dictionary."""
        params = {
            pname: jnp.array(var.value) for pname, var in self.model.vars.items()
        }  # allows single int input for params
        return params

    def compute_log_prob(
        self,
        param_values: dict[str, jnp.ndarray],
        dim_data,
        batch_size=None,
        batch_indices=None,
    ) -> jnp.ndarray:
        """
        Compute the log probability of the model given parameter values.

        This function calculates the log probability of the model based on the provided
        parameters. If `batch_size` is specified, it computes the log probability over
        a subset of the data corresponding to the given batch; otherwise, it evaluates
        the log probability over the entire dataset.

        Parameters:
            param_values (Dict[str, jnp.ndarray]): Dictionary of parameter names and their values.
            dim_data: Dimensionality of the data.
            batch_size: Size of the batch (optional).
            batch_indices: Indices for batching (optional).

        Returns:
            log_prob: Computed log probability.
        """
        model_copy = copy.deepcopy(self.model)
        model_copy.auto_update = False

        for pname, value in param_values.items():
            if pname in model_copy.vars:
                model_copy.vars[pname].value = value
            else:
                raise KeyError(f"Parameter '{pname}' not part of the model.")

        if batch_size is None:
            model_copy.update()
            return model_copy.log_prob
        else:
            model_copy = self._subset_data(model_copy, batch_indices)
            model_copy.update()

            scale = dim_data / batch_size
            log_likelihood = (
                scale * model_copy.log_lik
            )  # #Kucukelbir: only scaling of likelihood
            log_prior = model_copy.log_prior
            log_prob = log_likelihood + log_prior

            return log_prob

    def _subset_data(self, model, batch_indices):
        """
        Subset the data in the model based on batch indices.

        Parameters:
            model: The model instance to subset.
            batch_indices: Indices to select the batch.

        Returns:
            model: The model with subsetted data.
        """
        batch_indices = jnp.array(batch_indices)
        for var in model.vars.values():
            if getattr(var, "observed", True):
                var.value = jnp.take(var.value, batch_indices, axis=0)

        return model
