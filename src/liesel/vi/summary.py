from typing import Dict, Any

import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import arviz as az


class Summary:
    """
    Posterior statistics and diagnostics.

    This summary object computes key posterior statistics for each latent variable,
    including the posterior mean, variance, 2.5% and 97.5% quantiles, and the highest
    density interval (HDI) all based on samples from the posterior distributins. Oriented on lsl.goose module.

    Parameters
    ----------
    results : dict
        Dictionary containing the inference results with the following keys:
          - "final_variational_distributions": dict mapping latent variable names to their final
            variational distribution objects.
          - "elbo_values": list or array of ELBO values over iterations.
          - "samples": dict mapping latent variable names to pre-generated samples.
    """

    def __init__(self, results: Dict[str, Any]):
        self.final_variational_distributions = results[
            "final_variational_distributions"
        ]
        self.elbo_values = results["elbo_values"]
        self.samples = results["samples"]

    def compute_posterior_summary(self, hdi_prob: float = 0.9) -> pd.DataFrame:
        """
        Compute and return a tidy table of univariate summary statistics for each latent variable's posterior.
        Note: This summary ignores any interactions or covariances between variables. In cases of multivariate latent variables,
        while covariance is taken into account during optimization, only the marginal summaries for each individual variable are displayed.

        For each latent variable $ \theta $, the following statistics are computed:

        $$
        \begin{align}
        \mu &= \mathbb{E}[\theta],\\[1mm]
        \sigma^2 &= \mathbb{V}[\theta],\\[1mm]
        Q_{0.025}(\theta) &= \text{2.5\% quantile},\\[1mm]
        Q_{0.975}(\theta) &= \text{97.5\% quantile},\\[1mm]
        \text{HDI}_{\text{low}}(\theta),\ \text{HDI}_{\text{high}}(\theta) &=
        \text{Highest Density Interval at the specified level (e.g. 90\% if } hdi\_prob=0.9\text{)}.
        \end{align}
        $$

        Returns
        -------
        df : pd.DataFrame
            A DataFrame with columns "variable", "mean", "variance", "2.5%", "97.5%",
            "hdi_low" and "hdi_high" containing the computed statistics.
        """
        rows = []
        for var in self.samples.keys():
            samples = self.samples[var]
            samples_np = np.asarray(samples)
            mean = jnp.mean(samples, axis=0)
            variance = jnp.var(samples, axis=0)
            lower = jnp.percentile(samples, 2.5, axis=0)
            upper = jnp.percentile(samples, 97.5, axis=0)
            if mean.shape == ():
                hdi_interval = az.hdi(samples_np, hdi_prob=hdi_prob)
                rows.append(
                    {
                        "variable": var,
                        "mean": float(mean),
                        "variance": float(variance),
                        "2.5%": float(lower),
                        "97.5%": float(upper),
                        "hdi_low": float(hdi_interval[0]),
                        "hdi_high": float(hdi_interval[1]),
                    }
                )
            elif mean.ndim == 1:
                for idx in range(mean.shape[0]):
                    hdi_interval = az.hdi(samples_np[:, idx], hdi_prob=hdi_prob)
                    rows.append(
                        {
                            "variable": f"{var}[{idx}]",
                            "mean": float(mean[idx]),
                            "variance": float(variance[idx]),
                            "2.5%": float(lower[idx]),
                            "97.5%": float(upper[idx]),
                            "hdi_low": float(hdi_interval[0]),
                            "hdi_high": float(hdi_interval[1]),
                        }
                    )
            else:
                raise ValueError(
                    "compute_posterior_summary only supports scalar or 1D latent variables."
                )
        df = pd.DataFrame(rows)
        return df

    def plot_elbo(
        self,
        title: str = "ELBO Progress",
        xlabel: str = "Iterations",
        ylabel: str = "ELBO",
        style: str = "whitegrid",
        color: str = "blue",
        save_path: str = None,
    ) -> None:
        """
        Plot the ELBO progression over iterations.

        Parameters
        ----------
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        style : str, optional
            Seaborn style.
        color : str, optional
            Color for the line plot.
        save_path : str, optional
            File path to save the plot.
        """
        sns.set_theme(style=style)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(self.elbo_values)), y=self.elbo_values, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_density(
        self,
        variable: str,
        title: str = None,
        style: str = "whitegrid",
        xlabel: str = None,
        save_path: str = None,
    ) -> None:
        """
        Plot the posterior density of a latent variable using pre-generated samples.

        Parameters
        ----------
        variable : str
            Name of the latent variable.
        title : str, optional
            Plot title; defaults to "Density Plot for <variable>".
        style : str, optional
            Seaborn style.
        xlabel : str, optional
            X-axis label; defaults to the variable name.
        save_path : str, optional
            File path to save the plot.
        """
        if variable not in self.samples:
            raise ValueError(f"Samples for variable {variable} not provided.")
        samples = self.samples[variable]
        sns.set_theme(style=style)

        if samples.ndim == 1 or (samples.ndim == 2 and samples.shape[1] == 1):
            plt.figure(figsize=(8, 6))
            sns.kdeplot(x=samples.ravel(), fill=True)
            if title is None:
                title = f"Density Plot for {variable}"
            if xlabel is None:
                xlabel = variable
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Density")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()

        elif samples.ndim == 2:
            num_dims = samples.shape[1]
            fig, axes = plt.subplots(
                num_dims, 1, figsize=(8, 4 * num_dims), squeeze=False
            )
            for i, ax in enumerate(axes[:, 0]):
                sns.kdeplot(x=samples[:, i], fill=True, ax=ax)
                ax.set_xlabel(f"{variable}[{i}]")
                ax.set_ylabel("Density")
            if title is None:
                title = f"Density Plot for {variable}"
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            raise ValueError("Unsupported sample dimensions for density plot.")

    def plot_pairwise(
        self,
        variable: str,
        title: str = None,
        style: str = "whitegrid",
        save_path: str = None,
    ) -> None:
        """
        Produce a pairwise scatter plot matrix for a multivariate latent variable.

        Parameters
        ----------
        variable : str
            Name of the latent variable.
        title : str, optional
            Plot title.
        style : str, optional
            Seaborn style.
        save_path : str, optional
            File path to save the plot.
        """
        if variable not in self.samples:
            raise ValueError(f"Samples for variable {variable} not provided.")
        samples = self.samples[variable]
        if samples.ndim == 1 or (samples.ndim == 2 and samples.shape[1] == 1):
            print("Pairwise plot is not applicable for univariate distributions.")
            return
        elif samples.ndim == 2:
            df = pd.DataFrame(
                samples, columns=[f"{variable}_{i}" for i in range(samples.shape[1])]
            )
            sns.set_theme(style=style)
            g = sns.pairplot(df)
            if title is not None:
                g.fig.suptitle(title, y=1.02)
            if save_path:
                g.fig.savefig(save_path)
            else:
                plt.show()
        else:
            raise ValueError("Unsupported sample dimensions for pairwise plot.")

    def __repr__(self) -> str:
        return repr(self.compute_posterior_summary())

    def __str__(self) -> str:
        return str(self.compute_posterior_summary())
