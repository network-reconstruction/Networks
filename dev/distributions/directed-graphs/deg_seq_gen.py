import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, Optional, Any
from distributions import bivariate_lognormal, bivariate_lognormal_pareto2
from scipy import stats

class DegreeSeq:
    def __init__(self, n: int, kappa_out: Optional[jnp.ndarray] = None, kappa_in: Optional[jnp.ndarray] = None, verbose: bool = False):
        self.n = n
        self.verbose = verbose

        if kappa_out is None:
            if verbose:
                print('No kappa_out provided. Setting kappa_out to 0')
            self.kappa_out = jnp.zeros(n)
        else:
            self.kappa_out = kappa_out
        
        if kappa_in is None:
            if verbose:
                print('No kappa_in provided. Setting kappa_in to 0')
            self.kappa_in = jnp.zeros(n)
        else:
            self.kappa_in = kappa_in

    def generate(self, key: Any, distribution: Callable, **kwargs: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.verbose:
            print('Generating degree sequence...')
        
        in_degrees, out_degrees = distribution(key, self.n, **kwargs)
        self.in_degrees = in_degrees
        self.out_degrees = out_degrees

        if self.verbose:
            print('Degree sequence generated successfully!')
            print(f'Average out-degree: {jnp.mean(self.out_degrees)}')
            print(f'Average in-degree: {jnp.mean(self.in_degrees)}')

        return self.in_degrees, self.out_degrees

    def plot(self, save_path: str) -> None:
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))

        def plot_ccdf(data, ax, color, title, xlabel):
            sorted_data = jnp.sort(data)
            ccdf = 1.0 - jnp.arange(len(sorted_data)) / len(sorted_data)
            ax.plot(sorted_data, ccdf, marker='.', linestyle='none', color=color)
            ax.set_title(title)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('CCDF (log scale)')

        # Plot for out-degrees CCDF
        plot_ccdf(self.out_degrees, axs[0, 0], 'skyblue', 'Out-Degrees CCDF', 'Out-Degrees')

        # Plot for in-degrees CCDF
        plot_ccdf(self.in_degrees, axs[0, 1], 'salmon', 'In-Degrees CCDF', 'In-Degrees')
        
        # Plot for joint distribution
        sns.scatterplot(x=self.out_degrees, y=self.in_degrees, ax=axs[0, 2], alpha=0.5)
        axs[0, 2].set_title('Log-Log Joint Distribution of In-Degrees and Out-Degrees')
        axs[0, 2].set_xscale('log')
        axs[0, 2].set_yscale('log')
        axs[0, 2].set_xlabel('Out-Degrees (log scale)')
        axs[0, 2].set_ylabel('In-Degrees (log scale)')
        
        # QQ plot for out-degrees against lognormal
        stats.probplot(self.out_degrees, dist="lognorm", sparams=(jnp.std(jnp.log(self.out_degrees)),), plot=axs[1, 0])
        axs[1, 0].set_title('QQ Plot of Out-Degrees vs Lognormal')
        
        # QQ plot for out-degrees against pareto
        stats.probplot(self.out_degrees, dist="pareto", sparams=(1.59,), plot=axs[1, 1])
        axs[1, 1].set_title('QQ Plot of Out-Degrees vs Pareto')
        
        # QQ plot for in-degrees against lognormal
        stats.probplot(self.in_degrees, dist="lognorm", sparams=(jnp.std(jnp.log(self.in_degrees)),), plot=axs[2, 0])
        axs[2, 0].set_title('QQ Plot of In-Degrees vs Lognormal')

        # QQ plot for in-degrees against pareto
        stats.probplot(self.in_degrees, dist="pareto", sparams=(2.38,), plot=axs[2, 1])
        axs[2, 1].set_title('QQ Plot of In-Degrees vs Pareto')

        # PDF plot for out-degrees
        sns.kdeplot(jnp.log(self.out_degrees), ax=axs[1, 2], color='skyblue', fill=True)
        axs[1, 2].set_title('PDF of Log Out-Degrees')
        axs[1, 2].set_xlabel('Log Out-Degrees')

        # PDF plot for in-degrees
        sns.kdeplot(jnp.log(self.in_degrees), ax=axs[2, 2], color='salmon', fill=True)
        axs[2, 2].set_title('PDF of Log In-Degrees')
        axs[2, 2].set_xlabel('Log In-Degrees')
        
        plt.tight_layout()
        plt.savefig(save_path)
        if self.verbose:
            print(f'Degree sequence plot saved to {save_path}')

    def save(self, save_path: str) -> None:
        with open(save_path, 'w') as f:
            f.write('# kappa_out  kappa_in\n')
            for i in range(self.n):
                f.write(f'{self.out_degrees[i]}  {self.in_degrees[i]}\n')
        
        if self.verbose:
            print(f'Degree sequence saved to {save_path}')


if __name__ == "__main__":
    #Ecuador 2015
    mu = jnp.array([35, 35])
    Sigma = jnp.log(jnp.array([[2.89, 1.37], [1.37, 2.06]]))
    key = random.PRNGKey(0)
    transition_point = 5.0  # Transition point for the weighting function

    degree_seq = DegreeSeq(n=10000, verbose=True)
    in_degrees, out_degrees = degree_seq.generate(key, bivariate_lognormal_pareto2, exp_mu=mu, Sigma=Sigma, tail_indices=[1.59, 2.38], transition_point=transition_point)
    degree_seq.plot('degree_sequence_mixture.png')
    degree_seq.save('degree_sequence_mixture.txt')
