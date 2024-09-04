import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, Optional, Any
from distributions import bivariate_lognormal, bivariate_lognormal_pareto2
from scipy import stats

class DegreeSeq:
    """
    A class to generate, plot, and save degree sequences for networks.

    Attributes:
        n (int): Number of nodes in the network.
        kappa_out (Optional[jnp.ndarray]): Out-degree sequence.
        kappa_in (Optional[jnp.ndarray]): In-degree sequence.
        verbose (bool): Enable verbose output for debugging.

    Methods:
        generate(key: Any, distribution: Callable, **kwargs: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Generate in-degree and out-degree sequences using a specified distribution.
        plot(save_path: str) -> None:
            Plot various distributions and relationships of the degree sequences.
        save(save_path: str) -> None:
            Save the degree sequences to a file.
    """

    def __init__(self, 
                 n: int, 
                 kappa_out: Optional[jnp.ndarray] = None,
                 kappa_in: Optional[jnp.ndarray] = None,
                 verbose: bool = False):
        """
        Initialize the DegreeSeq class.

        Args:
            n (int): Number of nodes.
            kappa_out (Optional[jnp.ndarray]): Out-degree sequence.
            kappa_in (Optional[jnp.ndarray]): In-degree sequence.
            verbose (bool): Enable verbose output.
        """
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
        """
        Generate in-degree and out-degree sequences using a specified distribution.

        Args:
            key (Any): Random key for reproducibility.
            distribution (Callable): Distribution function to generate degrees.
            **kwargs (Any): Additional arguments for the distribution function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Generated in-degrees and out-degrees.
        """
        if self.verbose:
            print('Generating degree sequence...')
        
        out_degrees, in_degrees  = distribution(key, self.n, **kwargs)
        self.in_degrees = in_degrees
        self.out_degrees = out_degrees

        if self.verbose:
            print('Degree sequence generated successfully!')
            print(f'Average out-degree: {jnp.mean(self.out_degrees)}')
            print(f'Average in-degree: {jnp.mean(self.in_degrees)}')

        return self.in_degrees, self.out_degrees

    def plot(self, save_path: str) -> None:
        """
        Plot various distributions and relationships of the degree sequences.

        Args:
            save_path (str): Path to save the plot.
        """
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))

        def plot_ccdf(data: jnp.ndarray, ax: plt.Axes, color: str, title: str, xlabel: str) -> None:
            """
            Plot the complementary cumulative distribution function (CCDF).

            Args:
                data (jnp.ndarray): Data to plot.
                ax (plt.Axes): Axis to plot on.
                color (str): Color of the plot.
                title (str): Title of the plot.
                xlabel (str): Label for the x-axis.
            """
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

        # Plot 2D histogram for joint distribution of in-degrees and out-degrees
        x = self.out_degrees
        y = self.in_degrees
        # print(f"first 100 points: {jnp.stack([x, y], axis=1)[:100]}")
        # Log-spaced bins
        bins = [jnp.logspace(jnp.log10(min(x[x > 0])), jnp.log10(max(x)), 60), 
                jnp.logspace(jnp.log10(min(y[y > 0])), jnp.log10(max(y)), 60)]

        # 2D histogram
        hist, xedges, yedges, img = axs[0, 2].hist2d(x, y, bins=bins, norm="log", cmap='Blues')
        
        # Only show squares with at least 10 observations
        hist[hist < 10] = jnp.nan
        axs[0, 2].imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='Blues', norm="log")

        #TLS:
        #-----------------------------------
        # from sklearn.linear_model import LinearRegression

        
        # # Perform linear regression using TLS
        # reg = LinearRegression().fit(jnp.log(x[x > 0]).reshape(-1, 1), jnp.log(y[x > 0]))
        # slope = reg.coef_[0]
        # intercept = reg.intercept_
        # print(F"tls slope: {slope}, intercept: {intercept}")

        # # Plot TLS estimator line
        # x_fit = jnp.linspace(jnp.log(min(x[x > 0])), jnp.log(max(x)), 100)
        # y_fit = slope * x_fit + intercept
        # axs[0, 2].plot(jnp.exp(x_fit), jnp.exp(y_fit), 'r-', label=f'TLS Slope: {slope:.2f}')
        #-----------------------------------
        axs[0, 2].set_xscale('log')
        axs[0, 2].set_yscale('log')
        #x and y min are 1
        axs[0, 2].set_xlim(left=1)
        axs[0, 2].set_ylim(bottom=1)
        axs[0, 2].set_title('2D Histogram of In-Degrees and Out-Degrees')
        axs[0, 2].set_xlabel('Out-Degrees (log scale)')
        axs[0, 2].set_ylabel('In-Degrees (log scale)')
        # axs[0, 2].legend()

        # QQ plot for out-degrees against lognormal
        stats.probplot(self.out_degrees, dist="lognorm", sparams=(jnp.std(jnp.log(self.out_degrees)),), plot=axs[1, 0])
        axs[1, 0].set_title('QQ Plot of Out-Degrees vs Lognormal')

        # QQ plot for out-degrees against pareto
        stats.probplot(self.out_degrees, dist="pareto", sparams=(1.59,), plot=axs[1, 1])
        axs[1, 1].set_title('QQ Plot of Out-Degrees vs Pareto (1.59)')

        # QQ plot for in-degrees against lognormal
        # print(f"self.in_degrees: {self.in_degrees[:10]}")
        stats.probplot(self.in_degrees, dist="lognorm", sparams=(jnp.std(jnp.log(self.in_degrees)),), plot=axs[2, 0])
        axs[2, 0].set_title('QQ Plot of In-Degrees vs Lognormal')

        # QQ plot for in-degrees against pareto
        stats.probplot(self.in_degrees, dist="pareto", sparams=(2.38,), plot=axs[2, 1])
        axs[2, 1].set_title('QQ Plot of In-Degrees vs Pareto (2.38)')

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
        """
        Save the degree sequences to a file.

        Args:
            save_path (str): Path to save the degree sequences.
        """
        with open(save_path, 'w') as f:
            f.write('# kappa_out  kappa_in\n')
            for i in range(self.n):
                f.write(f'{self.out_degrees[i]}  {self.in_degrees[i]}\n')
        
        if self.verbose:
            print(f'Degree sequence saved to {save_path}')


if __name__ == "__main__":
    # Ecuador 2015 example parameters
    exp_mu = jnp.array([39.06, 39.06])
    #out, in
    Sigma = jnp.array([[2.89, 1.37], [1.37, 2.06]])
    # mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    key = random.PRNGKey(0)
    transition_point = 0
    tail_indices = [1.59, 2.38] # out , in
    #exp_sigma, as Sigma is that of the lognormal distribution
    # exp_Sigma_11 = (jnp.exp(mu[0] + 1/2*Sigma[0,0]))**2 * (jnp.exp(Sigma[0, 0]) - 1)
    # exp_Sigma_22 = (jnp.exp(mu[1] + 1/2*Sigma[1,1]))**2 * (jnp.exp(Sigma[1, 1]) - 1)
    # exp_Sigma_12 = (jnp.exp(mu[0] + 1/2*Sigma[0,0])) * (jnp.exp(mu[1] + 1/2*Sigma[1,1])) * (jnp.exp(Sigma[0, 1]) - 1)
    # exp_Sigma_21 = exp_Sigma_12

    # exp_Sigma = jnp.array([[exp_Sigma_11, exp_Sigma_12], [exp_Sigma_21, exp_Sigma_22]])/4000
    # print(f"exp_Sigma: {exp_Sigma}")
    scale_2 = exp_mu[1] * (tail_indices[1] - 2) / (tail_indices[1] - 1)
    scale_1 = scale_2

    degree_seq = DegreeSeq(n=86345, verbose=True)
    out_degrees, in_degrees = degree_seq.generate(key,
                                                  bivariate_lognormal_pareto2,
                                                  exp_mu=exp_mu,
                                                  Sigma=Sigma,
                                                  tail_indices=tail_indices,
                                                  transition_point=transition_point,
                                                  scale_1=scale_1,
                                                  scale_2=scale_2,
                                                  capula="gaussian",
                                                  capula_cov_matrix=jnp.array([[1,-0.999],[-0.999,1]]))
    degree_seq.plot('degree_sequence_mixture.png')
    # degree_seq.save('degree_sequence_mixture.txt')
