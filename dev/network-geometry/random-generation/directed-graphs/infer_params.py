import jax
import jax.numpy as jnp
from jax.scipy.special import betainc
import numpy as np

class InferenceAlgorithm:
    """
    Class to perform parameter inference for the directed-reciprocal S1 model.
    
    Attributes:
        k_in (jax.numpy.ndarray): Observed in-degrees.
        k_out (jax.numpy.ndarray): Observed out-degrees.
        reciprocity_obs (float): Observed reciprocity.
        triangle_density_obs (float): Observed density of triangles.
        tol (float): Tolerance for convergence. Default is 0.01.
        max_iter (int): Maximum number of iterations. Default is 100.
    """

    def __init__(self, k_in, k_out, reciprocity_obs, triangle_density_obs, tol=0.01, max_iter=100):
        self.k_in = jnp.array(k_in)
        self.k_out = jnp.array(k_out)
        self.reciprocity_obs = reciprocity_obs
        self.triangle_density_obs = triangle_density_obs
        self.tol = tol
        self.max_iter = max_iter
        self.N = len(k_in)
        self.avg_degree = jnp.mean(k_in)
        self.beta = 1 + jax.random.uniform(jax.random.PRNGKey(0), shape=(), minval=0, maxval=1)
        self.mu = self.compute_mu(self.beta, self.avg_degree)

    @staticmethod
    def compute_mu(beta, avg_degree):
        """
        Compute the parameter mu.
        
        Args:
            beta (float): Parameter beta.
            avg_degree (float): Average degree.
        
        Returns:
            float: Computed mu value.
        """
        return beta * jnp.sin(jnp.pi / beta) / (2 * jnp.pi * avg_degree)

    @staticmethod
    def expected_degrees(kappa_in, kappa_out, beta, mu, N):
        """
        Compute the expected in-degrees and out-degrees.
        
        Args:
            kappa_in (jax.numpy.ndarray): Hidden in-degrees.
            kappa_out (jax.numpy.ndarray): Hidden out-degrees.
            beta (float): Parameter beta.
            mu (float): Parameter mu.
            N (int): Number of nodes.
        
        Returns:
            tuple: Expected in-degrees and out-degrees.
        """
        hypergeom = jax.vmap(lambda kappa_in, kappa_out: betainc(1 / beta, 1, 1 / (1 + N / (2 * mu * kappa_out * kappa_in))))
        expected_in = jnp.sum(hypergeom(kappa_in, kappa_out))
        expected_out = jnp.sum(hypergeom(kappa_out, kappa_in))
        return expected_in, expected_out

    def infer_hidden_degrees(self):
        """
        Infer the hidden in-degrees and out-degrees.
        
        Returns:
            tuple: Inferred hidden in-degrees and out-degrees.
        """
        kappa_in = self.k_in
        kappa_out = self.k_out
        for _ in range(self.max_iter):
            expected_in, expected_out = self.expected_degrees(kappa_in, kappa_out, self.beta, self.mu, self.N)
            deviation = jnp.max(jnp.abs(expected_in - self.k_in) + jnp.abs(expected_out - self.k_out))
            if deviation < self.tol:
                break
            kappa_in += (self.k_in - expected_in) * jax.random.uniform(jax.random.PRNGKey(_), shape=self.k_in.shape)
            kappa_out += (self.k_out - expected_out) * jax.random.uniform(jax.random.PRNGKey(_), shape=self.k_out.shape)
        return kappa_in, kappa_out

    def infer_nu(self, kappa_in, kappa_out):
        """
        Infer the parameter nu.
        
        Args:
            kappa_in (jax.numpy.ndarray): Hidden in-degrees.
            kappa_out (jax.numpy.ndarray): Hidden out-degrees.
        
        Returns:
            float: Inferred nu value.
        """
        def reciprocity(nu):
            if nu == 0:
                return 0.5
            elif nu == 1:
                return 1.0
            elif nu == -1:
                return 0.0
            return (1 + nu) * 0.5 - nu * 0.0 if nu <= 0 else (1 - nu) * 0.5 + nu * 1.0

        nu = (self.reciprocity_obs - reciprocity(0)) / (reciprocity(1) - reciprocity(0)) if self.reciprocity_obs > reciprocity(0) else (self.reciprocity_obs - reciprocity(0)) / (reciprocity(-1) + reciprocity(0))
        return nu

    def estimate_triangle_density(self, kappa_in, kappa_out, nu):
        """
        Estimate the expected density of triangles.
        
        Args:
            kappa_in (jax.numpy.ndarray): Hidden in-degrees.
            kappa_out (jax.numpy.ndarray): Hidden out-degrees.
            nu (float): Parameter nu.
        
        Returns:
            float: Estimated density of triangles.
        """
        def triangle_prob(kappa1, kappa2, theta12):
            return 1 / (1 + (self.N / (2 * self.mu * kappa1 * kappa2)) ** self.beta)

        def sample_triangle_density(kappa_in, kappa_out):
            key = jax.random.PRNGKey(0)
            k1 = jax.random.choice(key, len(kappa_in))
            k2 = jax.random.choice(key, len(kappa_out))
            theta12 = jax.random.uniform(key, shape=(1,), minval=0, maxval=2 * jnp.pi)
            return triangle_prob(kappa_in[k1], kappa_out[k2], theta12)
        
        triangle_densities = jax.vmap(sample_triangle_density, in_axes=(None, None))(kappa_in, kappa_out)
        return jnp.mean(triangle_densities)

    def run_inference(self):
        """
        Run the full inference algorithm.
        
        Returns:
            tuple: Inferred hidden degrees, beta, and nu.
        """
        for _ in range(self.max_iter):
            kappa_in, kappa_out = self.infer_hidden_degrees()
            nu = self.infer_nu(kappa_in, kappa_out)
            estimated_triangle_density = self.estimate_triangle_density(kappa_in, kappa_out, nu)
            
            if jnp.abs(estimated_triangle_density - self.triangle_density_obs) < self.tol:
                break

            beta_min, beta_max = 1, 25
            if estimated_triangle_density > self.triangle_density_obs:
                beta_max = self.beta
            else:
                beta_min = self.beta

            self.beta = beta_min + (beta_max - beta_min) * (self.triangle_density_obs - estimated_triangle_density) / (estimated_triangle_density - self.triangle_density_obs)
            self.mu = self.compute_mu(self.beta, self.avg_degree)
            
        return kappa_in, kappa_out, self.beta, nu

    @classmethod
    def from_file(cls, filepath, reciprocity_obs, triangle_density_obs, tol=0.01, max_iter=100):
        """
        Create an InferenceAlgorithm instance from a file.
        
        Args:
            filepath (str): Path to the file containing the degree sequences.
            reciprocity_obs (float): Observed reciprocity.
            triangle_density_obs (float): Observed density of triangles.
            tol (float): Tolerance for convergence. Default is 0.01.
            max_iter (int): Maximum number of iterations. Default is 100.
        
        Returns:
            InferenceAlgorithm: Initialized instance.
        """
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] != 2:
                raise ValueError("File format error: Each line must contain two numerical values separated by space or tab.")
            k_out, k_in = data[:, 0], data[:, 1]
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
        
        return cls(k_in, k_out, reciprocity_obs, triangle_density_obs, tol, max_iter)


if __name__ == "__main__":
    #Ecuador 2015
    # Example usage:
    filepath = "degree_sequence.txt"
    reciprocity_obs = .46  # Example observed reciprocity
    triangle_density_obs = 0.28  # Example observed triangle density

    # Create an instance of the InferenceAlgorithm from file
    inference_algo = InferenceAlgorithm.from_file(filepath, reciprocity_obs, triangle_density_obs)

    # Run the inference algorithm
    hidden_degrees, beta, nu = inference_algo.run_inference()
    print(f"Hidden degrees: {hidden_degrees}")
    print(f"Beta: {beta}")
    print(f"Nu: {nu}")
