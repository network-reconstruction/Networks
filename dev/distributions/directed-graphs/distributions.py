import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

def bivariate_lognormal(key, n, exp_mu, Sigma):
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))
    return samples[:, 1], samples[:, 0]

def pareto_samples(key, n, tail, scale=1.0):
    return scale * (random.uniform(key, (n,)) ** (-1 / tail))

def bivariate_lognormal_pareto(key, n, exp_mu, Sigma, tail_indices, weights):
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    lognormal_samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))

    key1, key2 = random.split(key)
    
    pareto_samples_1 = pareto_samples(key1, n, tail_indices[0])
    pareto_samples_2 = pareto_samples(key2, n, tail_indices[1])
    pareto_samples_combined = jnp.column_stack((pareto_samples_1, pareto_samples_2))

    mixed_samples = (weights[0] * lognormal_samples + weights[1] * pareto_samples_combined)

    return mixed_samples[:, 1], mixed_samples[:, 0]

def bivariate_lognormal_pareto2(key, n, exp_mu, Sigma, tail_indices, transition_point):
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    lognormal_samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))

    key1, key2 = random.split(key)
    
    pareto_samples_1 = pareto_samples(key1, n, tail_indices[0])
    pareto_samples_2 = pareto_samples(key2, n, tail_indices[1])
    pareto_samples_combined = jnp.column_stack((pareto_samples_1, pareto_samples_2))

    # Define a weighting function that smoothly transitions from lognormal to Pareto
    def weighting_function(x, transition_point):
        return 1 / (1 + jnp.exp((x - transition_point) / transition_point))

    weights_log = weighting_function(lognormal_samples, transition_point)
    weights_pareto = 1 - weights_log

    # Scale Pareto samples to match the expected mean
    scaled_pareto_samples = pareto_samples_combined * (exp_mu / jnp.mean(pareto_samples_combined, axis=0))

    mixed_samples = (weights_log * lognormal_samples + weights_pareto * scaled_pareto_samples)

    return mixed_samples[:, 1], mixed_samples[:, 0]

def frechet_samples(key, n, alpha, scale=1.0):
    return scale / (random.uniform(key, (n,)) ** (1 / alpha))

def gumbel_samples(key, n, loc=0.0, scale=1.0):
    return loc - scale * jnp.log(-jnp.log(random.uniform(key, (n,))))

def bivariate_frechet(key, n, alpha, scale=1.0):
    key1, key2 = random.split(key)
    frechet_samples_1 = frechet_samples(key1, n, alpha[0], scale)
    frechet_samples_2 = frechet_samples(key2, n, alpha[1], scale)
    return frechet_samples_1, frechet_samples_2

def bivariate_gumbel(key, n, loc, scale=1.0):
    key1, key2 = random.split(key)
    gumbel_samples_1 = gumbel_samples(key1, n, loc[0], scale)
    gumbel_samples_2 = gumbel_samples(key2, n, loc[1], scale)
    return gumbel_samples_1, gumbel_samples_2
