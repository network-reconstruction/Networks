# from jax.scipy.special import gammaln, logsumexp

# def gamma_negative(x):
#     """Determine the sign of gamma function."""
#     return jnp.sign(jnp.sin(jnp.pi * x)) == -1

# Future work: [3] M. Abramowitz, I. Stegun. Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables.
# def hyp2f1_z_near_one(a, b, c, z, n_terms=20):
#     """Compute 2F1(a, b, c, z) when z is near 1."""
#     d = c - a - b

#     log_first_coefficient = (gammaln(c) + gammaln(d) -
#                              gammaln(c - a) - gammaln(c - b))
    
#     sign_first_coefficient = (
#         gamma_negative(c) ^  gamma_negative(d) ^
#         gamma_negative(c - a) ^ gamma_negative(c - b))
#     sign_first_coefficient = -2. * float(sign_first_coefficient) + 1.

#     log_second_coefficient = (
#         jnp.log1p(-z) * d +
#         gammaln(c) + gammaln(-d) -
#         gammaln(a) - gammaln(b))

#     sign_second_coefficient = (
#         gamma_negative(c) ^ gamma_negative(a) ^ gamma_negative(b) ^
#         gamma_negative(-d))
#     sign_second_coefficient = -2. * float(sign_second_coefficient) + 1.

#     first_term = hyp2f1_expansion(a, b, 1 - d, 1 - z, n_terms)
#     second_term = hyp2f1_expansion(c - a, c - b, d + 1., 1 - z, n_terms)
    
#     log_first_term = log_first_coefficient + jnp.log(jnp.abs(first_term))
#     log_second_term = log_second_coefficient + jnp.log(jnp.abs(second_term))

#     sign_first_term = sign_first_coefficient * jnp.sign(first_term)
#     sign_second_term = sign_second_coefficient * jnp.sign(second_term)
    
#     log_diff = jnp.log(jnp.abs(jnp.exp(log_first_term) - jnp.exp(log_second_term)))
#     sign_log_diff = jnp.sign(jnp.exp(log_first_term) - jnp.exp(log_second_term))
    
#     sign = jnp.where(
#         sign_first_term == sign_second_term,
#         sign_first_term,
#         sign_first_term * sign_log_diff)
    
#     log_result = jnp.where(
#         sign_first_term == sign_second_term,
#         logsumexp(jnp.array([log_first_term, log_second_term])),
#         log_diff)
    
#     return jnp.exp(log_result) * sign

# def hyp2f1_approx(a: float, b: float, c: float, z: float, n_terms: int =20) -> float:
#     """
#     Compute the hypergeometric function _2F_1(a, b; c; z) for a given z, handling
#     special cases based on the value of z.
#     """
#     # if z > 0.9:
#     #     print("z > 0.9")
#     #     return hyp2f1_z_near_one(a, b, c, z, n_terms)
#     # else:
#     return hyp2f1_expansion(a, b, c, z, n_terms)