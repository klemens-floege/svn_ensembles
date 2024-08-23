import jax
import jax.numpy as jnp
from jax import lax



@jax.jit
def jitted_compute_bias_KFACx(padded_eigenvalues, padded_eigenvectors, padded_partitions):
    def apply_kron_eigen_decomp(eigenvalues, eigenvectors, x_partition):
        V = eigenvectors
        lambda_ = eigenvalues
        X = x_partition
        result = jnp.dot(V, jnp.multiply(lambda_, jnp.dot(V.T, X)))
        return result.ravel()  # Ensure it's a 1D vector
    
        """
        return lax.cond(
            block_type == 0,
            kronecker_block,
            bias_block,
            (eigenvalues, eigenvectors, x_partition)
        )"""
    
    results = jax.vmap(apply_kron_eigen_decomp)(
        padded_eigenvalues,
        padded_eigenvectors,
        padded_partitions,
    )
    return jnp.concatenate(results)



@jax.jit
def jitted_compute_kron_KFACx(padded_eigenvalues, padded_eigenvectors, padded_partitions):
    def apply_kron_eigen_decomp(eigenvalues, eigenvectors, x_partition):
        
        V_Q, V_K = eigenvectors
        lambda_Q, lambda_K = eigenvalues
        X = x_partition.reshape(V_Q.shape[0], V_K.shape[1])
        result = jnp.dot(V_K, jnp.dot(X.T, V_Q.T))
        result = jnp.dot(jnp.dot(V_K.T, result), V_Q)
        result = jnp.multiply(result, jnp.outer(lambda_K, lambda_Q))
        return result.flatten()
            
    results = jax.vmap(apply_kron_eigen_decomp)(
        padded_eigenvalues,
        padded_eigenvectors,
        padded_partitions
    )
    return jnp.concatenate(results)