# happyleaf
# modeltemplates.py

import jax.numpy as jnp
from jax import vmap
import jax

# Standard neural network functions

def cross_entropy(pred_logit, data_index):
    return -jnp.sum(jax.nn.log_softmax(pred_logit) * data_index)

def layer_norm(x, mu, sigma, eps = 1e-5):
    return mu + sigma * (x - jnp.mean(x)) / jnp.sqrt(eps + jnp.var(x))

def linear_layer(x, w, b):
    return jnp.tensordot(x, w, axes = 1) + b

def perceptron_layer(x, w, b, mu, sigma, do_info, nonlinear = True, ln = True):
    y = linear_layer(x, w, b)
    y = jnp.where(ln, layer_norm(y, mu, sigma), y)
    return jnp.where(nonlinear, dropout(leaky_relu(y), do_info), y)

def perceptron_layer_init(key, in_dim, out_dim, var = 1.0):
     w_key, b_key = jax.random.split(key)
     return {
        "w": jnp.sqrt(var / in_dim) * jax.random.normal(w_key, [in_dim, out_dim]),
        "b": jnp.sqrt(var) * jax.random.normal(b_key, [out_dim]),
        "mu": jnp.zeros([out_dim]),
        "sigma": jnp.ones([out_dim]) }

def dropout(x, do_info):
    [key, prob, infer] = do_info
    return jnp.where(infer, x, x * jax.random.bernoulli(key, 1 - prob, x.shape) / (1 - prob))

def leaky_relu(x, lower_slope = 0.01):
    """Normalized leaky relu"""
    mu = (1 - lower_slope) / jnp.sqrt(2 * jnp.pi)
    var = (1 + lower_slope ** 2) / 2 - mu ** 2
    return (jnp.maximum(lower_slope * x, x) - mu) / jnp.sqrt(var)

def spatial_embed(width, height, k, lat_sqrt):
    """Computes sinusoidal embeddding (same for different patches)"""
    # Output shape is [width, height, k, k, lat_sqrt ** 2]
    alpha = jnp.power(k, 1.0 / lat_sqrt)
    u, v = lat_sqrt // 2, lat_sqrt - lat_sqrt // 2
    sin_ent = jnp.tensordot(jnp.arange(k), 2 * jnp.pi / jnp.power(alpha, jnp.arange(u)), axes = 0)
    cos_ent = jnp.tensordot(jnp.arange(k), 2 * jnp.pi / jnp.power(alpha, jnp.arange(v)), axes = 0)
    x = jnp.concatenate([jnp.sin(sin_ent), jnp.cos(cos_ent)], axis = 1) # [k, lat_sqrt]
    out = jnp.swapaxes(jnp.tensordot(x, x, axes = 0), 1, 2) # [k, k, lat_sqrt, lat_sqrt]
    out = jnp.reshape(out, [k, k, lat_sqrt ** 2])
    return jnp.broadcast_to(out, [width, height, k, k, lat_sqrt ** 2])

def pool(x, k):
    """Pooling layer"""
    # [width, height, lat] = x.shape
    # Output shape is [width // k, height // k, k, k, lat]
    patches = jnp.reshape(x, [x.shape[0] // k, k, x.shape[1] // k, k, x.shape[2]])
    return jnp.moveaxis(patches, 1, 2)

def single_head_attention(x, w_query, w_key, w_val):
    """Single spatial attention head"""
    # [k * k, old_lat] = x.shape
    # [old_lat, middle_lat] = w_query.shape = w_key.shape
    # [old_lat, final_lat] = w_val.shape
    # Output shape is [k * k, final_lat]
    q = x @ w_query
    k = x @ w_key
    v = x @ w_val
    exps = q @ jnp.transpose(k) / jnp.sqrt(w_query.shape[1])
    smax = jax.nn.softmax(exps, 1)
    return smax @ v

def spatial_multiattention(x, w_queries, w_keys, w_vals, w_concat, do_proj):
    """Spatial multihead attention layer"""
    # [k, k, old_lat] = x.shape
    # [old_lat, middle_lat, head_count] = w_queries.shape = w_keys.shape
    # [old_lat, final_lat, head_count] = w_vals.shape
    # [final_lat * head_count, out_lat] = w_concat.shape
    # Output shape is [k, k, out_lat]
    x_reshaped = jnp.reshape(x, [-1, x.shape[2]])
    val = vmap(single_head_attention, (None, 2, 2, 2), 2)(x_reshaped, w_queries, w_keys, w_vals)
    concat_val = dropout(jnp.reshape(val, [-1, w_vals.shape[1] * w_vals.shape[2]]), do_proj)
    return jnp.reshape(jnp.tensordot(concat_val, w_concat, axes = 1), list(x.shape[:-1]) + [-1])

def spatial_transformer_block(x, mu_in, sigma_in,
        w_queries, w_keys, w_vals, w_concat,
        w, b, mu, sigma, do_proj, do_res):
    """Applies spatial transformer block to each k times k patch individually"""
    # [width, height, k, k, lat] = x.shape
    # [k, k, lat] = mu_in.shape
    # [k, k, lat] = sigma_in.shape
    # [lat, middle_lat, head_count] = w_queries.shape = w_keys.shape
    # [lat, final_lat, head_count] = w_vals.shape
    # [final_lat * head_count, out_lat] = w_concat.shape
    # [lat, lat] = w.shape
    # [lat] = b.shape
    # [lat] = mu.shape
    # [lat] = sigma.shape
    # Output shape is [width, height, k, k, lat]
    def local_stp(xs):
        ma_in = layer_norm(xs, mu_in, sigma_in)
        xs = xs + spatial_multiattention(ma_in, w_queries, w_keys, w_vals, w_concat, do_proj)
        xs = xs + perceptron_layer(xs, w, b, mu, sigma, do_res)
        return xs
    y = jnp.reshape(x, [-1] + list(x.shape[2:]))
    return jnp.reshape(vmap(local_stp, 0)(y), x.shape)

def spatial_transformer_block_init(key, k, lat, heads):
    w_queries_key, w_keys_key, w_vals_key, w_concat_key, w_key, b_key = jax.random.split(key, num = 6)
    return {
        "mu_in": jnp.zeros([k, k, lat]),
        "sigma_in": jnp.ones([k, k, lat]),
        "w_queries": jax.random.normal(w_queries_key, [lat, lat, heads]),
        "w_keys": jax.random.normal(w_keys_key, [lat, lat, heads]),
        "w_vals": jnp.sqrt(1 / lat) * jax.random.normal(w_vals_key, [lat, lat, heads]),
        "w_concat": jnp.sqrt(1 / (lat * heads)) * jax.random.normal(w_concat_key, [lat * heads, lat]),
        "w": jnp.sqrt(1 / heads) * jax.random.normal(w_key, [lat, lat]),
        "b": jax.random.normal(b_key, [lat]),
        "mu": jnp.zeros([lat]),
        "sigma": jnp.ones([lat]) }
