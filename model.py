# happyleaf
# model.py

import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
import math
import modeltemplates as mt

# Hyperparameters

lat_sqrt = 4
lat = lat_sqrt ** 2
heads = 8
tlayers = 3
num_categories = 39

# Model functions

def generate_params(key):
    """Return object with model parameters and dropout infos"""
    params, do_infos = {}, {}
    key, p1_key, p2_key, p3_key, = jax.random.split(key, num = 4)
    params["p1"] = mt.perceptron_layer_init(p1_key, 3, lat)
    for j in range(1, 3):
        for i in range(0, tlayers):
            k = 8
            do_infos[f"t{j}-{i}"] = {
                "do_proj": [key, 0.1, True],
                "do_res": [key, 0.1, True] }
            key, t_key = jax.random.split(key)
            params[f"t{j}-{i}"] = mt.spatial_transformer_block_init(t_key, k, lat, heads)
    params["p2"] = mt.perceptron_layer_init(p2_key, 8 * 8 * lat, lat)
    params["p3"] = mt.perceptron_layer_init(p3_key, 8 * 8 * lat, num_categories, var = 1e-4)
    for i in range(1, 4):
        # The actual value of the key does not matter here
        do_infos[f"p{i}"] = { "do_info": [key, 0.3, True] }
    return key, params, do_infos

def forward(params, x, do_infos):
    """Forward method"""
    x = mt.perceptron_layer(x, **params["p1"], **do_infos["p1"], nonlinear = False, ln = False)
    x = mt.pool(x, 8) # [8, 8, 8, 8, lat]
    x += mt.spatial_embed(8, 8, 8, lat_sqrt)
    # Transformer on 8 by 8 patches (in parallel)
    for i in range(0, tlayers):
        x = mt.spatial_transformer_block(x, **params[f"t1-{i}"], **do_infos[f"t1-{i}"])
    x = jnp.reshape(x, [8, 8, 8 * 8 * lat])
    x = mt.perceptron_layer(x, **params["p2"], **do_infos["p2"], nonlinear = False)
    x = mt.pool(x, 8) # [1, 1, 8, 8, lat]
    x += mt.spatial_embed(1, 1, 8, lat_sqrt)
    # Transformer on 8 by 8 patch
    for i in range(0, tlayers):
        x = mt.spatial_transformer_block(x, **params[f"t2-{i}"], **do_infos[f"t2-{i}"])
    x = jnp.reshape(x, [8 * 8 * lat])
    # Perceptron layers to logits
    x = mt.perceptron_layer(x, **params["p3"], **do_infos["p3"], nonlinear = False, ln = False)
    return x

def forward_loss(params, x, data_index, do_infos):
    """Forward loss method"""
    return mt.cross_entropy(forward(params, x, do_infos), data_index)    

@jit
def forward_batched_s(params, xs, do_infos):
    """Static atched forward method"""
    return vmap(forward, (None, 0, None))(params, xs, do_infos)

@jit
def forward_loss_batched_s(params, xs, data_indices, do_infos):
    """Static batched loss method"""
    losses = vmap(forward_loss, (None, 0, 0, None))(params, xs, data_indices, do_infos)
    return jnp.sum(losses) / losses.shape[0]

@jit
def forward_classify_batched_s(params, xs, data_indices, do_infos):
    """Static batched correct classification computation"""
    argmaxs = jnp.argmax(forward_batched_s(params, xs, do_infos), axis = 1)
    return jnp.sum(data_indices[jnp.arange(argmaxs.shape[0]), argmaxs]) / argmaxs.shape[0]

@jit
def backward_s(params, xs, data_indices, do_infos):
    """Static batched backward pass"""
    return grad(forward_loss_batched_s)(params, xs, data_indices, do_infos)

# Model definition

class MainModel:
    def __init__(self):
        self.key = jax.random.PRNGKey(42)
        self.params = {}
        self.velocity = {}
        self.velocity_sq = {}
        self.gradstep_counter = 0
        self.do_infos = {}

        # Adam parameters
        self.discount = 0.9
        self.adam_lr = 1e-4
        self.adam_weight_decay = 0.1
        self.adam_alpha = 0.9
        self.adam_beta = 0.99
        self.adam_eps = 1e-8
        self.adam_clip = 10.0

        self.key, self.params, self.do_infos = generate_params(self.key)
    
    # Dropout reinit function
    
    def reinit_do_infos(self, infer = True):
        """Generate new keys for all dropout functions"""
        for k in self.do_infos:
            for kn in self.do_infos[k]:
                self.key, self.do_infos[k][kn][0] = jax.random.split(self.key)
                self.do_infos[k][kn][2] = infer

    # Reinit dropout and run jit compiled model

    def forward_batched(self, xs):
        """Batched forward method"""
        self.reinit_do_infos()
        return forward_batched_s(self.params, xs, self.do_infos)

    def forward_loss_batched(self, xs, data_indices):
        """Batched loss method"""
        self.reinit_do_infos()
        return forward_loss_batched_s(self.params, xs, data_indices, self.do_infos)

    def forward_classify_batched(self, xs, data_indices):
        """Batched correct classification computation"""
        self.reinit_do_infos()
        return forward_classify_batched_s(self.params, xs, data_indices, self.do_infos)

    def backward(self, xs, data_indices):
        """Batched backward pass"""
        self.reinit_do_infos(infer = False)
        return backward_s(self.params, xs, data_indices, self.do_infos)

    def adamw_update(self, xs, data_indices):
        """Gradient update using AdamW optimizer"""
        lr = self.adam_lr
        wd = self.adam_weight_decay
        alpha = self.adam_alpha
        beta = self.adam_beta
        eps = self.adam_eps
        clip = self.adam_clip

        for k in self.params:
            for kn in self.params[k]:
                self.params[k][kn] -= lr * wd * self.params[k][kn]
        
        new_grad = self.backward(xs, data_indices)
        grad_norm = 0.0

        for k in self.params:
            for kn in self.params[k]:
                grad_norm += jnp.sum(new_grad[k][kn] ** 2)
        grad_norm = jnp.sqrt(grad_norm)

        for k in self.params:
            if self.gradstep_counter == 0:
                self.velocity[k] = {}
                self.velocity_sq[k] = {}
            for kn in self.params[k]:
                new_grad[k][kn] = jnp.where(grad_norm > clip, clip / grad_norm, 1.0) * new_grad[k][kn]
                if self.gradstep_counter == 0:
                    self.velocity[k][kn] = jnp.zeros_like(new_grad[k][kn])
                    self.velocity_sq[k][kn] = jnp.zeros_like(new_grad[k][kn])
                self.velocity[k][kn] = alpha * self.velocity[k][kn] + (1 - alpha) * new_grad[k][kn]
                self.velocity_sq[k][kn] = beta * self.velocity_sq[k][kn] \
                    + (1 - beta) * jnp.power(new_grad[k][kn], 2.0)
                norm_velocity = self.velocity[k][kn] / (1 - jnp.power(alpha, self.gradstep_counter + 1.0))
                norm_velocity_sq = self.velocity_sq[k][kn] / \
                    (1 - jnp.power(beta, self.gradstep_counter + 1.0))
                self.params[k][kn] -= lr * norm_velocity / (eps + jnp.sqrt(norm_velocity_sq))
                # self.params[k][kn] -= lr * new_grad[k][kn]
        self.gradstep_counter += 1
    
    def total_params(self):
        """Number of floating point parameters of model"""
        count = 0
        for k in self.params:
            for kn in self.params[k]:
                count += math.prod(self.params[k][kn].shape)
        return count
    
    def update_loss_and_classify(self, arr, xs, data_indices):
        """Append running loss and classification to arr"""
        loss = self.discount * arr[-1][0] + \
            (1 - self.discount) * self.forward_loss_batched(xs, data_indices)
        classify = self.discount * arr[-1][1] + \
            (1 - self.discount) * self.forward_classify_batched(xs, data_indices)
        arr.append([loss, classify])
