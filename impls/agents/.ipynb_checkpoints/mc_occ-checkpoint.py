import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules, decoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField
from functools import partial

class MCOccAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss_mc(self, batch, grad_params, rng): 
        """Compute the FQL actor loss."""

        ######################
        if self.config['encoder'] is not None:        
            future_observations = jax.lax.stop_gradient(self.network.select('critic_vf_encoder')\
                                                    (batch['future_observations'], params=grad_params))
            observations = jax.lax.stop_gradient(self.network.select('critic_vf_encoder')\
                                                    (batch['observations'], params=grad_params))
        else:
            future_observations = batch['future_observations']
            observations = batch['observations']
        ######################
        
        if self.config['future_prediction']:
            batch_size, num_samples, observation_dim = future_observations.shape
            sf = future_observations


            
        sf = jnp.reshape(sf, (batch_size * num_samples, observation_dim))
        rng, x_rng, t_rng, noise_rng = jax.random.split(rng, 4)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size * num_samples, observation_dim))
        x_1 = sf
        t = jax.random.uniform(t_rng, (batch_size * num_samples, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0


        if self.config['state_action']:
            obs = jnp.concatenate([observations, batch['actions']], axis=-1)
        else:
            obs = observations

        obs = jnp.reshape(jnp.repeat(jnp.expand_dims(obs, axis=-2), num_samples, axis=-2), (-1, obs.shape[-1]))
        pred = self.network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Total loss.
        actor_loss = bc_flow_loss 
        return actor_loss, {
            'actor_loss': actor_loss,
        }

    def encoder_loss(self, batch, grad_params): 
        """Compute the encoder loss."""

        ######################   
        features = self.network.select('actor_bc_flow_encoder')(batch['observations'], params=grad_params)
        pred_observations = self.network.select('actor_bc_flow_decoder')(features, params=grad_params)
        encoder_loss = jnp.mean((pred_observations - batch['observations']) ** 2)

        return encoder_loss, {
            'encoder_loss': encoder_loss,
        }
        
    def actor_loss_td(self, batch, grad_params, rng): 
        """Compute the FQL actor loss."""

        num_samples = 1

        ######################
        if self.config['encoder'] is not None:        
            future_observations = jax.lax.stop_gradient(self.network.select('critic_vf_encoder')\
                                                    (batch['future_observations'], params=grad_params))
            observations = jax.lax.stop_gradient(self.network.select('critic_vf_encoder')\
                                                    (batch['observations'], params=grad_params))
            next_observations = jax.lax.stop_gradient(self.network.select('critic_vf_encoder')\
                                                     (batch['next_observations'], params=grad_params))

        else:
            future_observations = batch['future_observations']
            observations = batch['observations']
            next_observations = batch['next_observations']
        ######################

        
        if self.config['future_prediction']:
            batch_size, observation_dim = next_observations.shape
            if self.config['state_action']:
                s_ = jnp.concatenate([next_observations, batch['next_actions']], axis=-1)
                extra_dim = batch['next_actions'].shape[-1]
            else:
                s_ = next_observations 
                extra_dim = 0
            s__ = jnp.reshape(jnp.repeat(jnp.expand_dims(s_, axis=-2), num_samples, axis=-2), (-1, extra_dim + observation_dim))
            
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        
        # BC flow loss for next/prev state.
        x_0 = jax.random.normal(x_rng, (batch_size * num_samples, observation_dim))
        x_1 = s__[:, :observation_dim]
        t = jax.random.uniform(t_rng, (batch_size * num_samples, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        if self.config['state_action']:
            obs = jnp.concatenate([observations, batch['actions']], axis=-1)
        else:
            obs = observations

        obs = jnp.reshape(jnp.repeat(jnp.expand_dims(obs, axis=-2), num_samples, axis=-2), (-1, obs.shape[-1]))
        pred_s_ = self.network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)
        bc_flow_loss_s_ = (pred_s_ - vel) ** 2

        # BC flow loss for future states
        x_0 = jax.random.normal(x_rng, (batch_size * num_samples, observation_dim))
        x_1 = jax.lax.stop_gradient(self.compute_flow_sf(x_0, s__, use_target = True))
        x_t = (1 - t) * x_0 + t * x_1
        vel = self.network.select('target_actor_bc_flow')(s__, x_t, t, params=grad_params)
        #vel = x_1 - x_0

        pred_sf = self.network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)
        bc_flow_loss_sf = (pred_sf - jax.lax.stop_gradient(vel)) ** 2       

        # Total loss.
        actor_loss = (1 - self.config['discount'])*bc_flow_loss_s_ + self.config['discount']*bc_flow_loss_sf
        actor_loss = jnp.mean(actor_loss)

        return actor_loss, {
            'actor_loss': actor_loss,
        }
        
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, td_training = False): 
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng, 2)
        def true_fun(_):
            return self.actor_loss_td(batch, grad_params, actor_rng)
        def false_fun(_):
            return self.actor_loss_mc(batch, grad_params, actor_rng)
        
        if self.config['encoder_only']:
            encoder_loss, encoder_info = self.encoder_loss(batch, grad_params)
            for k, v in encoder_info.items():
                info[f'encoder/{k}'] = v
            loss = encoder_loss
        else:
            #actor_loss, actor_info = self.actor_loss_td(batch, grad_params, actor_rng) 
            actor_loss, actor_info = jax.lax.cond(td_training, true_fun, false_fun, operand=None)  
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            loss = actor_loss
            
        return loss, info
        
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params
        
    @jax.jit
    def update(self, batch, td_training = False):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, td_training = td_training)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)          
        self.target_update(new_network, 'actor_bc_flow')

        return self.replace(network=new_network, rng=new_rng), info

    
    @jax.jit
    @partial(jax.jit, static_argnames=('obs_min', 'obs_max'))
    def compute_flow_sf(
        self,
        noises, 
        observations,
        obs_min = None, obs_max = None,
        use_target = False
    ):
        
        """Compute actions i.e. future states from the BC flow model using the Euler method."""
        
        def true_fun(operand):
            obs, act, t = operand
            return self.network.select('target_actor_bc_flow')(obs, act, t, is_encoded=True)
        def false_fun(operand):
            obs, act, t = operand
            return self.network.select('actor_bc_flow')(obs, act, t, is_encoded=True)
        
        #if self.config['encoder'] is not None:
        #    observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises #batch_size x num_samples, obs_dim
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = jax.lax.cond(use_target, true_fun, false_fun, operand=(observations, actions, t))
            actions = actions + vels / self.config['flow_steps']
            
            if obs_min is not None and obs_max is not None:
                actions = jnp.clip(actions, obs_min + 1e-5, obs_max - 1e-5)
        
        return actions #batch_size x num_samples, obs_dim

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()
            enc_inputs = ex_observations.copy()
            obs_dim = 512
            ex_observations = jnp.ones((ex_observations.shape[0], obs_dim))
            
        decoders = dict()
        if config['decoder'] is not None:
            decoder_module = decoder_modules[config['decoder']]
            decoders['actor_bc_flow'] = decoder_module(
                output_shape = ob_dims
            )

        if config['state_action']:
            ex_inputs = jnp.concatenate([ex_observations, ex_actions], axis = -1)
        else:
            ex_inputs = ex_observations
            
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=obs_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )


        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_inputs, ex_times)),
            target_actor_bc_flow=(copy.deepcopy(actor_bc_flow_def), (ex_observations, ex_inputs, ex_times)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (enc_inputs,)) #todo

        if decoders.get('actor_bc_flow') is not None:
            dec_inputs = jnp.ones((ex_observations.shape[0], 512))
            network_info['actor_bc_flow_decoder'] = (decoders.get('actor_bc_flow'), (dec_inputs,)) #todo
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_actor_bc_flow'] = params['modules_actor_bc_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='mc_occ',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            decoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            flow_steps=10,  # Number of flow steps.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            future_prediction = True,
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            state_action=False,
            encoder_only=False
        )
    )

    return config