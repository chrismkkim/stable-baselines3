from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from contextlib import nullcontext
from stable_baselines3.common.buffers import RolloutDopaBuffer, MetaRolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyDopaAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticDopaPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutDopaBufferSamples
from stable_baselines3.common.utils import explained_variance

SelfDopa = TypeVar("SelfDopa", bound="Dopa")

class MovingAverageTracker:
    def __init__(self, window_size=20, min_delta=0.05):
        self.window_size = window_size
        self.min_delta = min_delta
        self.loss_buffer = th.full((window_size,), float("nan")) # Initialize with NaNs
        self.moving_averages = th.full((window_size,), float("nan"))
        self.cnt_switch = 0
        self.cnt_stayed = 0
        
    def update(self, loss):
        # Shift and insert new loss
        self.loss_buffer = th.roll(self.loss_buffer, shifts=-1)
        self.loss_buffer[-1] = loss
        
        # Compute moving average (ignore NaNs in early steps)
        valid_losses = self.loss_buffer[~th.isnan(self.loss_buffer)]
        avg = valid_losses.mean()
        
        # Shift and insert new moving average
        self.moving_averages = th.roll(self.moving_averages, shifts=-1)
        self.moving_averages[-1] = avg
        return avg
    
    def has_plateaued(self):
        # Get valid moving averages
        if th.any(th.isnan(self.moving_averages)):
            has_plateaued = False
        else:
            valid_avgs = self.moving_averages
            variation = valid_avgs.max() - valid_avgs.min()
            has_plateaued = variation < self.min_delta * th.abs(valid_avgs).mean()
        return has_plateaued
    
    def below_threshold(self, threshold):
        # Get valid moving averages
        valid_avgs = self.moving_averages[~th.isnan(self.moving_averages)]
        return th.log10(valid_avgs[-1]) < threshold        

    def above_threshold(self, threshold):
        # Get valid moving averages
        valid_avgs = self.moving_averages[~th.isnan(self.moving_averages)]
        return th.log10(valid_avgs[-1]) > threshold        
    
    def stayed_inbetween(self, low, high, patience):
        # Get valid moving averages
        latest_avg = th.log10(self.moving_averages[~th.isnan(self.moving_averages)][-1])
        if (latest_avg < high) and (latest_avg > low):
            self.cnt_stayed += 1
        else:
            self.cnt_stayed = 0
        return self.cnt_stayed > patience
            
    def get_moving_averages(self):
        return self.moving_averages[~th.isnan(self.moving_averages)]

class ResetRLnet:
    def __init__(self, num_reset):
        self.k = 0
        self.num_reset = num_reset
        
    def time_for_reset(self, prg):
        if prg > self.k / self.num_reset:
            self.k += 1
            return True
        else:
            return False
                
    
class Dopa(OnPolicyDopaAlgorithm):
    """
    Advantage Actor Critic (A2C)    

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticDopaPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticDopaPolicy]],
        env: Union[GymEnv, str],
        traintype_meta: bool = True,
        learning_rate: Union[float, Schedule] = 1e-5, # better: 1e-5, works: 1e-4, default: 7e-4
        learning_rate_dopa: Union[float, Schedule] = 1e-5,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        normalize_values: bool = False,
        n_steps: int = 1, # number of rollout time steps
        n_meta_steps: int = 1,
        n_timesteps: float = 1.0, # number of simulation time steps
        n_envs: int = 1,
        n_rlnet_reset: int = 1, # number of resetting the RL net during meta training
        tracker_window_size: int = 200,
        gamma: float = 0.99,
        gae_lambda: float = 0.0, #<---- used to be 1.0
        ent_coef: float = 0.5, #<---- used to be 0.0
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutDopaBuffer]] = None,
        meta_rollout_buffer_class: Optional[type[MetaRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        log_path: Optional[str] = None,
        train_envs: Optional[dict[str, Any]] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            learning_rate_dopa=learning_rate_dopa,
            net_arch=net_arch,
            n_steps=n_steps,
            n_meta_steps=n_meta_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            meta_rollout_buffer_class=meta_rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        if isinstance(net_arch, str):
            # if using RL Zoo to run Dopa, net_arch is passed as a string from the yaml file.
            self.policy_kwargs["net_arch"] = eval(net_arch)        
        else:
            self.policy_kwargs["net_arch"] = net_arch
        
        if _init_setup_model:
            self._setup_model()
            
        self.loss_previous =1e4
        # path way to log
        self.tensorboard_log = tensorboard_log
        self.log_path = log_path
        self.train_envs = train_envs
        # training type (meta vs rl)
        self.traintype_meta = traintype_meta
        # learning rates
        # self.policy.learning_rate_dopa = learning_rate_dopa
        self.learning_rate = learning_rate
        # normalize values
        self.normalize_values = normalize_values
        # for training
        self.train_meta = True
        self.replace_tdnet = True
        # track training loss
        self.tracker_window_size = tracker_window_size
        self.tracker_metaLoss = MovingAverageTracker(window_size=tracker_window_size)
        self.tracker_rlLoss   = MovingAverageTracker(window_size=tracker_window_size)
        self.reset_rlnet      = ResetRLnet(n_rlnet_reset)
        # save training log
        self.n_envs           = n_envs
        self.n_timesteps      = n_timesteps
        self.Nsteps           = int(self.n_timesteps/self.n_envs)
        self._log_advantages  = np.zeros((self.Nsteps, self.n_envs))
        self._log_dopa        = np.zeros((self.Nsteps, self.n_envs))
        self._log_values      = np.zeros((self.Nsteps, self.n_envs))
        self._log_next_values = np.zeros((self.Nsteps, self.n_envs))
        self._log_rewards     = np.zeros((self.Nsteps, self.n_envs))
        self._log_raw_rewards = np.zeros((self.Nsteps, self.n_envs))
        self._log_dones       = np.zeros((self.Nsteps, self.n_envs))
        
    def _update_my_lr(self, optimizer: th.optim.Optimizer, optimizer_meta: th.optim.Optimizer, learning_rate: float, learning_rate_dopa: float):
        optimizer.param_groups[0]["lr"] = learning_rate
        # optimizer.param_groups[1]["lr"] = learning_rate_dopa
        optimizer_meta.param_groups[0]["lr"] = learning_rate_dopa                

    def train(self, time_step:int, total_timesteps:int) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # # Update optimizer learning rate
        # frac = time_step / total_timesteps
        # lr_dopa_adaptive = 0.1*self.policy.learning_rate_dopa * frac + self.policy.learning_rate_dopa * (1-frac)
        # loss_meta_avg = self.tracker_metaLoss.update(loss_meta)    
        # if time_step > self.tracker_window_size * self.n_envs:
        #     if self.tracker_metaLoss.below_threshold(threshold=-2):
        #         self._update_my_lr(optimizer=self.policy.optimizer, optimizer_meta=self.policy.optimizer_meta, learning_rate=self.learning_rate, learning_rate_dopa=1e-4)
        # self._update_learning_rate(self.policy.optimizer)
        
        progress = time_step / total_timesteps
        for rollout_data in self.rollout_buffer.get(batch_size=None):       
            
            if self.traintype_meta:
                # if (time_step > total_timesteps /2) and (th.sum(rollout_data.next_dones.float()) > 0):
                #     x=1
                # loss_meta, loss_rl = self.meta_dummy(time_step, total_timesteps)
                # loss_meta, loss_rl = self.meta_rollout_rl_dopa(rollout_data)
                loss_meta, loss_rl = self.meta_rollout_expanded_rl_dopa(rollout_data)
                # _, _ = self.meta_rollout_rl_td(rollout_data)                

                # reset RL network parameters
                if self.reset_rlnet.time_for_reset(prg=progress):
                    self.rlnet_param_reset()           
                    print('\nReset RL network: ', self.reset_rlnet.k)                         
            else:
                if self.replace_tdnet:
                    self.load_meta_tdnet()
                loss_meta, loss_rl = self.rl_dopa(rollout_data)
                                                
            # Clip grad norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            # save training data
            self.save_train_data(time_step, loss_meta, loss_rl, rollout_data)    
            # self.save_all_train_data(time_step, loss_meta, loss_rl, rollout_data)                                                               

    def rlnet_param_reset(self):
        # reset parameters of value / policy networks
        for module in self.policy.mlp_extractor.value_net.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.policy.mlp_extractor.policy_net.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()           
        

    def meta_rollout_rl_dopa(self, rollout_data:RolloutDopaBufferSamples):
        """
        TD network learns from rollout data. Model TD is used to train the RL network
        """
        loss_meta = self.compute_metaloss_using_rollout(rollout_data)
        loss_rl   = self.compute_rlloss_using_dopa(rollout_data)
        # Optimization step
        self.policy.optimizer.zero_grad()
        self.policy.optimizer_meta.zero_grad()
        loss_rl.backward()
        loss_meta.backward()
        self.policy.optimizer.step()
        self.policy.optimizer_meta.step()                       
        return loss_meta, loss_rl                 

    def meta_rollout_expanded_rl_dopa(self, rollout_data:RolloutDopaBufferSamples):
        """
        TD network learns from rollout data. Model TD is used to train the RL network
        """
        loss_meta = self.compute_metaloss_using_rollout_expanded(rollout_data)
        loss_rl   = self.compute_rlloss_using_dopa(rollout_data)
        # Optimization step
        self.policy.optimizer.zero_grad()
        self.policy.optimizer_meta.zero_grad()
        loss_rl.backward()
        loss_meta.backward()
        self.policy.optimizer.step()
        self.policy.optimizer_meta.step()                       
        return loss_meta, loss_rl                 

    def rl_dopa(self, rollout_data:RolloutDopaBufferSamples):
        # use trained D to train value / policy networks
        with th.no_grad():
            loss_meta = self.compute_metaloss_using_rollout(rollout_data)
        loss_rl = self.compute_rlloss_using_dopa(rollout_data)
        # Optimization step
        self.policy.optimizer.zero_grad()
        loss_rl.backward()
        self.policy.optimizer.step()
        return loss_meta, loss_rl                
    

    def compute_metaloss_using_rollout(self, rollout_data:RolloutDopaBufferSamples):             
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)            
        # Convert to tensor
        raw_rewards_tensor = th.as_tensor(rollout_data.raw_rewards).view(-1,1)
        rewards_tensor     = th.as_tensor(rollout_data.rewards).view(-1,1)
        next_values_tensor = th.as_tensor(rollout_data.next_values).view(-1,1)
        values_tensor      = th.as_tensor(rollout_data.old_values).view(-1,1)
        next_dones_tensor  = th.as_tensor(rollout_data.next_dones).view(-1,1)                                
        rollout_data_as_tensor = [raw_rewards_tensor, rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor]    
        if self.normalize_values:
            values_tensor_mean = values_tensor.mean()
            values_tensor      = values_tensor      - values_tensor_mean
            next_values_tensor = next_values_tensor - values_tensor_mean        
            
        _rewards, _next_values, _values, _dones = self.policy.process_truncated_states(*rollout_data_as_tensor)            
        # Evaluate dopa network
        #   * Use the saved rollout data as inputs.
        #   * The ordering is different from collect_rollouts() in OnPolicyDopaAlogrithm.
        dopa = self.policy.gen_td(_rewards, _next_values, _values, _dones)
        dopa = dopa.flatten()        
        loss_meta = F.mse_loss(advantages, dopa)           
        
        # if not self.traintype_meta:
        #     loss_current = loss_meta.mean().clone()
        #     diff = th.log10(loss_current) - th.log10(th.tensor(self.loss_previous))
        #     self.loss_previous = loss_current.clone()
        #     if diff > 1:
        #         x=1
                
        return loss_meta
    

    def compute_metaloss_using_rollout_expanded(self, rollout_data:RolloutDopaBufferSamples):             
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)            
        # Convert to tensor
        raw_rewards_tensor = th.as_tensor(rollout_data.raw_rewards).view(-1,1)
        rewards_tensor     = th.as_tensor(rollout_data.rewards).view(-1,1)
        next_values_tensor = th.as_tensor(rollout_data.next_values).view(-1,1)
        values_tensor      = th.as_tensor(rollout_data.old_values).view(-1,1)
        next_dones_tensor  = th.as_tensor(rollout_data.next_dones).view(-1,1)                                    
        rollout_data_as_tensor = [raw_rewards_tensor, rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor]
        if self.normalize_values:
            values_tensor_mean = values_tensor.mean()
            values_tensor      = values_tensor      - values_tensor_mean
            next_values_tensor = next_values_tensor - values_tensor_mean        
                        
        # compute advantages with flipped terminal states
        #   i.e., r + dv' - v, intead of the correct value, r + (1-d)v' - v
        # advtanges_checked = rewards_tensor + self.gamma * (th.tensor(1) - next_dones_tensor.float()) * next_values_tensor - values_tensor
        # advantages_flipped = rewards_tensor + self.gamma * next_dones_tensor.float() * next_values_tensor - values_tensor
        
        # _trunc              = (rewards_tensor != raw_rewards_tensor).float()
        # if th.any(_trunc):
        #     _rewards            = raw_rewards_tensor.clone()
        #     _next_values        = (1-_trunc) * next_values_tensor + _trunc * (rewards_tensor - raw_rewards_tensor) / self.gamma
        #     _values             = values_tensor.clone()
        #     _dones              = (1-_trunc) * next_dones_tensor  + _trunc * th.logical_not(next_dones_tensor)
        # else:
        #     _rewards            = raw_rewards_tensor.clone()
        #     _next_values        = next_values_tensor
        #     _values             = values_tensor.clone()
        #     _dones              = next_dones_tensor
            
        _rewards, _next_values, _values, _dones = self.policy.process_truncated_states(*rollout_data_as_tensor)
            
        rollout_data_processed = [advantages, _rewards, _next_values, _values, _dones]
        advantages_expand, rewards_expand, next_values_expand, values_expand, dones_expand = self.policy.include_flipped_dones(*rollout_data_processed)
        
        # _advantages         = _rewards + (th.tensor(1) - _dones) * self.gamma * _next_values - _values
        # _advantages_flipped = _rewards +                  _dones * self.gamma * _next_values - _values        
        # assert th.all(advantages == _advantages.flatten())        
        # # expand all inputs
        # rewards_expand     = th.cat([_rewards.flatten(),     _rewards.flatten()])
        # values_expand      = th.cat([_values.flatten(),      _values.flatten()])
        # next_values_expand = th.cat([_next_values.flatten(), _next_values.flatten()])
        # next_dones_expand  = th.cat([_dones.flatten(),       (th.tensor(1) - _dones).flatten()])
        # advantages_expand  = th.cat([advantages,             _advantages_flipped.flatten()])
                
        # Evaluate dopa network
        #   * Use the saved rollout data as inputs.
        #   * The ordering is different from collect_rollouts() in OnPolicyDopaAlogrithm.
        dopa = self.policy.gen_td(rewards_expand, next_values_expand, values_expand, dones_expand)
        dopa = dopa.flatten()        
        advantages_expand = advantages_expand.flatten()
        loss_meta = F.mse_loss(advantages_expand, dopa)
        
        return loss_meta
                    
    def compute_rlloss_using_dopa(self, rollout_data:RolloutDopaBufferSamples):    
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()            
        # Evaluate actor-critic networks
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()           
        # Policy gradient loss
        policy_loss = -(rollout_data.dopa * log_prob).mean()
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns_dopa, values)
        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        loss_rl = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss        
        return loss_rl


    def load_meta_tdnet(self) -> None:        
        # load the trained meta model
        meta_env_id       = self.train_envs['meta']
        meta_log          = '1'
        path_to_metamodel = self.log_path + meta_env_id + '_' + meta_log + '/' + 'best_model.zip'
        meta_model        = self.load(path_to_metamodel)
        
        # use the td net from the trained meta model
        meta_model_sd = meta_model.policy.mlp_extractor.state_dict()
        rl_model_sd   = self.policy.mlp_extractor.state_dict()
        for layer in range(len(self.policy.net_arch["td"])):
            rl_model_sd['td_net.'+str(2*layer)+'.weight'] = meta_model_sd['td_net.'+str(2*layer)+'.weight'].clone()
            rl_model_sd['td_net.'+str(2*layer)+'.bias']   = meta_model_sd['td_net.'+str(2*layer)+'.bias'].clone()    
        self.policy.mlp_extractor.load_state_dict(rl_model_sd)
        self.replace_tdnet = False          
        

    def save_train_data(self, time_step:int, loss_meta:th.Tensor, loss_rl:th.Tensor, rollout_data:RolloutDopaBufferSamples):                            

        if self.traintype_meta:
            ftime_path = self.log_path + 'meta_time.txt'
            floss_path = self.log_path + 'meta_lossmeta.txt'
            fdone_path = self.log_path + 'meta_done.txt'
        else:
            ftime_path = self.log_path + 'rl_time.txt'
            floss_path = self.log_path + 'rl_lossmeta.txt'
            fdone_path = self.log_path + 'rl_done.txt'
        with open(ftime_path, "a") as ftime:
            ftime.write(f"{time_step}\n")
        with open(floss_path, "a") as floss:
            floss.write(f"{loss_meta.detach().item()}\n")
        with open(fdone_path, "a") as fdone:
            fdone.write(f"{rollout_data.next_dones.float().mean().detach().item()}\n")
            
        if not self.traintype_meta:
            _idx                        = int(time_step / self.n_envs) - 1
            self._log_advantages[_idx]  = rollout_data.advantages
            self._log_dopa[_idx]        = rollout_data.dopa
            self._log_values[_idx]      = rollout_data.old_values
            self._log_next_values[_idx] = rollout_data.next_values
            self._log_rewards[_idx]     = rollout_data.rewards
            self._log_raw_rewards[_idx] = rollout_data.raw_rewards
            self._log_dones[_idx]       = rollout_data.next_dones
            
            if _idx == self.Nsteps-1:
                np.save(self.log_path + 'rl_advantages.npy',  self._log_advantages)
                np.save(self.log_path + 'rl_dopa.npy',        self._log_dopa)
                np.save(self.log_path + 'rl_values.npy',      self._log_values)
                np.save(self.log_path + 'rl_next_values.npy', self._log_next_values)
                np.save(self.log_path + 'rl_rewards.npy',     self._log_rewards)
                np.save(self.log_path + 'rl_raw_rewards.npy', self._log_raw_rewards)
                np.save(self.log_path + 'rl_dones.npy',       self._log_dones)
            

    def meta_rollout_rl_td(self, rollout_data:RolloutDopaBufferSamples):
        """
        TD network learns from rollout data. Actual TD is used to train the RL network
        """
        loss_meta = self.compute_metaloss_using_rollout(rollout_data)
        loss_rl   = self.compute_rlloss_using_td(rollout_data)
        # Optimization step
        self.policy.optimizer.zero_grad()
        self.policy.optimizer_meta.zero_grad()
        loss_rl.backward()
        loss_meta.backward()
        self.policy.optimizer.step()
        self.policy.optimizer_meta.step()                       
        return loss_meta, loss_rl                 
    
    
    def compute_rlloss_using_td(self, rollout_data:RolloutDopaBufferSamples):    
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()            
        # Evaluate actor-critic networks
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()           
        # Policy gradient loss
        policy_loss = -(rollout_data.advantages * log_prob).mean()
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values)
        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        loss_rl = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss        
        return loss_rl        


#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

    def meta_dummy(self, time_step, total_timesteps):
        """
        TD network learns from dummy data. 
        """
        loss_meta = self.compute_metaloss_using_dummy(time_step, total_timesteps)
        loss_rl   = th.tensor(0)
        # Optimization step
        self.policy.optimizer_meta.zero_grad()
        loss_meta.backward()
        self.policy.optimizer_meta.step()                       
        return loss_meta, loss_rl                     

    def compute_metaloss_using_dummy(self, time_step, total_timesteps):                     
        ndata = 100
        m0, m1 = 0, 20
        mi = (m1-m0) * time_step / total_timesteps + m0
        with th.no_grad():
            rewards = th.ones(ndata)
            old_values = th.normal(mean=th.tensor(mi), std=th.tensor(1), size=(ndata,))
            next_values = th.normal(mean=th.tensor(mi), std=th.tensor(1), size=(ndata,))
            next_dones = (th.rand(ndata) < 0.5).float()
            advantages = rewards + (th.tensor(1)-next_dones) * next_values - old_values            
            # Convert to tensor
            rewards_tensor     = th.as_tensor(rewards).view(-1,1)
            values_tensor      = th.as_tensor(old_values).view(-1,1)
            next_values_tensor = th.as_tensor(next_values).view(-1,1)
            next_dones_tensor  = th.as_tensor(next_dones).view(-1,1)                                        
            if False:
                values_tensor_mean = values_tensor.mean()
                values_tensor      = values_tensor      - values_tensor_mean
                next_values_tensor = next_values_tensor - values_tensor_mean        
        # Evaluate dopa network
        dopa = self.policy.gen_td(rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor)
        dopa = dopa.flatten()        
        loss_meta = F.mse_loss(advantages, dopa)           
        return loss_meta
                    
                                
    def train_RL_using_td(self, rollout_data:RolloutDopaBufferSamples):
        loss_meta = self.compute_metaloss_using_rollout(rollout_data)                
        loss_rl   = self.compute_rlloss_using_td(rollout_data)        
        # Optimization step
        self.policy.optimizer.zero_grad()
        self.policy.optimizer_meta.zero_grad()
        loss_rl.backward()
        loss_meta.backward()
        self.policy.optimizer.step()
        self.policy.optimizer_meta.step()           
            
        return loss_meta, loss_rl                     
                            
    def pretrain_with_dummy(self, time_step, total_timesteps, rollout_data:RolloutDopaBufferSamples):

        if self.train_meta:
            """
            pre-training D network with dummy data
            """
            loss_meta = self.compute_metaloss_dummy(time_step, total_timesteps)    
            with th.no_grad():
                loss_rl = self.compute_rlloss_using_dopa(rollout_data)        
            # Detect when pre-training is finished  
            with th.no_grad():
                _ = self.tracker_metaLoss.update(loss_meta)   
                if self.tracker_metaLoss.below_threshold(threshold=-3):
                    self.train_meta = not self.train_meta
                    self.tracker_metaLoss.cnt_switch += 1
                    self.ftime_switch.write(f"{time_step}\n")                                    
            # Optimization step
            self.policy.optimizer_meta.zero_grad()
            loss_meta.backward()
            self.policy.optimizer_meta.step()                        
        else:   
            """
            Use model TD to train RL network if pre-training finished
            """
            loss_meta = self.compute_metaloss_using_rollout(rollout_data)
            loss_rl = self.compute_rlloss_using_dopa(rollout_data)
            # Optimization step
            self.policy.optimizer.zero_grad()
            self.policy.optimizer_meta.zero_grad()
            loss_rl.backward()
            loss_meta.backward()
            self.policy.optimizer.step()
            self.policy.optimizer_meta.step()           
            
        return loss_meta, loss_rl             
        
    def pretrain_with_rollout(self, time_step, total_timesteps, rollout_data:RolloutDopaBufferSamples):

        if self.train_meta:
            """
            pre-training D network with rollout data
            
            """
            loss_meta = self.compute_metaloss_using_rollout(rollout_data)                
            loss_rl = self.compute_rlloss_using_td(rollout_data)        
            # Detect when pre-training is finished  
            with th.no_grad():
                _ = self.tracker_metaLoss.update(loss_meta)   
                if self.tracker_metaLoss.below_threshold(threshold=-3):
                    self.train_meta = not self.train_meta
                    self.tracker_metaLoss.cnt_switch += 1
                    self.ftime_switch.write(f"{time_step}\n")                                    
            # Optimization step
            self.policy.optimizer.zero_grad()
            self.policy.optimizer_meta.zero_grad()
            loss_rl.backward()
            loss_meta.backward()
            self.policy.optimizer.step()
            self.policy.optimizer_meta.step()           
        else:   
            """
            Use TD to train RL network if pre-training finished
            """
            loss_meta = self.compute_metaloss_using_rollout(rollout_data)
            loss_rl = self.compute_rlloss_using_dopa(rollout_data)
            # Optimization step
            self.policy.optimizer.zero_grad()
            self.policy.optimizer_meta.zero_grad()
            loss_rl.backward()
            loss_meta.backward()
            self.policy.optimizer.step()
            self.policy.optimizer_meta.step()           
            
        return loss_meta, loss_rl                     

    def switch_train_type(self, time_step:int, loss_meta:th.Tensor) -> None:                    
        loss_meta_avg = self.tracker_metaLoss.update(loss_meta)                    
        thresh_high = -2
        thresh_low  = -2
        if self.train_meta:                    
            # if time_step % (self.n_envs * self.n_record) == 0:
            #     print("Meta", th.log10(loss_meta_avg).item())
            # if self.tracker_metaLoss.stayed_inbetween(thresh_low, thresh_high, patience) or (self.tracker_metaLoss.below_threshold(threshold=thresh_low) and (time_step > self.n_avg * self.n_envs * self.n_record)):
            if self.tracker_metaLoss.below_threshold(threshold=thresh_low) and (time_step > self.tracker_window_size * self.n_envs):
                # print("------ switch to train RL ------")
                self.train_meta = not self.train_meta
                self.tracker_metaLoss.cnt_switch += 1
                self.ftime_switch.write(f"{time_step}\n")                
        else:      
            # print("RL", loss_rl_avg.item())
            # print("Meta", th.log10(loss_meta_avg).item(), "RL", loss_rl_avg.item())
            # if self.tracker_metaLoss.above_threshold(threshold=thresh_high) or self.tracker_rlLoss.has_plateaued():
            if self.tracker_metaLoss.above_threshold(threshold=thresh_high):
                # print("------ switch to train META ------")                    
                self.train_meta = not self.train_meta
                self.tracker_metaLoss.cnt_switch += 1
                self.ftime_switch.write(f"{time_step}\n")
            self._n_updates += 1                            
        
                            
    def compute_metaloss_using_meta_rollout(self):
        """
        meta rollout
        """
        for meta_rollout_data in self.meta_rollout_buffer.get(batch_size=None):                    
            # Normalize advantage (not present in the original implementation)
            advantages = meta_rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            # Convert to tensor
            rewards_tensor     = th.as_tensor(meta_rollout_data.rewards).view(-1,1)
            values_tensor      = th.as_tensor(meta_rollout_data.old_values).view(-1,1)
            next_values_tensor = th.as_tensor(meta_rollout_data.next_values).view(-1,1)
            next_dones_tensor  = th.as_tensor(meta_rollout_data.next_dones).view(-1,1)
            
            if self.normalize_values:
                values_tensor_mean = values_tensor.mean()
                values_tensor      = values_tensor      - values_tensor_mean
                next_values_tensor = next_values_tensor - values_tensor_mean
            # Evaluate dopa network
            #   * Use the saved rollout data as inputs.
            #   * The ordering is different from collect_rollouts() in OnPolicyDopaAlogrithm.
            dopa = self.policy.gen_dopa(rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor)
            dopa = dopa.flatten()                
            # Meta loss
            loss_meta = F.mse_loss(advantages, dopa)                  
        return loss_meta                    
                
    def save_all_train_data(self, time_step:int, loss_meta:th.Tensor, loss_rl:th.Tensor, rollout_data:RolloutDopaBufferSamples):                            
        self.ftime.write(f"{time_step}\n")
        self.floss_meta.write(f"{loss_meta.detach().item()}\n")
        self.floss_rl.write(f"{loss_rl.detach().item()}\n")
        self.fadva.write(" ".join(f"{x}" for x in rollout_data.advantages.tolist()) + "\n")
        self.fvalue.write(" ".join(f"{x}" for x in rollout_data.old_values.tolist()) + "\n")
        self.fnext_value.write(" ".join(f"{x}" for x in rollout_data.next_values.tolist()) + "\n")
        self.fdones.write(" ".join(f"{x}" for x in rollout_data.next_dones.float().tolist()) + "\n")
        self.freward.write(" ".join(f"{x}" for x in rollout_data.rewards.tolist()) + "\n")
        
        if th.sum(rollout_data.rewards) < 10:
            x=1

    def learn(
        self: SelfDopa,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "Dopa",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDopa:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


