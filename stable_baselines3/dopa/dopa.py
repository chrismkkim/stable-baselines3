from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from contextlib import nullcontext
from stable_baselines3.common.buffers import RolloutBuffer, MetaRolloutBuffer
from stable_baselines3.common.on_policy_dopa_algorithm import OnPolicyDopaAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticDopaPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
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

    def update_partial(self, loss, frac, nenv):
        frac_nenv = int(frac * nenv)
        
        # Shift and insert new loss
        self.loss_buffer = th.roll(self.loss_buffer, shifts=-1)
        self.loss_buffer[-1] = loss
        
        # Compute moving average (ignore NaNs in early steps)
        valid_losses = self.loss_buffer[~th.isnan(self.loss_buffer)]
        bottom_half, _ = th.topk(valid_losses, frac_nenv, largest=False)
        avg = bottom_half.mean()
        
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
        learning_rate: Union[float, Schedule] = 1e-5, # better: 1e-5, works: 1e-4, default: 7e-4
        learning_rate_dopa: Union[float, Schedule] = 1e-5,
        normalize_values: bool = False,
        n_steps: int = 1, #<---- used to be 5
        n_meta_steps: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.0, #<---- used to be 1.0
        ent_coef: float = 0.5, #<---- used to be 0.0
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        meta_rollout_buffer_class: Optional[type[MetaRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
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

        if _init_setup_model:
            self._setup_model()
            
        # learning rates
        self.policy.learning_rate_dopa = learning_rate_dopa
        self.learning_rate = learning_rate
        # normalize values
        self.normalize_values = normalize_values
        # for training
        self.train_meta = True
        self.n_record = 10
        self.n_avg = 20
        self.loss_avg_meta = th.zeros(self.n_avg)
        self.loss_avg_rl   = th.zeros(self.n_avg)
        self.loss_buffer_rl = th.zeros(self.n_avg)
        self.floss_meta = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/loss_meta.txt","w")
        self.floss_rl   = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/loss_rl.txt","w")
        self.fadva      = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/adva.txt","w")
        self.fvalue     = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/value.txt","w")
        self.freward    = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/reward.txt","w")
        self.ftime_switch = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/time_switch.txt","w")
        self.ftime = open("/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/RL/code/traindata/time.txt","w")
        # track training loss
        self.tracker_metaLoss = MovingAverageTracker(window_size=20)
        self.tracker_rlLoss   = MovingAverageTracker(window_size=20)
        
    def _update_da_lr(self, optimizer: th.optim.Optimizer, learning_rate: float, learning_rate_dopa: float):
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.param_groups[1]["lr"] = learning_rate_dopa
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = base_lr


    def train(self,time_step:int, total_timesteps:int) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        """
        two learning rates
        """
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        self._update_da_lr(optimizer=self.policy.optimizer, learning_rate=self.learning_rate, learning_rate_dopa=self.policy.learning_rate_dopa)

        # Train networks selectively
        #  - train dopa only if meta train
        #  - train rl only if not meta train
        # ctx_dopa = nullcontext() if self.train_meta else th.no_grad()
        # ctx_rl   = th.no_grad() if self.train_meta else nullcontext()
        ctx_dopa = nullcontext()
        ctx_rl   = th.no_grad() if self.train_meta else nullcontext()
        
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            
            with ctx_dopa:                

                # advantages = rollout_data.advantages
                # if self.normalize_advantage:
                #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                # # Convert to tensor
                # rewards_tensor     = th.as_tensor(rollout_data.rewards).view(-1,1)
                # values_tensor      = th.as_tensor(rollout_data.old_values).view(-1,1)
                # next_values_tensor = th.as_tensor(rollout_data.next_values).view(-1,1)
                # next_dones_tensor  = th.as_tensor(rollout_data.next_dones).view(-1,1)
                                               
                # if self.normalize_values:
                #     values_tensor_mean = values_tensor.mean()
                #     values_tensor      = values_tensor      - values_tensor_mean
                #     next_values_tensor = next_values_tensor - values_tensor_mean
                
                # # Evaluate dopa network
                # #   * Use the saved rollout data as inputs.
                # #   * The ordering is different from collect_rollouts() in OnPolicyDopaAlogrithm.
                # dopa = self.policy.gen_dopa(rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor)
                # dopa = dopa.flatten()
                
                # # Meta loss
                # loss_meta = F.mse_loss(advantages, dopa)   
                
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
                                
            # train rl network if condition is met
            with th.no_grad():
                self.switch_train_type(time_step, loss_meta)
                ctx_rl = th.no_grad() if self.train_meta else nullcontext()
            
            with ctx_rl:
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
            
            # Optimization step
            self.policy.optimizer.zero_grad()
            if self.train_meta:
                loss_meta.backward()
            else:
                loss_meta.backward()
                loss_rl.backward()

            # save training data
            if time_step % (self.env.num_envs * self.n_record) == 0:
                self.save_train_data(time_step, loss_meta, loss_rl, rollout_data)                

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        if time_step == total_timesteps:
            self.ftime.close()
            self.ftime_switch.close()
            self.floss_meta.close()
            self.floss_rl.close()
            self.fadva.close()
            self.fvalue.close()
            
    def switch_train_type(self, time_step:int, loss_meta:th.Tensor):                    
        loss_meta_avg = self.tracker_metaLoss.update(loss_meta)                    
        thresh_high = -4
        thresh_low  = -4
        patience    = 3 * 100                                    
        if self.train_meta:                    
            # if time_step % (self.env.num_envs * self.n_record) == 0:
            #     print("Meta", th.log10(loss_meta_avg).item())
            # if self.tracker_metaLoss.stayed_inbetween(thresh_low, thresh_high, patience) or (self.tracker_metaLoss.below_threshold(threshold=thresh_low) and (time_step > self.n_avg * self.env.num_envs * self.n_record)):
            if self.tracker_metaLoss.below_threshold(threshold=thresh_low) and (time_step > self.n_avg * self.env.num_envs * self.n_record):
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
                
    def save_train_data(self, time_step:int, loss_meta:th.Tensor, loss_rl:th.Tensor, rollout_data:RolloutBufferSamples):                            
        self.ftime.write(f"{time_step}\n")
        self.floss_meta.write(f"{loss_meta.detach().item()}\n")
        self.floss_rl.write(f"{loss_rl.detach().item()}\n")
        self.fadva.write(f"{rollout_data.advantages.mean().detach().item()}\n")
        self.fvalue.write(f"{rollout_data.old_values.mean().detach().item()}\n")
        self.freward.write(f"{rollout_data.rewards.mean().detach().item()}\n")                

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
