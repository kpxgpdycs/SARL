# -*- coding: utf-8 -*-
import numpy as np
import torch
import tianshou as ts
import gym
from gymnasium.spaces import Box, Dict as GymDict, MultiDiscrete
import logging
from pathlib import Path
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import VectorReplayBuffer, Collector, Batch
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import ActorProb
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import pickle
from collections import deque
import itertools
import time
import argparse
import csv
import os
import shutil
import sys

from module.assortment import assortment

# ==============================================================================
# Logging Setup
# ==============================================================================
log_dir_base = Path("./logs")
log_dir_base.mkdir(parents=True, exist_ok=True)
model_dir_name = "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DynamicCache")

# ==============================================================================
# System Configuration
# ==============================================================================
class SystemConfig:
    """System configuration parameters"""
    def __init__(self):           
        # --- Basic Parameters ---
        self.U = 20
        self.I = 100
        self.T = 50
        self.K = 6
        self.C_cs = 1000
        # --- Item Properties ---
        size_normal_mean = (10 + 100) / 2; size_normal_std = (100 - 10) / 6      
        rng_item = np.random.default_rng()
        self.s_i = np.clip(rng_item.normal(size_normal_mean, size_normal_std, self.I), 10, 100).astype(int)
        # --- User Preferences ---
        self.initial_user_prefs = self._generate_initial_prefs()
        self.pre_delta = 1 / self.I
        # --- Profit Parameters ---
        self.profit_omega = 1
        self.profit_discount = 0.5
        # --- Cost Parameters ---
        self.cache_update_cost_per_bit =  np.random.uniform(0, 1) 
        self.thrashing_base_cost_omega_tilde =  np.random.uniform(0, 1)
        self.thrashing_max_interval_vm = 4
        
        # Request History Parameters
        self.request_history_window = 10  
        self.num_request_features = 5    

        # --- RL Parameters ---
        #some parameters can be set through the command line for convenient parameter optimization
        self.gamma = None 
        self.epoch = None 
        self.lr = 1e-5
        self.batch_size = 256; self.buffer_size = 30000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_train_envs = 4; self.num_test_envs = 2
        self.step_per_epoch = 2000; self.step_per_collect = 3000
        self.repeat_per_collect = 10; self.gae_lambda = 0.95; self.eps_clip = 0.18
        # --- Path Configuration ---
        self.log_dir = Path("./logs/default_run")
        self.model_dir = Path("./") / model_dir_name
        self.data_path = "./records.pkl"
        self.checkpoint_path = self.model_dir / "checkpoint.pth"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.thrashing_max_interval_vm < 1: self.thrashing_max_interval_vm = 1

    def _generate_initial_prefs(self) -> np.ndarray:
        rng = np.random.default_rng(1024); prefs = rng.random((self.U, self.I))
        row_sums = prefs.sum(axis=1, keepdims=True); safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        return prefs / safe_row_sums

config = SystemConfig()
print(f"Using device: {config.device}")
print(f"Checkpoint path: {config.checkpoint_path}")

# ==============================================================================
# Recommendation Generator
# ==============================================================================
class RecommendationGenerator:
    def __init__(self):
        self.user_prefs = config.initial_user_prefs.copy()
        self.profit = np.zeros(config.I, dtype=np.float32)

    def reset(self):
        self.user_prefs = config.initial_user_prefs.copy()

    def update_user_prefs(self, u: int, i: int, pre_delta: float = config.pre_delta):
        if 0 <= u < config.U and 0 <= i < config.I:
            self.user_prefs[u, i] += pre_delta
            norm = self.user_prefs[u].sum()
            if norm > 0: self.user_prefs[u] /= norm
            else: self.user_prefs[u] = np.ones(config.I, dtype=np.float32) / config.I

    def update_profit(self, cached: np.ndarray):
        base_profit = config.profit_omega * config.s_i
        profit_multipliers = np.where(cached.astype(bool), 1.0, config.profit_discount)
        self.profit = base_profit * profit_multipliers

    def generate_initial_recommendations(self) -> np.ndarray:
        all_selected_items = []
        initial_prefs = config.initial_user_prefs
        for u in range(config.U):
            v = initial_prefs[u]; sorted_indices = np.argsort(-v)
            selected = sorted_indices[:config.K].tolist()
            while len(selected) < config.K: selected.append(selected[-1] if selected else 0)
            all_selected_items.extend(selected)
        return np.array(all_selected_items, dtype=np.int64)

    def generate_with_assortment(self, cached: np.ndarray, base_prefs: np.ndarray) -> np.ndarray:
        if base_prefs.shape != (config.U, config.I): raise ValueError(f"base_prefs shape mismatch")
        self.update_profit(cached); all_selected_items = []; start_time_total = time.time()
        for u in range(config.U):
            v_base = base_prefs[u]
            v_non_neg = np.maximum(v_base, 0)
            w_non_neg = np.maximum(self.profit, 0)
            v_temp = v_non_neg.astype(np.float64); w_temp = w_non_neg.astype(np.float64)

            if np.any(np.isnan(v_temp)) or np.any(np.isinf(v_temp)) or np.any(np.isnan(w_temp)) or np.any(np.isinf(w_temp)):
                logger.error(f"NaN/Inf in assortment input (user {u}). Fallback."); sorted_indices = np.argsort(-v_base); selected_indices = sorted_indices[:config.K].tolist()
            else:
                products = np.stack([v_temp, w_temp], axis=-1)
                try: selected_indices, _, _ = assortment(products, config.K)
                except Exception as e: logger.error(f"Assortment error (user {u}): {e}", exc_info=True); sorted_indices = np.argsort(-v_base); selected_indices = sorted_indices[:config.K].tolist()

            selected = list(selected_indices)
            if len(selected) > config.K: selected = selected[:config.K]
            elif len(selected) < config.K:
                remaining_indices = list(set(range(config.I)) - set(selected)); needed = config.K - len(selected)
                if remaining_indices:
                    remaining_sorted = sorted(remaining_indices, key=lambda idx: v_base[idx], reverse=True)
                    selected.extend(remaining_sorted[:needed])
                while len(selected) < config.K: selected.append(selected[-1] if selected else 0)
            all_selected_items.extend(selected)

        elapsed_total = time.time() - start_time_total
        return np.array(all_selected_items, dtype=np.int64)

# ==============================================================================
# Dynamic Data Controller
# ==============================================================================
class DynamicDataController:

    def __init__(self):
        self.recommender = RecommendationGenerator()
        self.requests = self._load_requests()
        self.cached_recommendations: Optional[np.ndarray] = None
        self.cache_state = np.zeros(config.I, dtype=np.float32)
        self.item_request_history = deque(maxlen=config.request_history_window)
        self.reset()

    def reset(self):
        self.recommender.reset()
        self.current_t = 0
        self.cache_state.fill(0.0)
        self.cached_recommendations = self.recommender.generate_initial_recommendations()
        
        self.item_request_history.clear()
        for _ in range(config.request_history_window):
            self.item_request_history.append(np.zeros(config.I, dtype=np.float32))

    def _load_requests(self) -> List[List[Tuple[int, int]]]:
        try:
            with open(config.data_path, 'rb') as f: raw_records_structure = pickle.load(f)
            requests_per_t = [[] for _ in range(config.T)]; processed_requests = 0; flat_records = []
            is_preformatted_valid = False

            if isinstance(raw_records_structure, list) and len(raw_records_structure) > 0:
                if isinstance(raw_records_structure[0], list):
                    if len(raw_records_structure) == config.T:
                        all_items_valid = True
                        count = 0
                        for t_idx, t_list in enumerate(raw_records_structure):
                            if not isinstance(t_list, list): all_items_valid = False; break
                            for item_idx, item in enumerate(t_list):
                                if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) and 0 <= item[0] < config.U and isinstance(item[1], int) and 0 <= item[1] < config.I):
                                    all_items_valid = False; break
                                count += 1
                            if not all_items_valid: break
                        if all_items_valid:
                            is_preformatted_valid = True
                            logger.info(f"Successfully validated {count} requests in pre-formatted structure.")
                            return raw_records_structure
                        else:
                             logger.warning("Data is List[List] but failed validation. Attempting to flatten.")
                    else:
                         logger.warning(f"Data is List[List] but outer length {len(raw_records_structure)} != config.T {config.T}. Attempting to flatten.")

                if not is_preformatted_valid:
                    if isinstance(raw_records_structure[0], tuple) and len(raw_records_structure[0]) == 3:
                        flat_records = raw_records_structure
                    else:
                        try:
                            flat_records = list(itertools.chain.from_iterable(raw_records_structure))
                        except TypeError:
                            flat_records = raw_records_structure

            elif isinstance(raw_records_structure, list) and len(raw_records_structure) == 0:
                 return requests_per_t
            else:
                 raise TypeError(f"Loaded data from {config.data_path} is not a list, but {type(raw_records_structure)}.")

            if flat_records:
                for record in flat_records:
                    if isinstance(record, (tuple, list)) and len(record) == 3:
                        uid, t, item_id = record
                        if isinstance(t, int) and 0 <= t < config.T and isinstance(item_id, int) and 0 <= item_id < config.I and isinstance(uid, int) and 0 <= uid < config.U:
                            requests_per_t[t].append((uid, item_id)); processed_requests += 1

            if processed_requests > 0:
                logger.info(f"Processed {processed_requests} valid requests into {config.T} timesteps.")
            elif not is_preformatted_valid:
                 logger.warning("No valid requests found or processed.")

            return requests_per_t

        except FileNotFoundError: logger.error(f"Request data file not found: {config.data_path}."); raise
        except Exception as e: logger.error(f"Error loading/processing request data: {e}", exc_info=True); raise

    def advance_time_and_update_prefs(self):
        time_idx = self.current_t

        current_step_requests = np.zeros(config.I, dtype=np.float32)
        if 0 <= time_idx < len(self.requests):
            for (uid, item_id) in self.requests[time_idx]:
                if 0 <= item_id < config.I:
                    current_step_requests[item_id] += 1
        self.item_request_history.append(current_step_requests)
        if 0 <= time_idx < len(self.requests):
            for (uid, item_id) in self.requests[time_idx]:
                if 0 <= uid < config.U and 0 <= item_id < config.I:
                    self.recommender.update_user_prefs(uid, item_id)
        
        self.current_t = (self.current_t + 1) % config.T

    def get_request_history_features(self) -> np.ndarray:
        if not self.item_request_history:
            return np.zeros((config.I, config.num_request_features), dtype=np.float32)

        # history_array has shape (window_size, I)
        history_array = np.array(self.item_request_history)
        mean_reqs = np.mean(history_array, axis=0)
        std_reqs = np.std(history_array, axis=0)
        max_reqs = np.max(history_array, axis=0)
        min_reqs = np.min(history_array, axis=0)
        last_reqs = history_array[-1]
        
        features = np.stack([mean_reqs, std_reqs, max_reqs, min_reqs, last_reqs], axis=1)

        for i in range(features.shape[1]):
            col_sum = np.sum(features[:, i])
            if col_sum > 0:
                features[:, i] /= col_sum
        
        return features.astype(np.float32)

    def update_cache(self, new_cache: np.ndarray):
        self.cache_state = new_cache.astype(np.float32, copy=True)

    def update_recommendations(self, new_recs: np.ndarray):
        self.cached_recommendations = new_recs.copy()

    def get_current_recommendations(self) -> np.ndarray:
        if self.cached_recommendations is None:
            self.cached_recommendations = self.recommender.generate_initial_recommendations()
        return self.cached_recommendations

    def get_current_requests(self) -> List[Tuple[int, int]]:
        time_idx = self.current_t
        return self.requests[time_idx] if 0 <= time_idx < len(self.requests) else []

    @property
    def current_cache_state(self) -> np.ndarray:
        return self.cache_state

# ==============================================================================
# Gym Environment
# ==============================================================================
class DynamicCacheEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, data_controller: DynamicDataController, writer: Optional[SummaryWriter] = None):
        super().__init__()
        self.dc = data_controller
        self.writer = writer
        self.episode_step = 0
        self.total_steps = 0
        self.current_recommendations: Optional[np.ndarray] = None
        self._pending_recommendations: Optional[np.ndarray] = None
        self.cache_history = deque(maxlen=config.thrashing_max_interval_vm + 2)
        
        self.observation_space = GymDict({
            'cache': Box(0, 1, shape=(config.I,), dtype=np.float32),
            'recommendations': MultiDiscrete([config.I] * (config.U * config.K)),
            'request_history': Box(
                low=0.0, high=1.0, 
                shape=(config.I, config.num_request_features), 
                dtype=np.float32
            )
        })
        
        self.action_space = Box(low=0.0, high=1.0, shape=(config.I,), dtype=np.float32)
        self.last_reward_info = {}

    @property
    def cache_state(self) -> np.ndarray:
        return self.dc.current_cache_state

    def update_recommendations(self, new_recs: np.ndarray):
        expected_shape = (config.U * config.K,)
        if new_recs.shape == expected_shape:
            self.current_recommendations = new_recs.copy()
            self.dc.update_recommendations(new_recs)
        else:
             if not hasattr(self, '_logged_shape_error'):
                 logger.error(f"Env {id(self)} received recommendations with incorrect shape: {new_recs.shape}. Expected: {expected_shape}.")
                 self._logged_shape_error = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.dc.reset()
        self.episode_step = 0
        self.current_recommendations = self.dc.get_current_recommendations()
        self._pending_recommendations = None
        self._logged_shape_error = False
        self.cache_history.clear()
        initial_state = np.zeros(config.I, dtype=np.float32)
        for _ in range(self.cache_history.maxlen):
            self.cache_history.append(initial_state.copy())
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        if self._pending_recommendations is not None:
            self.update_recommendations(self._pending_recommendations)
            self._pending_recommendations = None

        action = np.clip(action, self.action_space.low, self.action_space.high)
        previous_cache_state_t_minus_1 = self.dc.current_cache_state.copy()
        new_cache_binary_t = self._update_cache_from_action(action)
        
        # Get requests for the current time step to calculate reward
        current_requests = self.dc.get_current_requests()

        reward, reward_info = self._calculate_reward(
            current_cache_t=new_cache_binary_t,
            previous_cache_t_minus_1=previous_cache_state_t_minus_1,
            requests=current_requests,
            cache_history_for_reward=self.cache_history
        )
        self.last_reward_info = reward_info

        # Update system state
        self.dc.update_cache(new_cache_binary_t)
        self.cache_history.appendleft(new_cache_binary_t.copy())
        
        self.dc.advance_time_and_update_prefs()

        next_obs = self._get_obs()
        self.episode_step += 1
        self.total_steps += 1
        terminated = self.episode_step >= config.T
        truncated = False
        info = reward_info
        return next_obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        cache_obs = self.dc.current_cache_state.astype(np.float32)
        if self.current_recommendations is None:
            self.current_recommendations = self.dc.get_current_recommendations()
        rec_obs = self.current_recommendations.astype(np.int64)
        
        # Get request history features
        req_hist_obs = self.dc.get_request_history_features()

        if cache_obs.shape != self.observation_space['cache'].shape:
            raise ValueError(f"Cache obs shape mismatch")
        if rec_obs.shape != (config.U * config.K,):
             raise ValueError(f"Rec obs shape mismatch")
        if req_hist_obs.shape != self.observation_space['request_history'].shape:
            raise ValueError(f"Request history obs shape mismatch")

        return {'cache': cache_obs, 'recommendations': rec_obs, 'request_history': req_hist_obs}

    def _update_cache_from_action(self, action: np.ndarray) -> np.ndarray:
        priority_indices = np.argsort(-action)
        new_cache = np.zeros(config.I, dtype=np.float32)
        current_size = 0
        item_sizes = config.s_i

        for idx in priority_indices:
            if 0 <= idx < config.I:
                item_size = item_sizes[idx]
                if (current_size + item_size) <= config.C_cs:
                    new_cache[idx] = 1.0
                    current_size += item_size
        return new_cache

    def _calculate_reward(self, current_cache_t: np.ndarray, previous_cache_t_minus_1: np.ndarray, requests: List[Tuple[int, int]], cache_history_for_reward: deque) -> Tuple[float, Dict]:
        
        item_sizes = config.s_i
        hit_reward = 0.0; num_hits = 0; num_requests = 0

        if requests:
            valid_requests = [req for req in requests if isinstance(req, (tuple, list)) and len(req) >= 2 and isinstance(req[1], int) and 0 <= req[1] < config.I]
            num_requests = len(valid_requests)
            if valid_requests:
                requested_items = np.array([req[1] for req in valid_requests])
                if requested_items.size > 0:
                    is_hit = current_cache_t[requested_items].astype(bool)
                    no_hit = ~is_hit
                    num_hits = np.sum(is_hit)
                    hit_reward_part = np.sum(config.profit_omega * item_sizes[requested_items[is_hit]])
                    no_hit_reward_part = np.sum(config.profit_discount * config.profit_omega * item_sizes[requested_items[no_hit]])
                    hit_reward = hit_reward_part + no_hit_reward_part
        hit_rate = (num_hits / num_requests) if num_requests > 0 else 0.0

        # Update cost calculation
        cache_state_change = current_cache_t - previous_cache_t_minus_1
        items_added_mask = (cache_state_change > 0)
        update_cost = np.sum(items_added_mask * item_sizes) * config.cache_update_cost_per_bit

        # thrashing cost calculation
        thrashing_cost = 0.0
        for item_idx in np.where(items_added_mask)[0]:
            # Check for thrashing patterns in history
            for psi in range(1, config.thrashing_max_interval_vm + 1):
                if len(cache_history_for_reward) >= psi + 1:
                    was_cached_before = cache_history_for_reward[psi][item_idx] == 1
                    intermediate_states = [cache_history_for_reward[k][item_idx] for k in range(psi)]
                    was_absent_in_between = np.sum(intermediate_states) == 0

                    if was_cached_before and was_absent_in_between:
                        thrashing_cost += (config.thrashing_base_cost_omega_tilde / (psi)) * item_sizes[item_idx]
                        break

        reward = hit_reward - update_cost - thrashing_cost
        reward_info = {
            'system_reward': reward, 'reward_hit': hit_reward, 'cost_update': update_cost,
            'cost_thrashing': thrashing_cost, 'hit_rate': hit_rate, 'num_hits': float(num_hits), 'num_requests': float(num_requests)
        }
        return reward, reward_info

    def close(self):
        if self.writer: self.writer.close(); self.writer = None

# ==============================================================================
# Neural Network Definitions
# ==============================================================================
class CustomNet(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.cache_net = nn.Sequential(
            nn.Linear(config.I, 128), nn.LayerNorm(128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU()
        ).to(device)
        self.embed_dim_rec = 8 
        self.rec_embed = nn.Embedding(config.I, self.embed_dim_rec).to(device)
        self.user_rec_aggregator = nn.Linear(config.K * self.embed_dim_rec, 32).to(device)
        req_hist_input_dim = config.I * config.num_request_features
        self.req_hist_net = nn.Sequential(
            nn.Linear(req_hist_input_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU()
        ).to(device)

        self.joint_feature_dim = 64 + 32 + 128
        self.output_dim = 256
        self.joint_net = nn.Sequential(
            nn.Linear(self.joint_feature_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(),
            nn.Linear(256, self.output_dim), nn.LeakyReLU()
        ).to(device)

    def _process_input(self, obs: Union[Dict, Batch]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if isinstance(obs, Batch): obs_data = obs.obs if hasattr(obs, 'obs') else obs
        elif isinstance(obs, dict): obs_data = obs
        else: raise TypeError(f"Unsupported observation type: {type(obs)}")

        cache_in = obs_data.get('cache')
        recs_in = obs_data.get('recommendations')
        req_hist_in = obs_data.get('request_history')
        if any(x is None for x in [cache_in, recs_in, req_hist_in]):
            raise ValueError("Missing 'cache', 'recommendations', or 'request_history'")

        cache = torch.as_tensor(cache_in, dtype=torch.float32, device=self.device)
        recs = torch.as_tensor(recs_in, dtype=torch.long, device=self.device)
        req_hist = torch.as_tensor(req_hist_in, dtype=torch.float32, device=self.device)

        if cache.ndim == 1: cache = cache.unsqueeze(0)
        if recs.ndim == 1: recs = recs.unsqueeze(0)
        if req_hist.ndim == 2: req_hist = req_hist.unsqueeze(0) 

        return cache, recs, req_hist, cache.size(0)

    def forward(self, obs: Any, state: Any = None, info: Dict = {}) -> Tuple[torch.Tensor, Any]:
        cache, recs, req_hist, batch_size = self._process_input(obs)
        cache_feat = self.cache_net(cache)
        rec_emb = self.rec_embed(recs) 
        rec_emb_per_user = rec_emb.view(batch_size, config.U, -1) 
        user_rec_features = self.user_rec_aggregator(rec_emb_per_user) 
        global_rec_feat = user_rec_features.mean(dim=1) 
        req_hist_flat = req_hist.reshape(batch_size, -1) 
        req_hist_feat = self.req_hist_net(req_hist_flat)
        combined_feat = torch.cat([cache_feat, global_rec_feat, req_hist_feat], dim=1)
        features = self.joint_net(combined_feat)

        return features, state

class CustomCriticModule(nn.Module):
    def __init__(self, preprocess_net: CustomNet, value_head: nn.Module):
        super().__init__(); self.preprocess_net = preprocess_net; self.value_head = value_head
    def forward(self, obs: Union[Dict, Batch], **kwargs) -> torch.Tensor:
        features, _ = self.preprocess_net(obs, state=kwargs.get('state', None)); return self.value_head(features).squeeze(-1)

# ==============================================================================
# Policy Factory
# ==============================================================================
class PolicyFactory:
    @staticmethod
    def build() -> PPOPolicy:
        act_space = Box(low=0.0, high=1.0, shape=(config.I,), dtype=np.float32)
        net = CustomNet(config.device)
        actor = ActorProb(preprocess_net=net, action_shape=act_space.shape, hidden_sizes=[], max_action=1.0, device=config.device, unbounded=False, conditioned_sigma=True).to(config.device)
        critic_head = MLP(input_dim=net.output_dim, output_dim=1, hidden_sizes=[256, 128], device=config.device).to(config.device)
        critic = CustomCriticModule(net, critic_head).to(config.device)
        optim = torch.optim.Adam(itertools.chain(actor.parameters(), critic.parameters()), lr=config.lr)
        def dist_fn(actor_output_tuple: Tuple[torch.Tensor, torch.Tensor]):
            mu, sigma = actor_output_tuple; sigma = torch.clamp(sigma, min=1e-6)
            return torch.distributions.Independent(torch.distributions.Normal(mu, sigma), 1)
        policy = PPOPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist_fn, action_space=act_space, discount_factor=config.gamma, gae_lambda=config.gae_lambda, max_grad_norm=0.5, vf_coef=0.5, ent_coef=0.005, reward_normalization=True, advantage_normalization=True, recompute_advantage=True, eps_clip=config.eps_clip, value_clip=False, action_scaling=True, action_bound_method="clip", deterministic_eval=True).to(config.device)
        logger.info("PPO Policy built successfully.")
        return policy

# ==============================================================================
# Training Manager and Main Execution
# ==============================================================================
class TrainingManager:
    def __init__(self, resume_mode: Optional[str] = None):
        self.train_csv_path = config.log_dir / "manual_train_metrics.csv"
        self.test_csv_path = config.log_dir / "manual_test_metrics.csv"
        config.log_dir.mkdir(parents=True, exist_ok=True)

        self.resume_mode = resume_mode
        self.start_epoch = 0; self.start_env_step = 0; self.resumed_from_checkpoint = False
        
        self.policy = PolicyFactory.build()
        self.lr_scheduler = None
        if self.policy.optim:
            self.lr_scheduler = lr_scheduler.LinearLR(self.policy.optim, start_factor=1.0, end_factor=0.1, total_iters=config.epoch)
            logger.info("Initialized LinearLR scheduler.")

        if self.resume_mode == 'checkpoint': self._load_checkpoint()
        elif self.resume_mode == 'best_weights': self._load_best_weights()

        def train_env_factory(): return DynamicCacheEnv(DynamicDataController(), writer=None)
        def test_env_factory(): return DynamicCacheEnv(DynamicDataController(), writer=None)
        logger.info(f"Creating {config.num_train_envs} training and {config.num_test_envs} testing environments...")
        self.train_envs = SubprocVectorEnv([train_env_factory for _ in range(config.num_train_envs)], context="spawn")
        self.test_envs = SubprocVectorEnv([test_env_factory for _ in range(config.num_test_envs)], context="spawn")
        logger.info("Environments created successfully.")

        self.buffer = VectorReplayBuffer(config.buffer_size, config.num_train_envs)
        self.collector = Collector(self.policy, self.train_envs, self.buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, self.test_envs)

        self.writer = SummaryWriter(config.log_dir)
        self.logger = TensorboardLogger(self.writer, train_interval=1, test_interval=1, update_interval=100)
        self.train_fn_recommender = RecommendationGenerator()

    def _load_checkpoint(self):
        if config.checkpoint_path.exists():
            try:
                checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
                self.policy.load_state_dict(checkpoint['model'])
                if 'optim' in checkpoint and self.policy.optim: self.policy.optim.load_state_dict(checkpoint['optim'])
                if 'lr_scheduler' in checkpoint and self.lr_scheduler: self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.start_env_step = checkpoint.get('env_step', 0)
                self.resumed_from_checkpoint = True
                logger.info(f"Resuming from CHECKPOINT: epoch {self.start_epoch}, env_step {self.start_env_step}.")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}. Starting from scratch.", exc_info=True)
                self.resume_mode = None; self.resumed_from_checkpoint = False; self.start_epoch = 0; self.start_env_step = 0
        else:
            logger.warning(f"Checkpoint not found at {config.checkpoint_path}. Starting from scratch.")
            self.resume_mode = None; self.resumed_from_checkpoint = False

    def _load_best_weights(self):
        best_policy_path = Path(config.model_dir) / "best_policy.pth"
        if best_policy_path.exists():
            try:
                state_dict = torch.load(best_policy_path, map_location=config.device)
                self.policy.load_state_dict(state_dict)
                logger.info(f"Loaded BEST WEIGHTS from {best_policy_path}.")
            except Exception as e:
                logger.error(f"Error loading best weights: {e}. Starting from scratch.", exc_info=True)
                self.resume_mode = None
        else:
            logger.warning(f"Best weights file not found at {best_policy_path}. Starting from scratch.")
            self.resume_mode = None

    def _save_checkpoint_fn(self, epoch: int, env_step: int, gradient_step: int):
        state = {'model': self.policy.state_dict(), 'optim': self.policy.optim.state_dict() if self.policy.optim else None,
                 'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None, 'epoch': epoch, 'env_step': env_step}
        try:
            temp_path = config.checkpoint_path.with_suffix(".pth.tmp")
            torch.save(state, temp_path); temp_path.rename(config.checkpoint_path)
        except Exception as e: logger.error(f"Error saving checkpoint: {e}", exc_info=True)

    def _create_trainer(self) -> OnpolicyTrainer:
        def save_best_fn(policy): torch.save(policy.state_dict(), config.model_dir / "best_policy.pth")
        def stop_fn(mean_rewards): return False

        def epoch_update_and_log_fn(epoch: int, env_step: Optional[int] = None):
            if epoch > 0 :
                logger.info(f"Epoch {epoch}: Triggering recommendation update.")
                try:
                    cache_state_list = self.train_envs.get_env_attr('cache_state')
                    if cache_state_list and all(isinstance(s, np.ndarray) for s in cache_state_list):
                        representative_cache_state = np.round(np.mean(np.stack(cache_state_list), axis=0)).astype(np.float32)
                    else:
                        representative_cache_state = np.zeros(config.I, dtype=np.float32)

                    current_base_prefs = self.train_fn_recommender.user_prefs
                    
                    new_recs = self.train_fn_recommender.generate_with_assortment(
                        representative_cache_state, base_prefs=current_base_prefs
                    )
                    
                    attr_name = "_pending_recommendations"
                    if self.train_envs is not None: self.train_envs.set_env_attr(attr_name, new_recs)
                    if self.test_envs is not None: self.test_envs.set_env_attr(attr_name, new_recs)
                except Exception as e: logger.error(f"Error during recommendation update hook at epoch {epoch}: {e}", exc_info=True)

            def log_env_metrics(envs, prefix, current_epoch, csv_path):
                if envs is None or self.writer is None: return
                try:
                    info_dicts = [d for d in envs.get_env_attr("last_reward_info") if isinstance(d, dict) and d]
                    if not info_dicts: return
                    result_row = {"epoch": current_epoch}; metric_keys = list(info_dicts[0].keys())
                    for key in metric_keys:
                        values = [info[key] for info in info_dicts if key in info]
                        if values:
                            mean_val, max_val, min_val = np.mean(values), np.max(values), np.min(values)
                            result_row[key] = mean_val; result_row[f"{key}_max"] = max_val; result_row[f"{key}_min"] = min_val
                            self.writer.add_scalar(f"{prefix}/{key}", mean_val, current_epoch)
                            self.writer.add_scalar(f"{prefix}/{key}_max", max_val, current_epoch)
                            self.writer.add_scalar(f"{prefix}/{key}_min", min_val, current_epoch)
                    if len(result_row) > 1:
                        file_exists = csv_path.exists(); csv_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(csv_path, "a", newline='') as csvfile:
                            fieldnames = sorted(list(result_row.keys())); fieldnames.insert(0, fieldnames.pop(fieldnames.index('epoch')))
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            if not file_exists or csvfile.tell() == 0: writer.writeheader()
                            writer.writerow(result_row)
                except Exception as e: logger.warning(f"Failed to log {prefix} metrics: {e}", exc_info=True)

            log_env_metrics(self.train_envs, "env_metrics/train", epoch, self.train_csv_path)
            log_env_metrics(self.test_envs, "env_metrics/test", epoch, self.test_csv_path)

            if self.lr_scheduler:
                try: self.writer.add_scalar("train/learning_rate", self.lr_scheduler.get_last_lr()[0], epoch)
                except Exception: pass
                self.lr_scheduler.step()

        trainer = OnpolicyTrainer(
            policy=self.policy, train_collector=self.collector, test_collector=self.test_collector,
            max_epoch=config.epoch, step_per_epoch=config.step_per_epoch, step_per_collect=config.step_per_collect,
            repeat_per_collect=config.repeat_per_collect, episode_per_test=config.num_test_envs * 3,
            batch_size=config.batch_size, train_fn=epoch_update_and_log_fn,
            save_checkpoint_fn=self._save_checkpoint_fn, save_best_fn=save_best_fn,
            stop_fn=stop_fn, logger=self.logger, resume_from_log=self.resumed_from_checkpoint,
            show_progress=True
        )
        return trainer

    def run_final_evaluation(self):
        logger.info("\n" + "="*80)
        logger.info("--- Starting Final Evaluation with Best Policy ---")
        logger.info("="*80)

        best_policy_path = config.model_dir / "best_policy.pth"
        if not best_policy_path.exists():
            logger.warning(f"Best policy not found at {best_policy_path}. Skipping final evaluation.")
            return

        # 1. Load the best policy
        eval_policy = PolicyFactory.build()
        eval_policy.load_state_dict(torch.load(best_policy_path, map_location=config.device))
        eval_policy.eval() # Set policy to evaluation mode
        logger.info(f"Successfully loaded best policy from {best_policy_path}")

        # 2. Create a clean evaluation environment
        eval_dc = DynamicDataController()
        eval_env = DynamicCacheEnv(data_controller=eval_dc)
        obs, _ = eval_env.reset()

        # 3. Initialize metric accumulators
        total_hit_reward = 0.0
        total_update_cost = 0.0
        total_thrashing_cost = 0.0
        hit_rates = []

        # 4. Run the evaluation loop for all T steps
        for t in range(config.T):
            # At each step, update recommendations based on the latest user preferences
            current_cache = eval_env.cache_state
            current_base_prefs = eval_env.dc.recommender.user_prefs
            new_recs = eval_env.dc.recommender.generate_with_assortment(current_cache, base_prefs=current_base_prefs)
            eval_env.update_recommendations(new_recs)

            # Get the updated observation
            obs = eval_env._get_obs()
            
            # Get deterministic action from the policy
            batch = Batch(obs=[obs], info={})
            with torch.no_grad():
                result = eval_policy(batch, deterministic=True)
            action = result.act.cpu().numpy()[0]

            # Step the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)

            # Aggregate metrics from the info dictionary
            total_hit_reward += info.get('reward_hit', 0.0)
            total_update_cost += info.get('cost_update', 0.0)
            total_thrashing_cost += info.get('cost_thrashing', 0.0)
            if info.get('num_requests', 0) > 0:
                hit_rates.append(info.get('hit_rate', 0.0))
            
            logger.info(f"  Eval Step {t+1}/{config.T}: Reward={reward:.2f}, HitReward={info.get('reward_hit', 0):.2f}, UpdateCost={info.get('cost_update', 0):.2f}, ThrashCost={info.get('cost_thrashing', 0):.2f}, HitRate={info.get('hit_rate', 0):.2%}")

            if terminated or truncated:
                logger.info(f"Evaluation episode ended prematurely at step {t+1}.")
                break
        
        eval_env.close()

        # 5. Calculate final summary metrics
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0

        # 6. Log the final summary
        logger.info("\n" + "="*80)
        logger.info("--- Final Evaluation Summary (T={}) ---".format(config.T))
        logger.info(f"  - Cumulative System Reward:       {total_hit_reward-total_update_cost- total_thrashing_cost:.2f} (Profit - All Costs)")
        logger.info(f"  - Cumulative Hit Reward: {total_hit_reward:.2f}")
        logger.info(f"  - Cumulative Update Cost:         {total_update_cost:.2f}")
        logger.info(f"  - Cumulative Thrashing Cost:      {total_thrashing_cost:.2f}")
        logger.info(f"  - Average Cache Hit Rate:         {avg_hit_rate:.2%} (averaged over steps with requests)")
        logger.info("="*80 + "\n")

    def run(self):
        logger.info(f"Starting training with resume_mode='{self.resume_mode}'")
        trainer = self._create_trainer()
        try:
            result = trainer.run()
            logger.info(f"Training finished. Result: {result}")
            self.run_final_evaluation()
        except Exception as e: 
            logger.error("Error during training run:", exc_info=True)
            raise
        finally:
            logger.info("Cleaning up resources...")
            torch.save(self.policy.state_dict(), config.model_dir / "final_policy.pth")
            if self.train_envs: self.train_envs.close()
            if self.test_envs: self.test_envs.close()
            if self.writer: self.writer.close()
            logger.info("Cleanup complete.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for Dynamic Caching (No Interaction).")
    parser.add_argument("--run-name", type=str, default=f"run_no_interact_{time.strftime('%Y%m%d-%H%M%S')}", help="Name for logging.")
    parser.add_argument('--gamma', type=float, required=True, help='Discount factor for reinforcement learning (gamma).')
    parser.add_argument('--epoch', type=int, required=True, help='Total number of training epochs.')
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_const", dest="resume_mode", const="checkpoint", help="Resume from checkpoint.")
    resume_group.add_argument("--resume-best-weights", action="store_const", dest="resume_mode", const="best_weights", help="Resume from best weights.")
    parser.set_defaults(resume_mode=None)
    args = parser.parse_args()

    config.gamma = args.gamma
    config.epoch = args.epoch

    run_log_dir = log_dir_base / args.run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_log_dir / f"{args.run_name}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"File log initialized to: {file_handler.baseFilename}")

    try:
        shutil.copy2(Path(sys.argv[0]).resolve(), run_log_dir / f"{args.run_name}_{Path(sys.argv[0]).name}")
        logger.info("Main script copied to log directory.")
    except Exception as e:
        logger.error(f"Could not copy main script: {e}")

    config.log_dir = run_log_dir
    config.model_dir = Path("./models") / args.run_name
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path = config.model_dir / "checkpoint.pth"
    logger.info(f"Log dir: {config.log_dir}, Model dir: {config.model_dir}")

    # Pre-run checks
    logger.info("Performing pre-run checks...")
    if not Path(config.data_path).exists():
        logger.error(f"FATAL: Request data file not found: {config.data_path}"); exit(1)
    try:
        assortment(np.random.rand(5, 2).astype(np.float64), 2)
        logger.info("Assortment function check successful.")
    except NameError:
        logger.error("FATAL: Assortment function not found."); exit(1)
    logger.info("Pre-run checks passed.")

    # Start training
    try:
        manager = TrainingManager(resume_mode=args.resume_mode)
        manager.run()
        print("\n--- Training Process Ended ---")
        print(f"TensorBoard logs: tensorboard --logdir={log_dir_base}")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        print("\n--- An unexpected error occurred. Check logs for details. ---")