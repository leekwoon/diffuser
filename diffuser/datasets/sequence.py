from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

# ==== Debug =====

# def resample_trajectory(trajectory, actions, num_points):
#     import scipy.interpolate
#     # Calculate the cumulative distance along the trajectory
#     distances = np.sqrt((np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2))
#     cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

#     # Unwrap yaws for continuous interpolation

#     # Create interpolation functions
#     x_interpolator = scipy.interpolate.interp1d(cumulative_distance, trajectory[:, 0], kind='linear')
#     y_interpolator = scipy.interpolate.interp1d(cumulative_distance, trajectory[:, 1], kind='linear')
#     # Create interpolation functions for each dimension of actions
#     action_interpolators = [scipy.interpolate.interp1d(cumulative_distance, actions[:, i], kind='linear') for i in range(actions.shape[1])]


#     # Uniformly spaced distances along the trajectory
#     uniform_distances = np.linspace(0, cumulative_distance[-1], num_points)

#     # Interpolate the trajectory and yaws
#     resampled_trajectory = np.column_stack((x_interpolator(uniform_distances), y_interpolator(uniform_distances)))
#     resampled_actions = np.column_stack([interp(uniform_distances) for interp in action_interpolators])

#     return resampled_trajectory.astype(np.float32), resampled_actions.astype(np.float32)


# class ResampledGoalDataset(SequenceDataset):

#     """use only pose"""

#     def __init__(self, env='hopper-medium-replay', horizon=64,
#         normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
#         max_n_episodes=10000, termination_penalty=0, use_padding=True):
#         self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
#         self.env = env = load_environment(env)
#         self.horizon = horizon
#         self.max_path_length = max_path_length
#         self.use_padding = use_padding
#         itr = sequence_dataset(env, self.preprocess_fn)

#         fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
#         for i, episode in enumerate(itr):
#             """use only pose"""
#             episode['observations'] = episode['observations'][: ,:2]
#             fields.add_path(episode)
#         fields.finalize()

#         self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
#         self.indices = self.make_indices(fields.path_lengths, horizon)

#         self.observation_dim = fields.observations.shape[-1]
#         self.action_dim = fields.actions.shape[-1]
#         self.fields = fields
#         self.n_episodes = fields.n_episodes
#         self.path_lengths = fields.path_lengths
#         self.normalize()

#         print(fields)
#         # shapes = {key: val.shape for key, val in self.fields.items()}
#         # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

#     def normalize(self, keys=['observations', 'actions']):
#         '''
#             normalize fields that will be predicted by the diffusion model
#         '''
#         for key in keys:
#             array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
#             normed = self.normalizer(array, key)
#             self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             for start in range(max_start):
#                 end = start + horizon
#                 indices.append((i, start, end))
#         indices = np.array(indices)
#         return indices

#     def __len__(self):
#         return len(self.indices)

#     def get_conditions(self, observations):
#         '''
#             condition on both the current observation and the last observation in the plan
#         '''
#         return {
#             0: observations[0],
#             # self.horizon - 1: observations[-1],
#             32 - 1: observations[-1],
#         }

    
#     def __getitem__(self, idx, eps=1e-4):
#         path_ind, start, end = self.indices[idx]

#         observations = self.fields.normed_observations[path_ind, start:end]
#         actions = self.fields.normed_actions[path_ind, start:end]

#         # resample 384 -> 32
#         observations = observations[::12]
#         actions = actions[::12]

#         # observations, actions = resample_trajectory(observations, actions, num_points=32)

#         conditions = self.get_conditions(observations)
#         trajectories = np.concatenate([actions, observations], axis=-1)
#         batch = Batch(trajectories, conditions)
#         return batch
    

# class RelativeResampledGoalDataset(SequenceDataset):

#     """use only pose"""

#     def __init__(self, env='hopper-medium-replay', horizon=64,
#         normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
#         max_n_episodes=10000, termination_penalty=0, use_padding=True):
#         self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
#         self.env = env = load_environment(env)
#         self.horizon = horizon
#         self.max_path_length = max_path_length
#         self.use_padding = use_padding
#         itr = sequence_dataset(env, self.preprocess_fn)

#         fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
#         for i, episode in enumerate(itr):
#             """use only pose"""
#             episode['observations'] = episode['observations'][: ,:2]
#             fields.add_path(episode)
#         fields.finalize()

#         self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
#         self.indices = self.make_indices(fields.path_lengths, horizon)

#         self.observation_dim = fields.observations.shape[-1]
#         self.action_dim = fields.actions.shape[-1]
#         self.fields = fields
#         self.n_episodes = fields.n_episodes
#         self.path_lengths = fields.path_lengths

#         # update normalize statistics
#         observations_list = []
#         for _ in range(10000):
#             idx = np.random.choice(self.indices.shape[0])
#             path_ind, start, end = self.indices[idx]

#             observations = self.fields.observations[path_ind, start:end]

#             # resample 384 -> 32
#             observations = observations[::12]

#             # use relative positions
#             observations = observations.copy()
#             observations[:, 0] = observations[:, 0] - observations[0, 0]
#             observations[:, 1] = observations[:, 1] - observations[0, 1]
#             observations_list.append(observations)
#         self.normalizer.normalizers['observations'].mins = np.min(np.concatenate(observations_list), axis=0)
#         self.normalizer.normalizers['observations'].maxs = np.max(np.concatenate(observations_list), axis=0)

#         self.normalize(keys=['actions'])
#         # self.normalize()

#         print(fields)
#         # shapes = {key: val.shape for key, val in self.fields.items()}
#         # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

#     def normalize(self, keys=['observations', 'actions']):
#         '''
#             normalize fields that will be predicted by the diffusion model
#         '''
#         for key in keys:
#             array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
#             normed = self.normalizer(array, key)
#             self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             for start in range(max_start):
#                 end = start + horizon
#                 indices.append((i, start, end))
#         indices = np.array(indices)
#         return indices

#     def __len__(self):
#         return len(self.indices)

#     def get_conditions(self, observations):
#         '''
#             condition on both the current observation and the last observation in the plan
#         '''
#         return {
#             0: observations[0],
#             # self.horizon - 1: observations[-1],
#             32 - 1: observations[-1],
#         }
    
#     def __getitem__(self, idx, eps=1e-4):
#         path_ind, start, end = self.indices[idx]

#         # observations = self.fields.normed_observations[path_ind, start:end]
#         observations = self.fields.observations[path_ind, start:end]
#         actions = self.fields.normed_actions[path_ind, start:end]

#         # resample 384 -> 32
#         observations = observations[::12]
#         actions = actions[::12]

#         # use relative positions
#         observations = observations.copy()
#         observations[:, 0] = observations[:, 0] - observations[0, 0]
#         observations[:, 1] = observations[:, 1] - observations[0, 1]

#         # normalize strictly
#         observations = self.normalizer.normalize(observations, 'observations')

#         # observations, actions = resample_trajectory(observations, actions, num_points=32)

#         conditions = self.get_conditions(observations)
#         trajectories = np.concatenate([actions, observations], axis=-1)
#         batch = Batch(trajectories, conditions)
#         return batch