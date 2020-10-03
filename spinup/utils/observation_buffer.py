import gym
import torch
import numpy as np

import spinup.algos.pytorch.ppo.core as core
from spinup.utils.torch_ext import as_tensor

class ObservationBuffer:
    def __init__(self, obs_space, size, dtype=torch.float32, device=None):
        self.size = size
        self.dtype = dtype
        self.device = device

        if isinstance(obs_space, gym.spaces.Box):
            obs_dim = obs_space.shape[0]
            self._buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        elif isinstance(obs_space, gym.spaces.Dict):
            self._buf = [None] * size
        else:
            raise RuntimeError(f"Only support Box or Dict obs_space. Got {type(obs_space)} instead.")

    def __setitem__(self, index, value):
        self._buf[index] = value

    def __getitem__(self, index):
        return self._buf[index]

    def get(self, index=None):
        if index is None:
            _buf = self._buf
        else:
            _buf = self._buf[index]

        if isinstance(self._buf, np.ndarray):
            return _buf

        return convert_aos_to_soa(_buf, self.dtype, self.device)

def convert_aos_to_soa(array_of_struct, dtype, device):
    return {
        k: as_tensor([struct[k] for struct in array_of_struct], dtype, device)
        for k, v in array_of_struct[0].items()
    }
