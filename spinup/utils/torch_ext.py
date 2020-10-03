import functools
import collections

import torch

def _generalize_to_dict(func):
    @functools.wraps(func)
    def wrapper(data, dtype=None, device="cuda"):  # None):
        if isinstance(data, collections.abc.Mapping):
            return type(data)(
                {k: wrapper(v, dtype=dtype, device=device) for k, v in data.items()}
            )
        return func(data, dtype=dtype, device=device)

    return wrapper

as_tensor = _generalize_to_dict(torch.as_tensor)
