from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from model.fiber import Fiber

class DropoutSE3(nn.Module):
    """
    Per-degree implementation of dropout. 
    """

    def __init__(self, fiber_in: Fiber, prob: float = 0.1):
        self.fiber_in = fiber_in
        # Generate an identity matrix and set elements to zero with probability prob
        b = torch.distributions.Bernoulli(torch.tensor(1 - prob)) # Since we want 90% chance of keeping 
        self.drop_vecs = {}
        for degree, channel in fiber_in:
            self.drop_vecs[degree] = b.sample(sample_shape=[channel])*(1/(1-prob))

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:    
        if self.training:    
            return {degree: torch.einsum('i..., i -> i...', feature, self.drop_vecs[degree])
                for degree, feature in features.items()}
        else:
            return features
        