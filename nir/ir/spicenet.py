from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .node import NIRNode


class SPICENet(NIRNode):
    """SPICEnet
    
    This is a network of N-SPICEnet SOMs of the same neuron length M.
    
    Assumes input of shape (N,).
    Outputs one HCM matrix for each SOM pair and each matrix having shape (M, M).
    """
    
    soms: Dict[str, "SPICEnetSOM"] # Named SOMs to name HCM combinations
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    
    def __post_init__(self):
        # Assert that each SOM has the same number of neurons
        assert len(set([len(som.neurons) for som in self.soms.values()])) == 1
        
        # Assert that at least two SOMs are present
        assert len(self.soms.values()) >= 2
        
        self.input_type = {
            "input": np.array([0] * len(self.soms.values()))
        }
        
        n_neurons = len(self.soms.values()[0].neurons)
        possible_som_pairs = (len(self.soms.values()) * (len(self.soms.values()) - 1)) / 2
        self.output_type = {
            "output": np.array([[[0] * n_neurons] * n_neurons] * possible_som_pairs)
        }
        
    @staticmethod
    def from_list(*soms: "SPICEnetSOM") -> "SPICENet":        
        if len(soms) > 0 and (
                isinstance(soms[0], list) or isinstance(soms[0], tuple)
            ):
                soms = [*soms[0]]
        
        return SPICENet(soms={f"{i}": som for i, som in enumerate(soms)})

@dataclass(eq=False)
class SPICEnetSOM(NIRNode):
    """SPICEnet SOM
    
    This is a one dimensional SOM layer filled by N-SPICEnet SOM Neurons.
    
    Assumes a single input value.
    Outputs activations for each neuron (N,).
    """
    
    neurons: List["SPICEnetSOMNeuron"]
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    
    def __post_init__(self):
        self.input_type = {
            "input": np.array(0)
        }
        self.output_type = {
            "output": np.array([0] * len(self.neurons))
        }

@dataclass(eq=False)
class SPICEnetSOMNeuron(NIRNode):
    r"""SPICEnet SOM Neuron

    This is equivalent to the gaussian PDF function.

    Assumes a one-dimensional input vector of shape (1,).
    Outputs a single activity value.

    .. math::
        a^p_i(t) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x(t)-\mu)^2}{2\sigma^2}
    """
    
    std: np.ndarray
    mean: np.ndarray
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # STD and mean must be scalar
        assert self.std.shape == (1,)
        assert self.mean.shape == (1,)
        
        # STD must be positive
        assert self.std > 0
        
        self.input_type = {
            "input": np.array(self.std.shape)
        }
        self.output_type = {
            "output": np.array(self.std.shape)
        }
        