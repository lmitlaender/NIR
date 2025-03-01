from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .node import NIRNode

@dataclass(eq=False)
class SPICENet(NIRNode):
    """SPICEnet
    
    This is a network of N-SPICEnet SOMs of the same neuron length M.
    
    Assumes input of shape (N,).
    Has one HCM matrix for each SOM pair and each matrix having shape (M, M).
    """
    
    soms: Dict[str, "SPICEnetSOM"] # Named SOMs to name HCM combinations
    hcms: Dict[str, "SPICEnetHCM"] # Named HCM combinations
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    
    def __post_init__(self):
        # Dont allow "_" in keys
        assert all("_" not in key for key in self.soms.keys())
        
        # Assert that each SOM has the same number of neurons
        soms_value_list = list(self.soms.values())
        hcms_value_list = list(self.hcms.values())
        
        # Assert that at least two SOMs are present
        assert len(soms_value_list) >= 2
        
        self.input_type = {
            "input": np.array([len(soms_value_list), 1])
        }
        self.output_type = {
            "output": np.array([len(hcms_value_list), hcms_value_list[0].weights.shape[0], hcms_value_list[0].weights.shape[1]])
        }
        
        # Get all possible combinations of SOMs
        som_combos = []
        som_keys_list = list(self.soms.keys())
        for key in range(0, len(som_keys_list)):
            for key2 in range(key + 1, len(som_keys_list)):
                som_combos.append((key, key2))
                
        # Assert that the number of HCMs is equal to the number of SOM combinations
        assert len(som_combos) == len(self.hcms)
        
        # Assert that each HCMs shape fits the SOMs
        for combo in som_combos:
            # Also allow "flipped" HCMs
            som_1_neuron_len = len(self.soms[som_keys_list[combo[0]]].neurons)
            som_2_neuron_len = len(self.soms[som_keys_list[combo[1]]].neurons)
            assert self.hcms[f"{combo[0]}_{combo[1]}"].weights.shape == (som_1_neuron_len, som_2_neuron_len) or self.hcms[f"{combo[1]}_{combo[0]}"].weights.shape == (som_2_neuron_len, som_1_neuron_len)
        
    @staticmethod
    def from_lists(soms: list["SPICEnetSOM"], hcms: list[tuple[int, int, "SPICEnetHCM"]]) -> "SPICENet":
        
        # Create SOMs and HCMs dict
        soms = {f"{i}": som for i, som in enumerate(soms)}
        hcms = {f"{i1}_{i2}": hcm for i1, i2, hcm in hcms}
        
        return SPICENet(soms=soms, hcms=hcms)
    
    def to_dict(self):
        ret = super().to_dict()
        ret["soms"] = {k: v.to_dict() for k, v in self.soms.items()}
        ret["hcms"] = {k: v.to_dict() for k, v in self.hcms.items()}
        return ret
    
    @classmethod
    def from_dict(cls, node):
        from . import dict2NIRNode
        
        node["soms"] = {k: dict2NIRNode(v) for k, v in node["soms"].items()}
        node["hcms"] = {k: dict2NIRNode(v) for k, v in node["hcms"].items()}
        
        return super().from_dict(node)
        
@dataclass(eq=False)
class SPICEnetHCM(NIRNode):
    """SPICEnet HCM
    
    This represents a hebbian correlation matrix between 2 SOMs.
    It needs to be NxM in size where N is the number of neurons in the first SOM and M the number of neurons in the second SOM.
    
    We define the forward step as follows:
    input: (2,) -> two activation vectors from the two SOMs
    output (N, M) -> updated HCM matrix
    """
    
    weights: np.ndarray
    activation_bar_vector_1: np.ndarray
    activation_bar_vector_2: np.ndarray
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    
    def __post_init__(self):
        assert self.weights.ndim == 2
        assert self.activation_bar_vector_1.shape == (self.weights.shape[0],)
        assert self.activation_bar_vector_2.shape == (self.weights.shape[1],)
        
        self.input_type = {
            "input": np.array((2,))
        }
        self.output_type = {
            "output": np.array(self.weights.shape)
        }

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
    
    def to_dict(self):
        ret = super().to_dict()
        ret["neurons"] = {str(k): v.to_dict() for k, v in enumerate(self.neurons)}
        return ret
    
    @classmethod
    def from_dict(cls, data):
        from . import dict2NIRNode
        
        data["neurons"] = [dict2NIRNode(data["neurons"][str(i)]) for i in range(len(data["neurons"]))]
        return super().from_dict(data)
    
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
        assert self.std.shape == ()
        assert self.mean.shape == ()
        
        # STD must be positive
        assert self.std.item() > 0
        
        self.input_type = {
            "input": np.array(0)
        }
        self.output_type = {
            "output": np.array(0)
        }
        