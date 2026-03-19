from typing import Any, List
from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod

class BaseVAE(nn.Module, ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def encode(self, input: Any) -> List[Tensor]:
        pass
    
    @abstractmethod
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def decode(self, input: Tensor) -> Any:
        pass
    
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Any:
        pass
    
    @abstractmethod
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def reconstruct(self, x: Tensor, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Any:
        pass