from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch


@dataclass
class TSPipeModelOutput:
    data: Any
    lst_require_grad_tensor_path: Optional[List[Tuple]]


class TSPipeModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input) -> Union[TSPipeModelOutput, torch.Tensor]:
        if isinstance(input, TSPipeModelOutput):
            model_output = self.model.forward(input.data)
        else:
            model_output = self.model.forward(input)

        if isinstance(model_output, torch.Tensor):
            return model_output

        return TSPipeModelOutput(data=model_output, lst_require_grad_tensor_path=None)
