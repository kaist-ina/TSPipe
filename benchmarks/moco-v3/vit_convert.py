
from typing import Optional
import torch

from timm.models.vision_transformer import VisionTransformer

class ClsTokenWrapper(torch.nn.Module):
    def __init__(self, cls_token: torch.nn.Module):
        super().__init__()
        self.cls_token = cls_token

    def forward(self, x):
        # assume self.dist_token is None
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        return x

class PosDropWrapper(torch.nn.Module):
    def __init__(self, pos_drop: torch.nn.Module, pos_embed: torch.nn.Parameter):
        super().__init__()
        self.pos_drop = pos_drop
        self.pos_embed = pos_embed

    def forward(self, x):
        return self.pos_drop(x + self.pos_embed)

class PreLogitsWrapper(torch.nn.Module):
    def __init__(self, pre_logits: torch.nn.Module):
        super().__init__()
        self.pre_logits = pre_logits

    def forward(self, x):
        # assume self.dist_token is None
        return self.pre_logits(x[:, 0])

def convert_vit_to_sequential(vit: VisionTransformer, predictor: Optional[torch.nn.Sequential] = None) -> torch.nn.Sequential:
    assert vit.head_dist is None
    assert vit.dist_token is None
    assert isinstance(vit.blocks, torch.nn.Sequential)
    assert isinstance(vit.head, torch.nn.Sequential)
    assert predictor is None or isinstance(predictor, torch.nn.Sequential)

    if predictor is None:
        predictor = torch.nn.Sequential(
            torch.nn.Identity()
        )

    return torch.nn.Sequential(
        vit.patch_embed,
        ClsTokenWrapper(vit.cls_token),
        PosDropWrapper(vit.pos_drop, vit.pos_embed),
        *vit.blocks.children(),
        vit.norm,
        PreLogitsWrapper(vit.pre_logits),
        *vit.head.children(),
        *predictor.children()
    )

__all__ = ['convert_vit_to_sequential']