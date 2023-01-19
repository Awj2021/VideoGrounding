# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from timm.models import create_model
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from .position_encoding import build_position_encoding
import ipdb

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": 0}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class GroupNormBackbone(BackboneBase):
    """ResNet backbone with GroupNorm with 32 channels."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        name_map = {
            "resnet50-gn": (
                "resnet50",
                "/checkpoint/szagoruyko/imagenet/22014122/checkpoint.pth",
            ),
            "resnet101-gn": (
                "resnet101",
                "/checkpoint/szagoruyko/imagenet/22080524/checkpoint.pth",
            ),
        }
        backbone = getattr(torchvision.models, name_map[name][0])(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=GroupNorm32,
        )
        checkpoint = torch.load(name_map[name][1], map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)
        num_channels = 512 if name_map[name][0] in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def replace_bn(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight)
            frozen.bias.data.copy_(bn.bias)
            frozen.running_mean.data.copy_(bn.running_mean)
            frozen.running_var.data.copy_(bn.running_var)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_bn(ch, n)


class GN_8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gn = torch.nn.GroupNorm(8, num_channels)

    def forward(self, x):
        return self.gn(x)


class TimmBackbone(nn.Module):
    def __init__(self, name, return_interm_layers, main_layer=-1, group_norm=False):
        super().__init__()
        backbone = create_model(
            name,
            pretrained=True,
            in_chans=3,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        with torch.no_grad():
            replace_bn(backbone)
        num_channels = backbone.feature_info.channels()[-1]
        self.body = backbone
        self.num_channels = num_channels
        self.interm = return_interm_layers
        self.main_layer = main_layer

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        if not self.interm:
            xs = [xs[self.main_layer]]
        out = OrderedDict()
        for i, x in enumerate(xs):
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[f"layer{i}"] = NestedTensor(x, mask)
        return out

class Dynamic(nn.Module):
    def __init__(self, num_blocks = 1, num_layers = 3):
        super().__init__()
        dynamic_layers = []
        for _ in range(num_layers):
            dynamic_layers.append(self.build_layer(num_blocks))
        self.dynamic_layers = nn.Sequential(*dynamic_layers)
        
    def build_block(self):
        input_layer = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1)), 
            nn.ReLU()
        )
        
        score_layer  = nn.Sequential(
            nn.Linear(2048, 1)
        )
        
        block = nn.ModuleDict({
            'input_layer': input_layer,
            'score_layer': score_layer    
        })
        return block
    
    def build_layer(self, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(self.build_block())
        
        blocks = nn.ModuleList(blocks)

        
        layer = nn.ModuleDict({
            'blocks': blocks
        })
        return layer
        
    def forward_layer(self, layer, input):
        # batch_size = input.shape[0]
        # input = input[0].decompose()[0]
        scores = []
        input_xs = []
        for block in layer['blocks']:
            input_x = block['input_layer'](input)
            out = F.adaptive_max_pool2d(input_x, 1)
            out = out.squeeze(-1).squeeze(-1)
            score = block['score_layer'](out)
            input_xs.append(input_x)
            scores.append(score)
        scores =torch.stack(scores, dim=1)
        discrete_scores = F.gumbel_softmax(scores, dim=1, hard=True)

        discrete_scores = discrete_scores.reshape(list(discrete_scores.shape) + [1, 1])
        
        input_xs = torch.stack(input_xs, dim=1)

        inputs = discrete_scores * input_xs
        inputs = inputs.sum(1)
        
        return inputs
        
    def forward(self, x):
        """
        Output: B x 512 x 14 x 14, for input of size B x 3 x 224 x 224
        Output: B x 512 x 20 x 20, for input of size B x 3 x 320 x 320
        """
        # out = self.feat_extractor(x)
        out_result = OrderedDict()
        mask = x[0].decompose()[1]
        x = x[0].decompose()[0]
        
        residual = x
        for layer in self.dynamic_layers:
            x = self.forward_layer(layer, x)
            
        out = residual + x
        out_result[f"layer0"] = NestedTensor(out, mask)
        return out_result
    
        # out = OrderedDict()
        # for name, x in xs.items():
        #     mask = F.interpolate(
        #         tensor_list.mask[None].float(), size=x.shape[-2:]
        #     ).bool()[0]
        #     out[name] = NestedTensor(x, mask)
        # return out
        

class Joiner(nn.Sequential):
    def __init__(self, backbone, dynamic, position_embedding):
        super().__init__(backbone, dynamic, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        # TODO: 将tensor list 传入到dynamic network. 查看xs是什么类型 (<class 'collections.OrderedDict'>)
        # ipdb.set_trace()
        xs = self[1](xs)
        out = []
        pos = []
        # ipdb.set_trace()
        for name, x in xs.items():
            out.append(x)
            pos.append(self[2](x).to(x.tensors.dtype))
        # TODO: 没有dynamic，这里的输出为？
        return out, pos

# backbone: resnet101
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if args.backbone[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            args.backbone[len("timm_") :],
            False,
            main_layer=-1,
            group_norm=True,
        )
    elif args.backbone in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(
            args.backbone, train_backbone, False, args.dilation
        )
    else:
        # 默认是resnet101.
        backbone = Backbone(args.backbone, train_backbone, False, args.dilation)
    # TODO:在backbone的后面添加dynamic networks.
    dynamic = Dynamic(num_blocks = 2, num_layers = 5)  # num_layers: 3, 4, 5
    model = Joiner(backbone, dynamic, position_embedding)

    if args.freeze_backbone:
        for p in model[0].parameters():
            p.requires_grad_(False)
        for p in model[2].parameters():
            p.requires_grad_(False)
    model.num_channels = backbone.num_channels
    return model
