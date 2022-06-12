# -*- coding: utf-8 -*-
# @Time : 2022/5/13 22:13
# @Version : v-dev-0.0
# @Function 

"""Modified VGG module form pytorch.

The code here is copied from torchvision's GitHub repository.
Link: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

* The code is part of SJTU MATH8013 Final project, for learning purposes only.

Example:

"""

from typing import Union, List, Dict, Any, Optional, cast,Tuple
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, Tuple[int,int]]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for t in cfg:
        if t == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            t = cast(Tuple[int,int], t)
            v=t[0]
            ks=t[1]
            if ks==3:
                p=1
            elif ks==5:
                p=2
            elif ks==7:
                p=3
            conv2d = nn.Conv2d(in_channels, v, kernel_size=ks, padding=p)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, Tuple[int,int]]]] = {
    "A": [(64,3), "M", (128,3), "M", (256,3),(256,3), "M", (512,3), (512,3), "M", (512,3), (512,3), "M"],
    "B": [(64,3), (64,3), "M", (128,3), (128,3), "M", (256,3), (256,3), "M", (512,3), (512,3), "M", (512,3), (512,3), "M"],
    "D": [(64,3), (64,3), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), "M", (512,3), (512,3), (512,3), "M"],
    "E": [(64,3), (64,3), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), (512,3), "M", (512,3), (512,3), (512,3), (512,3), "M"],
    "D1":[(64,7), (64,7), "M", (128,5), (128,5), "M", (256,5), (256,5), (256,5), "M", (512,3), (512,3), (512,3), "M", (512,3), (512,3), (512,3), "M"],
    "D2":[(64,3), (64,3), "M", (128,3), (128,3), "M", (256,5), (256,5), (256,5), "M", (512,5), (512,5), (512,5), "M", (512,7), (512,7), (512,7), "M"],
    "F1":[(64,5), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), "M", (512,3), (512,3), (512,3), "M"],
    "F2":[(64,7), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), "M", (512,3), (512,3), (512,3), "M"],
    "G1":[(64,3), (64,3), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), "M", (512,5), "M"],
    "G2":[(64,3), (64,3), "M", (128,3), (128,3), "M", (256,3), (256,3), (256,3), "M", (512,3), (512,3), (512,3), "M", (512,7), "M"],

    }


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)

def vgg16_bn_kdec(**kwargs:Any) ->VGG:
    return _vgg("D1",True,**kwargs)

def vgg16_bn_kinc(**kwargs:Any) ->VGG:
    return _vgg("D2",True,**kwargs)

def vgg15_bn_k5(**kwargs: Any) ->VGG:
    return _vgg("F1",True,**kwargs)

def vgg15_bn_k7(**kwargs: Any) ->VGG:
    return _vgg("F2",True,**kwargs)

def vgg14_bn_k5(**kwargs: Any) ->VGG:
    return _vgg("G1",True,**kwargs)

def vgg14_bn_k7(**kwargs: Any) ->VGG:
    return _vgg("G2",True,**kwargs)
