# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from pysot.models.backbone.peleenet import PeleeNet17b, PeleeNet31b
from pysot.models.backbone.peleenetv1 import PeleeNet7a

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'peleenet17': PeleeNet17b,
              'peleenetv1': PeleeNet7a,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
