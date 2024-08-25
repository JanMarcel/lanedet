import torch.nn as nn
import torch

from lanedet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg, backbone=None):
        super(Detector, self).__init__()
        self.cfg = cfg
        if backbone is None:
            self.backbone = build_backbones(cfg)
            self.is_onnx = False
        else:
            self.backbone = backbone
            self.is_onnx = True
            self.training = False
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
    
    def get_lanes(self, output):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        # Original
        #fea = self.backbone(batch['img'])   # size is (1, 1280, 12, 20)
        
        # Modified
        if not self.is_onnx:
            fea = self.backbone(batch['img'])
        else:
            ort_inputs = {self.backbone.get_inputs()[0].name: batch['img']}
            fea = self.backbone.run(["461"], ort_inputs)
            #tmp = torch.transpose(torch.from_numpy(fea[0]), 2, 3).unsqueeze(0).cuda()
            fea = list(torch.from_numpy(fea[0]).unsqueeze(0).cuda())

        # Original
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            out = self.heads(fea, batch=batch)
            output.update(self.heads.loss(out, batch))
        else:
            output = self.heads(fea)
        return output
