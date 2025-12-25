import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from typing import Optional

from RRP_model.RRP import RRPFeatureExtractor
from RRP_model.models import *
class DepthPredModels(nn.Module):
    def __init__(self, config, encoder_type="dptv2", decoder_type="f3mlp"):
        super().__init__()

        self.config = config
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type # f3mlp / rrp
        
        # encoder
        self._init_encoders()
        self._init_decoders()
        
    def forward(self, func_name, **kwargs):
        # encoder
        if func_name == "encode": # 集成encode
            features = self._encode(**kwargs)
            return features

        elif func_name == "decoder_train":
            return self._decoder_train_get_pred_loss(cond=kwargs["depth_cond"], gt_ray=kwargs["gt_ray"])
        
        elif func_name == "decoder_inference":
            return self._decoder_inference_get_pred(cond=kwargs["depth_cond"], num_samples=kwargs.get("num_samples", 1))
        else:
            raise NotImplementedError

    def _encode(self, obs_img):
        if self.encoder_type == "dptv2":
            features, _, _ = self.dptv2_encoder(obs_img=obs_img)
        return features
    
    def _decoder_train_get_pred_loss(self, cond, gt_ray):
        if self.decoder_type == "f3mlp":
            return self._forward_mlp_train(cond, gt_ray)
        elif self.decoder_type == "rrp":
            return self._forward_rrp_train(cond, gt_ray)
        
    def _decoder_inference_get_pred(self, cond, num_samples=1):
        if self.decoder_type == "f3mlp":
            return self._forward_mlp_inference(cond)
        elif self.decoder_type == "rrp":
            return self._forward_rrp_inference(cond)
            
    def _forward_mlp_train(self, cond, gt_ray):
        pred = self.f3mlp_decoder(cond)
        loss = F.l1_loss(pred, gt_ray)
        return {"pred": pred, "loss": loss} 
    
    def _forward_mlp_inference(self, cond):
        pred = self.f3mlp_decoder(cond)
        return pred
    
    def _forward_rrp_train(self, cond, gt_ray):
        d = self.rrp_decoder(cond)
        loss = F.l1_loss(d, gt_ray)
        return {"pred": d, "loss": loss}
    
    def _forward_rrp_inference(self, cond):
        d = self.rrp_decoder(cond)
        return d
    
    def _init_encoders(self):
        if self.encoder_type == "dptv2":
            self.dptv2_encoder = RRPFeatureExtractor(checkpoint_path=self.config["dptv2_ckpt_path"])
    
    def _init_decoders(self):
        if self.decoder_type == "f3mlp":
            self.f3mlp_decoder = F3MlpDecoder()
        elif self.decoder_type == "rrp":
            self.rrp_decoder = F3MlpDecoder()