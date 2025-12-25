import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from typing import Optional
from model.mono.depth_net import depth_feature_res
from model.models import *
from model.unloc import UnlocFeatureExtractor

class DepthPredModels(nn.Module):
    def __init__(self, config, encoder_type="dptv2", decoder_type="f3mlp"):
        super().__init__()
        """
        encoder:
            res50: resnet50
            dptv2: DepthAnythingV2的dinoV2
            res50_3D: Res50 3D先验
            res50_RSK Res50 Rsk
        decoder:
            f3mlp: 分类bin
            rrp: 与f3mlp一致 (F3MlpDecoder)
            
        修改encoder以后，要检查_encoder()函数的逻辑，是否能给decoder返回正确的tensor[B, 40, 128]
        """
        self.config = config
        self.encoder_type = encoder_type
        # 兼容旧配置 "unloc" -> "rrp"
        if decoder_type == "unloc":
            decoder_type = "rrp"
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
        if self.encoder_type == "res50":
            features, _ = self.res50_encoder(obs_img)
        elif self.encoder_type == "res50_3D":
            features, _ = self.res50_3D(obs_img)
        elif self.encoder_type == "res50_RSK":
            features, _ = self.res50_RSK(obs_img)
        elif self.encoder_type == "dptv2":
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
            self.dptv2_encoder = UnlocFeatureExtractor(checkpoint_path=self.config["dptv2_ckpt_path"])
        elif self.encoder_type == "res50":
            self.res50_encoder = depth_feature_res()
        elif self.encoder_type == "res50_3D":
            self.res50_3D = depth_feature_res(checkpoint_path=self.config["res50_3D_ckpt_path"])
        elif self.encoder_type == "res50_RSK":
            self.res50_RSK = depth_feature_res(checkpoint_path=self.config["res50_RSK_ckpt_path"])
    
    def _init_decoders(self):
        if self.decoder_type == "f3mlp":
            self.f3mlp_decoder = F3MlpDecoder()
        elif self.decoder_type == "rrp":
            self.rrp_decoder = F3MlpDecoder()