import torch

from data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import json
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import warnings
# PIL.Image.MAX_IMAGE_PIXELS = None # to ignore DecompressionBombWarning, but will increase RAM usage
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class AlbefCombiner(nn.Module):

    def __init__(self, combiner: nn.Module, combiner_m:  nn.Module, loss_ids: list, alpha=0.4, margin=0.2, momentum=0.995):
        """
        :param blip_feature_dim: BLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(AlbefCombiner, self).__init__()

        self.momentum = momentum
        self.temp = 0.07
        self.mlm_probability = 0.15
        self.alpha = alpha
        self.queue_size = 16000
        self.embed_dim = 256
        self.model_pairs = [[combiner, combiner_m]]

        self.combiner = combiner
        self.combiner_m = combiner_m
        self.loss_id = loss_ids
        self.margin = margin
        self.copy_params()

        # create the queue
        self.register_buffer("fusion_queue", torch.randn(512, self.queue_size))
        self.register_buffer("target_queue", torch.randn(512, self.queue_size))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.fusion_queue = F.normalize(self.fusion_queue, dim=0)
        self.target_queue = F.normalize(self.target_queue, dim=0)

    def forward(self, reference_features, target_features, text_features) -> torch.tensor:

        # 获取基本特征
        target_feat = F.normalize(target_features, dim=-1)
        fusion_feature = self.combiner.combine_features(reference_features, text_features)

        # 获取动量 特征
        with torch.no_grad():
            self._momentum_update()
            target_feat_m = target_feat

            fusion_feature_m = self.combiner_m.combine_features(reference_features, text_features)
            fusion_feature_all = torch.cat([fusion_feature_m.t(), self.fusion_queue.clone().detach()], dim=1)

            target_feat_all = torch.cat([target_feat_m.t(), self.target_queue.clone().detach()], dim=1)
            sim_i2t_m = fusion_feature_m @ target_feat_all * 100
            sim_t2i_m = target_feat_m @ fusion_feature_all * 100

            sim_targets = torch.zeros(sim_i2t_m.size()).to(target_feat.device)
            sim_targets.fill_diagonal_(1)
            sim_i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets
            sim_t2i_targets = self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets

        self._dequeue_and_enqueue(fusion_feature_m, target_feat_m)

        loss = 0
        if 1 in self.loss_ids:
            sim_i2t = fusion_feature @ target_feat_all * 100
            sim_t2i = target_feat @ fusion_feature_all * 100
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss = loss + loss_i2t + loss_t2i

        if 2 in self.loss_ids:
            if 1 not in self.loss_ids:
                sim_i2t = fusion_feature @ target_feat_all * 100
                sim_t2i = target_feat @ fusion_feature_all * 100
            # ================= MLM ========================
            with torch.no_grad():
                bs = target_feat.size(0)
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

                # 对角线 变成0， 剩下的找最相似的。
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)

            target_embedding_neg = []  # 获得相对于fusion为最难区分的negative target_image
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # 依次每一行采样一个样本，数值高的更容易被采样。
                # 挑选出来 最难区分的 样本
                target_embedding_neg.append(target_feat[neg_idx])  # 每个元素形状为(3,512)
            target_embedding_neg = torch.stack(target_embedding_neg, dim=0)  # 按照第0维度堆叠，形状为(2,3,512)

            # select a negative fusion feature for each target
            fusion_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                fusion_embeds_neg.append(fusion_feature[neg_idx])
            fusion_embeds_neg = torch.stack(fusion_embeds_neg, dim=0)
            loss += F.mse_loss(fusion_feature, fusion_feature_m) + self.tripletRankingLoss(fusion_feature, target_feat, target_embedding_neg, self.margin) + self.tripletRankingLoss(target_feat,fusion_feature,fusion_embeds_neg, self.margin)

        if 3 in self.loss_ids:
            ground_truth = torch.arange(target_feat.size(0), dtype=torch.long, device=target_feat.device)
            loss += F.cross_entropy(fusion_feature @ target_feat.T * 100, ground_truth)

        return loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        # gather keys before updating queue

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            ptr = 0

        # replace the keys at ptr (dequeue and enqueue)
        self.fusion_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.target_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr