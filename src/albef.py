import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x



class TripletRankingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletRankingLoss, self).__init__()
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, ref, pos, neg):
        x1 = nn.functional.cosine_similarity(ref, pos, dim=1)
        x2 = nn.functional.cosine_similarity(ref, neg, dim=1)
        target = torch.FloatTensor(ref.size(0)).fill_(1)
        target = target.to(ref.device)
        loss = self.criterion(x1, x2, target)
        return loss

class Albef(nn.Module):

    def __init__(self, clip_model: nn.Module, clip_model_m:  nn.Module, loss_ids: list, alpha=0.4, margin=0.2, momentum=0.995, queue_size=16000):
        """
        :param blip_feature_dim: BLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Albef, self).__init__()

        self.momentum = momentum
        self.temp = 0.07
        self.mlm_probability = 0.15
        self.alpha = alpha
        self.queue_size = queue_size
        self.embed_dim = 256
        self.margin = margin
        self.model_pairs = [[clip_model, clip_model_m]]
        self.clip_model = clip_model
        self.clip_model_m = clip_model_m
        self.loss_ids = loss_ids
        self.copy_params()
        self.tripletRankingLoss = TripletRankingLoss(margin=margin)

        # create the queue
        self.register_buffer("fusion_queue", torch.randn(512, self.queue_size))
        self.register_buffer("target_queue", torch.randn(512, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # normalize queue
        self.fusion_queue = F.normalize(self.fusion_queue, dim=0)
        self.target_queue = F.normalize(self.target_queue, dim=0)

    def forward(self, reference_images, target_image, text_embedding, combine_funcation: callable) -> torch.tensor:

        # get base features
        ref_feat = self.clip_model.get_image_features(**reference_images)
        text_feat = self.clip_model.get_text_features(**text_embedding)
        target_feat = self.clip_model.get_image_features(**target_image)

        # fusion features
        fusion_feature = combine_funcation(ref_feat, text_feat)
        target_feat = F.normalize(target_feat, dim=-1)

        loss = 0

        # 获取动量 特征
        with torch.no_grad():
            self._momentum_update()

            image_feat_m = self.clip_model_m.get_image_features(**reference_images)
            text_feat_m = self.clip_model_m.get_text_features(**text_embedding)
            target_feat_m = self.clip_model_m.get_image_features(**target_image)

            target_feat_m = F.normalize(target_feat_m, dim=-1)
            fusion_feature_m = combine_funcation(image_feat_m, text_feat_m)

            fusion_feature_all = torch.cat([fusion_feature_m.t(), self.fusion_queue.clone().detach()], dim=1)
            target_feat_all = torch.cat([target_feat_m.t(), self.target_queue.clone().detach()], dim=1)

            sim_i2t_m = fusion_feature_m @ target_feat_all * 100
            sim_t2i_m = target_feat_m @ fusion_feature_all * 100

            sim_targets = torch.zeros(sim_i2t_m.size()).to(target_feat.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets
            sim_t2i_targets = self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets

        self._dequeue_and_enqueue(fusion_feature_m, target_feat_m)

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

            loss += F.mse_loss(fusion_feature, fusion_feature_m) + F.mse_loss(target_feat, target_feat_m) + self.tripletRankingLoss(fusion_feature, target_feat, target_embedding_neg) + self.tripletRankingLoss(target_feat, fusion_feature, fusion_embeds_neg)

            # loss += F.triplet_margin_loss(fusion_feature, target_feat, target_embedding_neg, margin=self.margin) + F.triplet_margin_loss(target_feat, fusion_feature, fusion_embeds_neg, margin=self.margin)

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
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            ptr = 0
        # replace the keys at ptr (dequeue and enqueue)
        self.fusion_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.target_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr