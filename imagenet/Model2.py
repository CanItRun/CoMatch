import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import math

from lumo.contrib.nn.loss import contrastive_loss2


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Wrapper(nn.Module):
    def __init__(self, backbone, low_dim, num_class, mlp=True):
        super().__init__()
        feature_num = self.encoder.feature_num
        self.encoder = backbone
        self.classifier = nn.Linear(feature_num, num_class)
        # self.l2norm = Normalize(2)
        self.mlp = mlp
        if self.mlp:  # use an extra projection layer
            self.head = nn.Sequential(
                nn.Linear(feature_num, 2048),
                nn.ReLU(),
                nn.Linear(2048, low_dim),
                Normalize(2),
            )
            # self.fc1 = nn.Linear(feature_num, 2048)
            # self.fc2 =
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_num, low_dim),
                Normalize(2)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, xs):
        feature_map = self.encoder(xs)
        logits = self.classifier(feature_map)
        feature = self.head(feature_map)
        return logits, feature


class Model(nn.Module):

    def __init__(self, base_encoder, args, width):

        super(Model, self).__init__()

        self.K = args.K

        self.encoder = Wrapper(base_encoder(num_class=args.num_class, mlp=True, low_dim=args.low_dim, width=width),
                               low_dim=args.low_dim, num_class=args.num_class, mlp=True)
        self.m_encoder = Wrapper(base_encoder(num_class=args.num_class, mlp=True, low_dim=args.low_dim, width=width),
                                 low_dim=args.low_dim, num_class=args.num_class, mlp=True)

        for param, param_m in zip(self.encoder.parameters(), self.m_encoder.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        # queue to store momentum feature for strong augmentations
        self.register_buffer("queue_s", torch.randn(args.low_dim, self.K))
        self.register_buffer("queue_y", torch.zeros(self.K))
        self.queue_s = F.normalize(self.queue_s, dim=0)
        # self.queue_y = F.normalize(self.queue_s, dim=0)
        self.register_buffer("queue_ptr_s", torch.zeros(1, dtype=torch.long))
        # queue to store momentum probs for weak augmentations (unlabeled)
        self.register_buffer("probs_u", torch.zeros(args.num_class, self.K))

        # queue (memory bank) to store momentum feature and probs for weak augmentations (labeled and unlabeled)
        self.register_buffer("queue_w", torch.randn(args.low_dim, self.K))
        self.register_buffer("queue_ptr_w", torch.zeros(1, dtype=torch.long))
        self.register_buffer("probs_xu", torch.zeros(args.num_class, self.K))

        # for distribution alignment
        self.hist_prob = []

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """
        Update momentum encoder
        """
        for param, param_m in zip(self.encoder.parameters(), self.m_encoder.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z, t, ws, pys=None):
        z = concat_all_gather(z)
        t = concat_all_gather(t)

        batch_size = z.shape[0]

        if ws == 's':
            ptr = int(self.queue_ptr_s)

            if (ptr + batch_size) > self.K:
                batch_size = self.K - ptr
                z = z[:batch_size]
                t = t[:batch_size]
                # replace the samples at ptr (dequeue and enqueue)
            self.queue_s[:, ptr:ptr + batch_size] = z.T
            self.probs_u[:, ptr:ptr + batch_size] = t.T
            if pys is not None:
                self.queue_y[ptr:ptr + batch_size] = pys
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr_s[0] = ptr

        elif ws == 'w':
            ptr = int(self.queue_ptr_w)
            if (ptr + batch_size) > self.K:
                batch_size = self.K - ptr
                z = z[:batch_size]
                t = t[:batch_size]
                # replace the samples at ptr (dequeue and enqueue)
            self.queue_w[:, ptr:ptr + batch_size] = z.T
            self.probs_xu[:, ptr:ptr + batch_size] = t.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr_w[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, args, labeled_batch, unlabeled_batch=None, is_eval=False, epoch=0):

        img_x = labeled_batch[0].cuda(args.gpu, non_blocking=True)
        labels_x = labeled_batch[1].cuda(args.gpu, non_blocking=True)

        if is_eval:
            outputs_x, _ = self.encoder(img_x)
            return outputs_x, labels_x

        btx = img_x.size(0)

        img_u_w = unlabeled_batch[0][0].cuda(args.gpu, non_blocking=True)
        img_u_s0 = unlabeled_batch[0][1].cuda(args.gpu, non_blocking=True)
        img_u_s1 = unlabeled_batch[0][2].cuda(args.gpu, non_blocking=True)

        btu = img_u_w.size(0)

        imgs = torch.cat([img_x, img_u_w, img_u_s0], dim=0)
        outputs, features = self.encoder(imgs)

        outputs_x = outputs[:btx]
        outputs_u_w, outputs_u_s0 = outputs[btx:].chunk(2)
        features_u_w, features_u_s0 = features[btx:].chunk(2)

        with torch.no_grad():
            self._update_momentum_encoder(args.m)
            # forward through the momentum encoder
            imgs_m = torch.cat([img_x, img_u_w, img_u_s1], dim=0)
            imgs_m, idx_unshuffle = self._batch_shuffle_ddp(imgs_m)

            outputs_m, features_m = self.m_encoder(imgs_m)
            outputs_m = self._batch_unshuffle_ddp(outputs_m, idx_unshuffle)
            features_m = self._batch_unshuffle_ddp(features_m, idx_unshuffle)

            outputs_u_w = outputs_m[btx:btx + btu]

            feature_u_w = features_m[btx:btx + btu]
            feature_xu_w = features_m[:btx + btu]
            features_u_s1 = features_m[btx + btu:]

            outputs_u_w = outputs_u_w.detach()
            feature_u_w = feature_u_w.detach()
            feature_xu_w = feature_xu_w.detach()
            features_u_s1 = features_u_s1.detach()

            un_w_probs = torch.softmax(outputs_u_w, dim=1)

            # distribution alignment
            probs_bt_avg = un_w_probs.mean(0)
            torch.distributed.all_reduce(probs_bt_avg, async_op=False)
            self.hist_prob.append(probs_bt_avg / args.world_size)

            if len(self.hist_prob) > 128:
                self.hist_prob.pop(0)

            probs_avg = torch.stack(self.hist_prob, dim=0).mean(0)
            un_w_probs = un_w_probs / probs_avg
            un_w_probs = un_w_probs / un_w_probs.sum(dim=1, keepdim=True)
            probs_orig = un_w_probs.clone()

            scores, un_w_pys = torch.max(un_w_probs, dim=1)
            # memory-smoothed pseudo-label refinement (starting from 2nd epoch)
            # if epoch > 0:
            #     m_feat_xu = self.queue_w.clone().detach()
            #     m_probs_xu = self.probs_xu.clone().detach()
            #     A = torch.exp(torch.mm(feature_u_w, m_feat_xu) / args.temperature)
            #     A = A / A.sum(1, keepdim=True)
            #     un_w_probs = args.alpha * un_w_probs + (1 - args.alpha) * torch.mm(A, m_probs_xu.t())

            # construct pseudo-label graph

            # similarity with current batch
            # Q_self = torch.mm(un_w_probs, un_w_probs.t())
            # Q_self.fill_diagonal_(1)

            # similarity with past samples
            # m_probs_u = self.probs_u.clone().detach()
            # Q_past = torch.mm(un_w_probs, m_probs_u)

            # concatenate them
            # Q = torch.cat([Q_self, Q_past], dim=1)

        # construct embedding graph for strong augmentations
        # sim_self = torch.exp(torch.mm(features_u_s0, features_u_s1.t()) / args.temperature)
        m_feat = self.queue_s.clone().detach()
        m_pys = self.queue_y.clone().detach()
        # sim_past = torch.exp(torch.mm(features_u_s0, m_feat) / args.temperature)
        # sim = torch.cat([sim_self, sim_past], dim=1)

        # store strong augmentation features and un_w_probs (unlabeled) into momentum queue
        self._dequeue_and_enqueue(features_u_s1, un_w_probs, 's', un_w_pys)

        # store weak augmentation features and un_w_probs (labeled and unlabeled) into memory bank
        onehot = torch.zeros(btx, args.num_class).cuda().scatter(1, labels_x.view(-1, 1), 1)
        probs_xu = torch.cat([onehot, probs_orig], dim=0)

        self._dequeue_and_enqueue(feature_xu_w, probs_xu, 'w')

        return ((outputs_x, outputs_u_w, outputs_u_s0, labels_x, un_w_probs, None, None),
                (features_u_s0, features_u_s1, m_feat, un_w_pys, m_pys))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
