"""
图对比学习 + 半监督场景
"""
from collections import Counter, defaultdict

import torch
from lumo.calculate.tensor import onehot
from lumo.contrib import EMA
from lumo.contrib.nn.functional import batch_cosine_similarity
# from trainer import *
# from data import *
# from params import ParamsType
from lumo.contrib.nn.loss import contrastive_loss, sup_contrastive_loss, contrastive_loss2
from lumo.kit.trainer import TrainerResult
from lumo.nest.trainer.losses import MSELoss, L2Loss
from lumo import Params, Trainer, DataBundler, Meter, DataModule
from lumo.proc.path import cache_dir

from datasets.cifar import get_train_loader, get_val_loader
from wrn2 import WideResnet

# ParamsType = CCParams
torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)


class ParamsType(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 1700
        self.batch_size = 128
        self.num_workers = 8
        self.optim = self.OPTIM.create_optim('SGD', momentum=0.9, lr=0.03,
                                             weight_decay=5e-4,
                                             nesterov=True)
        self.weight_decay = 5e-4

        self.sample = 4
        self.dataset = self.choice('cifar10', 'cifar100')
        self.n_classes = 10
        self.n_percls = 4
        self.wview = 2
        self.sview = 2

        self.pretrain = True
        self.semiseed = 123
        self.limit_cls = -1

        # dataset params
        self.include_sup_in_un = True
        self.repeat_sup = True
        self.imb_type = self.choice('exp', 'step', 'none')
        self.imb_ratio = 0.02

        self.p_thresh = 0.9
        self.p_t = 1

        self.w_cs = 1
        self.unloader_c = 1
        self.ema = True
        self.ema_label = False  # 加了指标好看，但是降点

    def iparams(self):
        if self.dataset == 'cifar100':
            self.n_classes = 100
            self.weight_decay = 0.001

        if self.limit_cls > 0:
            self.n_classes = self.limit_cls


class GraphParams(ParamsType):

    def __init__(self):
        super().__init__()
        self.graph_fill_eye = True
        self.n_classes = 10
        self.ema = True
        self.detach_fc = False
        self.temperature = 0.2
        self.alpha = 0.9
        self.p_thresh = 0.95
        self.queue = 10
        self.cs_thresh = 0.8


ParamsType = GraphParams
from torch.nn import functional as F
from lumo import callbacks
from torch import nn


class CoMatch(Trainer, MSELoss, L2Loss, callbacks.InitialCallback, callbacks.TrainCallback):
    def norm(self, feature):
        return feature / torch.norm(feature, dim=-1, keepdim=True)

    def sim_matrix(self, a, b) -> torch.Tensor:
        """

        Args:
            a: [ba, dim]
            b: [bb, dim]

        Returns:
            [ba, bb]

        """
        aa = (a / a.norm(dim=-1, keepdim=True))
        bb = (b / b.norm(dim=-1, keepdim=True))
        return torch.mm(aa, bb.T)

    def on_train_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        self.logger.raw(f' - [x] {self.exp.test_name} {self.exp.test_root}')

    def icallbacks(self, params: ParamsType):
        callbacks.LoggerCallback(step_frequence=1, breakline_in=-1).hook(self)
        callbacks.AutoLoadModel().hook(self)
        callbacks.ScalarRecorder().hook(self)
        # callbacks.EMAUpdate().hook(self)

        if isinstance(self, callbacks.TrainCallback):
            self.hook(self)

    def evaluate_step(self, idx, batch, params: ParamsType, *args, **kwargs) -> Meter:
        meter = Meter()
        xs = batch['xs0']
        ys = batch['ys']
        logits = self.to_logits(xs)
        meter.mean.acc = (logits.argmax(dim=-1) == ys).float().mean()
        return meter

    def sharpen_(self, x: torch.Tensor, T=0.5):
        """
        让概率分布变的更 sharp，即倾向于 onehot
        :param x: prediction, sum(x,dim=-1) = 1
        :param T: temperature, default is 0.5
        :return:
        """
        with torch.no_grad():
            temp = torch.pow(x, 1 / T)
            return temp / temp.sum(dim=1, keepdims=True)

    def loss_ce_with_targets_masked_(self,
                                     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None,
                                     meter: Meter = None, name: str = "Ltce"):
        """
        用于 fixmatch 的 loss 计算方法，大 batch size 下天然可以对样本进行一个加权衰减
        :param logits:
        :param targets:
        :param mask:
        :param meter:
        :param name:
        :return:
        """
        _loss = torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1)
        if mask is not None:
            _loss = _loss * mask
        loss = -torch.mean(_loss)
        if meter is not None:
            meter[name] = loss
        return loss

    def on_train_step_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        super().on_train_step_begin(trainer, func, params, *args, **kwargs)
        if params.idx % 150 == 0:
            self.logger.newline()

    def on_prepare_dataloader_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_prepare_dataloader_end(trainer, func, params, meter, *args, **kwargs)
        res = self.getfuncargs(func, *args, **kwargs)
        stage = res['stage'].name
        if stage == 'train':
            self.lr_sche = params.SCHE.Cos(
                start=params.optim.lr, end=1e-7,
                left=0,
                right=len(self.train_dataloader) * params.epoch
            )

    def imodels(self, params: ParamsType):
        super().imodels(params)
        # self.model = build_wideresnet(num_classes=params.n_classes)
        self.model = WideResnet(n_classes=params.n_classes)
        feature_dim = self.model.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            # nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(feature_dim, 64),
        )

        self.graph_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 128),
        )

        self.prob_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, params.n_classes),
        )

        if params.ema:
            self.ema_model = EMA(self.model)
            self.ema_head = EMA(self.head)
            self.ema_graph = EMA(self.graph_head)
            self.ema_prob = EMA(self.prob_head)

        no_decay = ['bias', 'bn']

        wd_params, non_wd_params = [], []
        for name, param in nn.ModuleList([self.model, self.head]).named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)

        param_list = [
            {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        self.optim = params.optim.build(param_list)

        self.queue_list = []
        self.g_queue_list = []
        self.queue_prob = []

        self.counter = Counter()
        self.yscounter = Counter()

        self.cls_counter = defaultdict(Counter)

        self.to_device()

    def to_logits(self, xs) -> torch.Tensor:
        return self.model.forward(xs).last_hidden_state

    def test_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        meter = Meter()
        xs, ys = batch
        logits = self.ema_model.forward(xs).last_hidden_state
        meter.mean.Acc = (logits.argmax(dim=-1) == ys).float().mean()
        meter.sum.Ac = (logits.argmax(dim=-1) == ys).sum()
        meter.sum.All = len(ys)
        return meter

    # def metric_sim(self, graph, label):

    def train_step(self, idx, batch, params: ParamsType, *args, **kwargs):
        super().train_step(idx, batch, params, *args, **kwargs)
        meter = Meter()
        meter.mean.Lall = 0

        lbatch, unbatch = batch
        # idx = lbatch['idx0']
        # xs, ys = lbatch['xs0'], lbatch['ys0']
        # sxs = lbatch['sxs']
        # unxs, unys = unbatch['xs0'], unbatch['ys0']
        # unidx = unbatch['idx0']
        # unsxs_0 = unbatch['sxs']
        # unsxs_1 = unbatch['ssxs0']

        (xs, _, _), ys = lbatch
        (unxs, unsxs_0, unsxs_1), unys = unbatch

        # qxs = torch.cat([xs, unxs])
        # kxs = torch.cat([sxs_0, unsxs_0])
        axs = torch.cat([xs, unxs, unsxs_0, unsxs_1])
        # axs = interleave(axs, 2 * params.unloader_c + 1)
        # outputs = self.model.forward(axs, return_hidden_states=True)

        bt = xs.size(0)
        btu = unxs.size(0)

        outputs = self.model.forward(axs, return_hidden_states=True)
        logits = outputs.last_hidden_state
        logits_x = logits[:bt]
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt:], btu)

        features = self.head(outputs.hidden_states[-2])
        features = self.norm(features)
        feats_x = features[:bt]
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[bt:], btu)

        meter.mean.Lce = F.cross_entropy(logits_x, ys)

        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()

            probs = torch.softmax(logits_u_w, dim=1)
            probs_orig = probs.clone()

            if len(self.queue_list) > 0:
                A = torch.softmax(torch.mm(feats_u_w, torch.cat(self.queue_list).t()) / params.temperature, dim=-1)

                sim_prob = torch.mm(A, torch.cat(self.queue_prob))
                # sim_probs
                probs = params.alpha * probs + (1 - params.alpha) * sim_prob
                meter.mean.As = (sim_prob.argmax(dim=-1) == unys).float().mean()

            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(params.p_thresh)
            if mask.any():
                meter.mean.Amop = (probs_orig.argmax(dim=-1) == unys)[mask].float().mean()
                meter.mean.Ams = (sim_prob.argmax(dim=-1) == unys)[mask].float().mean()

            feats_w = torch.cat([feats_u_w, feats_x], dim=0)
            # onehot = torch.zeros(bt, params.n_classes).cuda().scatter(1, lbs_x.view(-1, 1), 1)
            probs_w = torch.cat([probs_orig, onehot(ys, params.n_classes)], dim=0)

            self.queue_list.append(feats_w.detach())
            self.queue_prob.append(probs_w.detach())
            if len(self.queue_list) > params.queue:
                self.queue_list.pop(0)
                self.queue_prob.pop(0)
            # update memory bank

        # embedding similarity
        # sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t()) / args.temperature)
        # sim_probs = sim / sim.sum(1, keepdim=True)

        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= params.cs_thresh).float()

        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)

        loss_contrast = contrastive_loss2(feats_u_s0, feats_u_s1, temperature=params.temperature,
                                          norm=False,
                                          qk_graph=Q, eye_one_in_qk=False)

        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0, dim=1) * probs), dim=1) * mask.float()
        loss_u = loss_u.mean()

        meter.Lall = meter.Lce + loss_u + loss_contrast
        meter.mean.Lu = loss_u
        meter.mean.Lcs = loss_contrast
        with torch.no_grad():
            meter.mean.Pm = pos_mask.float().mean()
            meter.mean.Ax = (logits_x.argmax(dim=-1) == ys).float().mean()
            meter.mean.um = mask.float().mean()
            meter.mean.Au = (logits_u_w.argmax(dim=-1) == unys).float().mean()
            if mask.any():
                meter.mean.Aum = (logits_u_w.argmax(dim=-1) == unys)[mask].float().mean()

        # # if params.train_sup:
        # self.queue_list.append(torch.cat([sup_query, un_query]).detach())
        # # self.g_queue_list.append(un_gquery.detach())
        # self.queue_prob.append(torch.cat([onehot(ys, params.n_classes), ori_un_prob]).detach())
        # if len(self.queue_list) > params.queue:
        #     self.queue_list.pop(0)
        #     self.queue_prob.pop(0)

        loss = meter.Lall
        self.optim.zero_grad()
        self.accelerator.backward(loss)
        self.optim.step()

        if params.ema:
            self.ema_model.step()
            self.ema_head.step()
            self.ema_prob.step()
            self.ema_graph.step()

        return meter

    def loss_l2_reg_(self, tensors: torch.Tensor, w_l2=1,
                     meter: Meter = None, name: str = 'Lreg'):

        loss = sum([(tensor ** 2).sum(dim=-1).mean() for tensor in tensors]) * w_l2
        if meter is not None:
            meter[name] = loss
        return loss

    def loss_ce_with_masked_(self,
                             logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                             meter: Meter = None, name: str = 'Lcem'):
        loss = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
        if meter is not None:
            meter[name] = loss
        return loss

    # def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
    #     super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
    #     if self.global_step % 100 == 0:
    #         self.test()
    #     if self.global_step % 1024 == 0:
    #         self.test()
    #     self.change_mode(train=True)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, result, *args, **kwargs)
        self.test()
        # self.logger.info(len(self.counter), len(self.yscounter))
        # self.logger.info(self.counter.most_common(100))
        # self.logger.info(self.yscounter.most_common(100))

        # self.logger.info(self.cls_counter)


def main():
    params = ParamsType()
    params.unloader_c = 7
    params.head_cls = False
    params.p_thresh = 0.95
    params.batch_size = 64
    params.limit_cls = -1
    params.n_percls = 40
    params.n_classes = 10
    params.from_args()

    dltrain_x, dltrain_u = get_train_loader(
        'CIFAR10', params.batch_size, params.unloader_c,
        1024, L=params.n_percls, root=cache_dir(), method='comatch')
    dlval = get_val_loader(dataset='CIFAR10', batch_size=128, num_workers=2, root=cache_dir())

    db = DataBundler().cycle(dltrain_x).add(dltrain_u).zip_mode()

    dm = DataModule(train=db, test=dlval)

    trainer = CoMatch(params)
    trainer.train(dm)


if __name__ == '__main__':
    main()
