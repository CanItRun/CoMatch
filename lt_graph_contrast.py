"""
图对比学习 + 有监督场景
"""
from collections import Counter, defaultdict

import torch
from ema import EMA
from lumo.contrib.nn.functional import batch_cosine_similarity
from lumo.contrib.nn.loss import contrastive_loss, sup_contrastive_loss, contrastive_loss2
from lumo.kit.trainer import TrainerResult
from lumo.nest.trainer.losses import MSELoss, L2Loss
from lumo import Params, Trainer, DataBundler, Meter, DataModule
from lumo.proc.path import cache_dir

from datasets.cifar import get_train_loader, get_val_loader, get_train_loader2
from wrn2 import WideResnet

torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)


def choice_(tensor, size=128):
    return tensor[torch.randperm(len(tensor))[:size]]


class ParamsType(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 15
        self.imb_factor = 0.01
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

        self.sharp_r = 1
        self.graph_s = 0
        self.graph_head = True

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
        self.s = 0
        self.cs_thresh = 0.8
        self.thr = 0.95

        self.contrast_th = 0.8


ParamsType = GraphParams
from torch.nn import functional as F
from lumo import callbacks
from torch import nn


class SupContrast(Trainer, MSELoss, L2Loss, callbacks.InitialCallback, callbacks.TrainCallback):
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
        feature_dim = 64 * self.model.k
        self.feature_dim = feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            # nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(feature_dim, 64),
        )

        self.graph_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(feature_dim, 128),
        )

        self.prob_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 128),
        )

        if params.ema:
            self.ema_model = EMA(self.model)
            self.ema_head = EMA(self.head)
            self.ema_graph = EMA(self.graph_head)
            self.ema_prob = EMA(self.prob_head)

        no_decay = ['bias', 'bn']

        wd_params, non_wd_params = [], []
        for name, param in nn.ModuleList([self.model,
                                          self.head,
                                          self.graph_head, self.prob_head]).named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)

        param_list = [
            {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        self.optim = params.optim.build(param_list)

        self.queue_list = []
        self.da_prob_list = []
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
        Lgcs2 = 0
        lbatch, unbatch = batch
        # idx = lbatch['idx0']
        xs, ys = lbatch['xs0'], lbatch['ys']
        xs_s0, xs_s1 = lbatch['sxs0'], lbatch['sxs1']

        unxs, unys = unbatch['xs0'], unbatch['ys']
        unsxs_0, unsxs_1 = unbatch['sxs0'], unbatch['sxs1']

        axs = torch.cat([xs, xs_s0, xs_s1, unxs, unsxs_0, unsxs_1])

        bt = xs.size(0)
        btu = unxs.size(0)

        outputs = self.model.forward(axs)

        features = self.head(outputs.hidden_states[-2])
        features = self.norm(features)

        logits = self.model.classifier(outputs.hidden_states[-2].detach())

        # logits_x, _, _ = logits[:bt * 3].chunk(3)
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt * 3:], btu)

        sup_w_query, _, _ = features[:bt * 3].chunk(3)
        un_w_query, un_query, un_key = torch.split(features[bt * 3:], btu)

        def graph_feature0():
            """沿用对比学习 head"""
            self.exp.add_tag('graph_feature1')
            sup_w_query, sup_query, sup_key = features[:bt * 3].chunk(3)
            un_w_query, un_query, un_key = torch.split(features[bt * 3:], btu)
            return (sup_w_query, sup_query, sup_key), (un_w_query, un_query, un_key)

        def graph_feature1():
            """
            用额外的映射头
            Returns:

            """
            self.exp.add_tag('graph_feature2')
            gfeatures = self.prob_head(outputs.hidden_states[-2])
            sup_w_query, sup_query, sup_key = gfeatures[:bt * 3].chunk(3)
            un_w_query, un_query, un_key = torch.split(gfeatures[bt * 3:], btu)
            return (sup_w_query, sup_query, sup_key), (un_w_query, un_query, un_key)

        if params.graph_s == 0:
            (sup_w_gquery, sup_gquery, sup_gkey), (un_w_gquery, un_gquery, un_gkey) = graph_feature0()
        elif params.graph_s == 1:
            (sup_w_gquery, sup_gquery, sup_gkey), (un_w_gquery, un_gquery, un_gkey) = graph_feature1()
        else:
            raise NotImplementedError()

        Q = unys.unsqueeze(0) == unys.unsqueeze(1)

        def sup_contrast():
            self.exp.add_tag('Lcs')
            loss = contrastive_loss2(un_query, un_key,
                                     memory=None,
                                     norm=True,
                                     temperature=0.2,
                                     qk_graph=Q,
                                     )
            meter.mean.Lcs = loss
            return loss

        def graph_cs2():
            # ./log/l.0.2110191754.log
            self.exp.add_tag('sup_gcs')
            memory = torch.cat(self.g_queue_list)
            pos_memory = choice_(memory, self.feature_dim)

            anchor = batch_cosine_similarity(sup_gquery, pos_memory)
            positive = batch_cosine_similarity(sup_gkey, pos_memory)
            if params.graph_head:
                anchor, positive = self.graph_head(torch.cat([anchor, positive])).chunk(2)

            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True,
                                     temperature=0.2,
                                     qk_graph=(ys.unsqueeze(0) == ys.unsqueeze(1)),
                                     eye_one_in_qk=False)
            meter.mean.Lgcs2 = loss
            return loss

        def graph_cs3():
            # ./log/l.0.2110191754.log
            self.exp.add_tag('un_gcs')
            memory = torch.cat(self.g_queue_list)
            pos_memory = choice_(memory, self.feature_dim)

            anchor = batch_cosine_similarity(un_gquery, pos_memory)
            positive = batch_cosine_similarity(un_gkey, pos_memory)
            if params.graph_head:
                anchor, positive = self.graph_head(torch.cat([anchor, positive])).chunk(2)

            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True,
                                     temperature=0.2,
                                     qk_graph=Q,
                                     eye_one_in_qk=False)
            meter.mean.Lgcs = loss
            return loss

        Lce = F.cross_entropy(logits_u_s1, unys)

        def strategy0():
            """sup contrast"""

            return Lce + sup_contrast()

        def strategy1():
            """graph contrast"""
            if len(self.queue_list) > 0:
                Lgcs2 = graph_cs3() + graph_cs2()
                return Lgcs2 + Lce
            return Lce

        def strategy2():
            """sup+graph contrast"""
            if len(self.queue_list) > 0:
                Lgcs2 = graph_cs3() + graph_cs2()
                return Lgcs2 + sup_contrast() + Lce
            return sup_contrast() + Lce

        if params.s == 0:
            loss = strategy0()
        elif params.s == 1:
            loss = strategy1()
        elif params.s == 2:
            loss = strategy2()
        else:
            loss = 0

        self.optim.zero_grad()
        self.accelerator.backward(loss)
        self.optim.step()
        self.lr_sche.apply(self.optim, self.global_step)

        if params.ema:
            self.ema_model.step()
            self.ema_head.step()
            self.ema_prob.step()
            self.ema_graph.step()

        with torch.no_grad():
            meter.mean.Lall = loss
            meter.mean.Lce = Lce
            meter.mean.Au = (logits_u_w.argmax(dim=-1) == unys).float().mean()

            feats_w = torch.cat([un_w_query, sup_w_query], dim=0)
            gfeats_w = torch.cat([un_w_gquery, sup_w_gquery], dim=0)
            self.queue_list.append(feats_w)
            self.g_queue_list.append(gfeats_w)
            if len(self.queue_list) > params.queue:
                self.queue_list.pop(0)
                self.g_queue_list.pop(0)
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

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, result: TrainerResult, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, result, *args, **kwargs)
        self.test()
        self.save_model()

    def on_train_step_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_step_end(trainer, func, params, meter, *args, **kwargs)
        if (params.idx + 1) % 150 == 0:
            self.test()
            self.change_mode()


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

    trainer = SupContrast(params)
    trainer.rnd.mark('12')
    dltrain_x, dltrain_u = get_train_loader2(
        'CIFAR10', params.batch_size, params.unloader_c,
        1024, L=params.n_percls, root=cache_dir(), method='comatch', long_tail=True, imb_factor=params.imb_factor)
    dlval = get_val_loader(dataset='CIFAR10', batch_size=128, num_workers=2, root=cache_dir())

    db = DataBundler().cycle(dltrain_x).add(dltrain_u).zip_mode()

    dm = DataModule(train=db, test=dlval)

    trainer.train(dm)


if __name__ == '__main__':
    main()
