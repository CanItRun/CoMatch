'''
 * Copyright (c) 2018, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
from __future__ import print_function
import random

import time
import argparse
import os
import sys
from lumo.contrib.nn.loss import contrastive_loss, sup_contrastive_loss, contrastive_loss2
from lumo.contrib.nn.functional import batch_cosine_similarity

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lumo.proc.path import cache_dir

from WideResNet2 import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler

from lumo import Logger, Meter, AvgMeter, TrainerExperiment

exp = TrainerExperiment('graph_comatch.CoMatch')
exp.start()

log = Logger()
log.add_log_dir(exp.test_root)


def set_model(args):
    model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        msg = model.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint: %s' % args.checkpoint)
    model.train()
    model.cuda()

    if args.eval_ema:
        ema_model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()
        ema_model.eval()
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    return model, criteria_x, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)


queue_ptr = 0
graph_queue_feats = []


def train_one_epoch(epoch,
                    model,
                    ema_model,
                    prob_list,
                    criteria_x,
                    optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    queue_feats,
                    queue_probs,

                    dlval=None
                    ):
    global queue_ptr
    model.train()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)

    avg = AvgMeter()
    for it in range(1, n_iters + 1):
        meter = Meter()
        loss_contrast = 0
        loss_contrast2 = 0
        Lgcs = 0
        Lgcs2 = 0
        Lgcs3 = 0

        (xs, sxs_0, sxs_1), ys = next(dl_x)
        (uxs, usxs_0, usxs_1), unys = next(dl_u)

        ys = ys.cuda()
        unys = unys.cuda()

        # --------------------------------------
        bt = xs.size(0)
        btu = uxs.size(0)

        axs = torch.cat([xs, sxs_0, sxs_1, uxs, usxs_0, usxs_1], dim=0).cuda()
        logits, features, graph_features = model(axs)

        logits_x, _, _ = logits[:bt * 3].chunk(3)
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt * 3:], btu)

        _, sup_query, sup_key = features[:bt * 3].chunk(3)
        un_w_query, un_query, un_key = torch.split(features[bt * 3:], btu)

        _, sup_gquery, sup_gkey = graph_features[:bt * 3].chunk(3)
        un_w_gquery, un_gquery, un_gkey = torch.split(graph_features[bt * 3:], btu)

        loss_x = criteria_x(logits_x, ys)

        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            un_probs = torch.softmax(logits_u_w, dim=1)
            sup_probs = torch.softmax(logits_x, dim=1)
            # DA
            prob_list.append(un_probs.mean(0))
            if len(prob_list) > 32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            un_probs = un_probs / prob_avg
            un_probs = un_probs / un_probs.sum(dim=1, keepdim=True)

            probs_orig = un_probs.clone()

            if epoch > 0 or it > args.queue_batch:  # memory-smoothing
                # A = torch.exp(torch.mm(feats_u_w, queue_feats.t()) / args.temperature)
                # A = A / A.sum(1, keepdim=True)  # 概率分布
                A = torch.softmax(torch.mm(un_w_query, queue_feats.t()) / args.temperature, dim=-1)

                sim_prob = torch.mm(A, queue_probs)
                # sim_probs
                un_probs = args.alpha * un_probs + (1 - args.alpha) * sim_prob
                meter.mean.As = (sim_prob.argmax(dim=-1) == unys).float().mean()

            scores, lbs_u_guess = torch.max(un_probs, dim=1)
            mask = scores.ge(args.thr)
            if mask.any():
                meter.mean.Amop = (probs_orig.argmax(dim=-1) == unys)[mask].float().mean()
                meter.mean.Ams = (sim_prob.argmax(dim=-1) == unys)[mask].float().mean()

            feats_w = torch.cat([un_w_query, sup_query], dim=0)
            onehot = torch.zeros(bt, args.n_classes).cuda().scatter(1, ys.view(-1, 1), 1)
            probs_w = torch.cat([probs_orig, onehot], dim=0)

            # update memory bank
            n = bt + btu
            queue_feats[queue_ptr:queue_ptr + n, :] = feats_w
            queue_probs[queue_ptr:queue_ptr + n, :] = probs_w
            queue_ptr = (queue_ptr + n) % args.queue_size

            gfeats_w = torch.cat([un_w_gquery, sup_gquery], dim=0)
            graph_queue_feats.append(gfeats_w)
            if len(graph_queue_feats) > args.queue_size:
                graph_queue_feats.pop(0)

        # pseudo-label graph with self-loop
        qk_graph = torch.mm(un_probs, un_probs.t())
        qk_graph.fill_diagonal_(1)
        pos_mask = (qk_graph >= args.contrast_th).float()

        qk_graph = qk_graph * pos_mask
        qk_graph = qk_graph / qk_graph.sum(1, keepdim=True)

        #
        # probs = torch.cat([sup_probs, un_probs])
        # qk_graph = torch.mm(probs, probs.t())
        # pos_mask = (qk_graph >= args.contrast_th)
        #
        #
        #
        # pos_mask = pos_mask * label_graph  # 将 有监督部份不相等的标签的强制抹零
        #
        # qk_graph.fill_diagonal_(1)
        # qk_graph = qk_graph * pos_mask.float()
        # qk_graph = qk_graph / (qk_graph.sum(1, keepdim=True) + 1e-10)

        # loss_contrast = contrastive_loss2(un_query, un_key, temperature=args.temperature,
        #                                   norm=False, qk_graph=qk_graph)

        # semi cs
        def semi_cs():
            query, key = torch.cat([sup_query, un_query]), torch.cat([sup_key, un_key])
            qys = torch.cat([ys, -torch.ones_like(unys)])
            label_graph = qys.unsqueeze(0) == qys.unsqueeze(1)
            label_graph[len(ys):] = 0
            label_graph[:, len(ys):] = 0
            label_graph.fill_diagonal_(1)

            loss = contrastive_loss2(query, key, temperature=args.temperature,
                                     norm=False, qk_graph=label_graph)
            return loss

        # graph cs
        def graph_cs():
            memory = queue_feats

            memory = memory[torch.randperm(len(memory))[:len(un_query)]]

            # ./log/l.0.2110191736.log
            anchor = batch_cosine_similarity(un_gquery, un_gquery)
            positive = batch_cosine_similarity(un_gkey, un_gkey)
            negative = batch_cosine_similarity(memory, memory)

            # ./log/l.0.2110191739.log
            # anchor = batch_cosine_similarity(un_query, un_query)
            # positive = batch_cosine_similarity(un_query, un_key)
            # negative = batch_cosine_similarity(un_query, memory)

            g_mask = (1 - torch.eye(len(un_query), dtype=torch.float, device=anchor.device))
            anchor = anchor * g_mask
            positive = positive * g_mask
            negative = negative * g_mask

            loss = contrastive_loss2(anchor, positive, negative, norm=True, temperature=args.temperature)
            return loss

        # Lgcs = graph_cs()

        # graph cs2
        def graph_cs2():
            # ./log/l.0.2110191754.log

            # memory = queue_feats
            memory = torch.cat([un_query, un_key, queue_feats])
            pos_memory = memory[torch.randperm(len(memory))[:512]]
            neg_memory = memory[torch.randperm(len(memory))[:512]]
            # memory = torch.cat([un_query, un_key, memory])

            anchor = batch_cosine_similarity(sup_query, pos_memory)
            positive = batch_cosine_similarity(sup_key, pos_memory)
            # negative = batch_cosine_similarity(sup_key, neg_memory)
            gqk = ys.unsqueeze(0) == ys.unsqueeze(1)
            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True, temperature=0.2, qk_graph=gqk)
            return loss * 0.5

        def graph_cs3():
            # ./log/l.0.2110191754.log

            # memory = queue_feats
            memory = torch.cat([sup_query, sup_key, queue_feats])
            pos_memory = memory[torch.randperm(len(memory))[:512]]
            neg_memory = memory[torch.randperm(len(memory))[:512]]

            anchor = batch_cosine_similarity(un_query, pos_memory)
            positive = batch_cosine_similarity(un_key, pos_memory)
            # negative = batch_cosine_similarity(un_key, neg_memory)

            # gqk = ys.unsqueeze(0) == ys.unsqueeze(1)
            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True, temperature=0.2, qk_graph=qk_graph)
            return loss * 0.5
            # contrastive loss
            # loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
            # loss_contrast = loss_contrast.mean()

        if len(graph_queue_feats) > 0:
            Lgcs2 = graph_cs2()
            Lgcs3 = graph_cs3()

        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0, dim=1) * un_probs), dim=1) * mask.float()
        loss_u = loss_u.mean()

        loss = loss_x + args.lam_u * loss_u + args.lam_c * loss_contrast + loss_contrast2 + Lgcs + Lgcs2 + Lgcs3

        meter.mean.Lall = loss
        meter.mean.Lx = loss_x
        meter.mean.Lu = loss_u
        meter.mean.Lcs = loss_contrast
        meter.mean.Lscs = loss_contrast2
        # meter.mean.Lgcs = Lgcs
        meter.mean.Lgcs2 = Lgcs2
        meter.mean.Lgcs3 = Lgcs3
        with torch.no_grad():
            meter.mean.Pm = pos_mask.float().mean()
            meter.mean.Ax = (logits_x.argmax(dim=-1) == ys).float().mean()
            meter.mean.um = mask.float().mean()
            meter.mean.Au = (logits_u_w.argmax(dim=-1) == unys).float().mean()
            if mask.any():
                meter.mean.Aum = (logits_u_w.argmax(dim=-1) == unys)[mask].float().mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)

        if mask.any():
            corr_u_lb = (lbs_u_guess == unys).float()[mask]
            meter.mean.Corr = corr_u_lb.mean()
        # meter.mean.CM = mask.float().mean()
        avg.update(meter)
        log.inline(f'{it}/{n_iters}', avg)
        if it % 150 == 0:
            # evaluate(model, ema_model, dlval)
            # model.train()
            # avg.clear()
            log.newline()
    avg.clear()
    log.newline()
    return loss_x_meter.avg, loss_u_meter.avg, loss_contrast_meter.avg, mask_meter.avg, pos_meter.avg, n_correct_u_lbs_meter, queue_feats, queue_probs, queue_ptr, prob_list


def evaluate(model, ema_model, dataloader):
    model.eval()
    avg = AvgMeter()
    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs in dataloader:
            meter = Meter()
            ims = ims.cuda()
            lbs = lbs.cuda()

            logits, *_ = model(ims)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            meter.sum.A1 = top1
            meter.sum.A5 = top5

            if ema_model is not None:
                logits, *_ = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                ema_top1_meter.update(top1.item())
                meter.sum.Ae1 = top1
                meter.sum.Ae5 = top5
            avg.update(meter)
    log.info('TEST', avg)
    return top1_meter.avg, ema_top1_meter.avg


def main():
    parser = argparse.ArgumentParser(description='CoMatch Cifar Training')
    parser.add_argument('--root', default=cache_dir(), type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=512,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')

    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random behaviors, no seed if negtive')

    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--queue-batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='CoMatch', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--device', default=0, type=int, help='use pretrained model')

    args = parser.parse_args()

    from torch.cuda import set_device
    set_device(args.device)

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    # tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 200

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")

    model, criteria_x, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_x, dltrain_u = get_train_loader(
        args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root, method='comatch')
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    # memory bank
    args.queue_size = args.queue_batch * (args.mu + 1) * args.batchsize
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, args.n_classes).cuda()
    queue_ptr = 0

    # for distribution alignment
    prob_list = []

    train_args = dict(
        model=model,
        ema_model=ema_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )

    best_acc = -1
    best_epoch = 0
    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):

        train_one_epoch(epoch, **train_args, queue_feats=queue_feats, queue_probs=queue_probs,
                        dlval=dlval)

        top1, ema_top1 = evaluate(model, ema_model, dlval)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))

        if epoch % 10 == 0:
            save_obj = {
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))


if __name__ == '__main__':
    main()
