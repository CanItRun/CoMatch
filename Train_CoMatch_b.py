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

import numpy as np
from lumo.contrib.nn.functional import batch_cosine_similarity, masked_log_softmax
from lumo.contrib.nn.loss import contrastive_loss2
from lumo.proc.path import cache_dir
import torch
import torch.nn as nn
import torch.nn.functional as F

from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler

import tensorboard_logger
from lumo import Logger, AvgMeter, Meter

from lumo import TrainerExperiment

exp = TrainerExperiment('comatch.origin2')
exp.start()
log = Logger()
log.add_log_dir(exp.test_root)


def contrastive_loss2(query: torch.Tensor, key: torch.Tensor,
                      memory: torch.Tensor = None,
                      norm=False,
                      temperature=0.7,
                      query_neg=False,
                      key_neg=True,
                      qk_graph=None,
                      qm_graph=None,
                      eye_one_in_qk=False,
                      ):
    """
    Examples:
        >>> assert contrastive_loss(a,b,inbatch_neg=False) == contrastive_loss2(a,b,query_neg=False,key_neg=True)

    Args:
        query: [bq, f_dim]
        key: [bq, f_dim]
        memory: [bm, f_dim]
        norm: bool, normalize feature or not
        temperature: float
        query_neg: bool
        key_neg: bool
        qk_graph: [bq, bq], >= 0
        qm_graph: [bq, bm], >= 0

    Returns:
        loss

    """
    if memory is None:
        key_neg = True
        qm_graph = None

    if memory is not None:
        key = torch.cat([key, memory])

    if query_neg:
        key = torch.cat([query, key])

    q_size = query.shape[0]

    if norm:
        logits = torch.cosine_similarity(query.unsqueeze(1), key.unsqueeze(0), dim=2)
    else:
        logits = torch.mm(query, key.t())  # query @ key.t()

    logits /= temperature

    neg_index = torch.ones_like(logits, dtype=torch.bool, device=logits.device)

    _temp_eye_indice = torch.arange(q_size, device=logits.device).unsqueeze(1)

    neg_offset = q_size if query_neg else 0
    if query_neg:
        neg_index.scatter_(1, _temp_eye_indice, 0)

    if key_neg:
        neg_index.scatter_(1, _temp_eye_indice + neg_offset, 0)
    else:
        neg_index[:, neg_offset:neg_offset + q_size] = 0

    pos_index = torch.zeros_like(logits, dtype=torch.float, device=logits.device)
    pos_offset = q_size if query_neg else 0

    # for supervised cs
    if qk_graph is not None:
        pos_index[:, :q_size] = qk_graph.float()
        if query_neg:
            pos_index[:, q_size:q_size * 2] = qk_graph.float()

    # for supervised cs with moco memory bank
    if qm_graph is not None:
        pos_index[:, pos_offset + q_size:] = qm_graph.float()

    if qk_graph is None or eye_one_in_qk:
        pos_index.scatter_(1, _temp_eye_indice + pos_offset, 1)

    logits_mask = (pos_index > 0) | neg_index
    loss = -torch.sum(masked_log_softmax(logits, logits_mask, dim=-1) * pos_index, dim=1)
    loss = (loss / pos_index.sum(1)).mean()
    return loss


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
                    queue_ptr,
                    dlval
                    ):
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
    for it in range(n_iters):
        meter = Meter()
        (ims_x_weak, ims_x_s0, ims_x_s1), ys = next(dl_x)
        (ims_u_weak, ims_u_s0, ims_u_s1), unys = next(dl_u)

        ys = ys.cuda()
        unys = unys.cuda()

        # --------------------------------------
        bt = ims_x_weak.size(0)
        btu = ims_u_weak.size(0)

        imgs = torch.cat([ims_x_weak, ims_x_s0, ims_x_s1,
                          ims_u_weak, ims_u_s0, ims_u_s1], dim=0).cuda()
        logits, features = model(imgs)

        logits_x, _, _ = logits[:bt * 3].chunk(3)
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt * 3:], btu)

        sup_w_query, sup_query, sup_key = features[:bt * 3].chunk(3)
        un_w_query, un_query, un_key = torch.split(features[bt * 3:], btu)

        loss_x = criteria_x(logits_x, ys)

        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            sup_w_query = sup_w_query.detach()
            un_w_query = un_w_query.detach()

            probs = torch.softmax(logits_u_w, dim=1)
            # DA
            prob_list.append(probs.mean(0))
            if len(prob_list) > 32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)

            probs_orig = probs.clone()

            if epoch > 0 or it > args.queue_batch:  # memory-smoothing
                A = torch.exp(torch.mm(un_w_query, queue_feats.t()) / args.temperature)
                A = A / A.sum(1, keepdim=True)
                probs = args.alpha * probs + (1 - args.alpha) * torch.mm(A, queue_probs)

            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()

            feats_w = torch.cat([un_w_query, sup_w_query], dim=0)
            onehot = torch.zeros(bt, args.n_classes).cuda().scatter(1, ys.view(-1, 1), 1)
            probs_w = torch.cat([probs_orig, onehot], dim=0)

            # update memory bank
            n = bt + btu
            queue_feats[queue_ptr:queue_ptr + n, :] = feats_w
            queue_probs[queue_ptr:queue_ptr + n, :] = probs_w
            queue_ptr = (queue_ptr + n) % args.queue_size

        # embedding similarity
        sim = torch.exp(torch.mm(un_query, un_key.t()) / args.temperature)
        sim_probs = sim / sim.sum(1, keepdim=True)

        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= args.contrast_th).float()

        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)

        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0, dim=1) * probs), dim=1) * mask
        loss_u = loss_u.mean()

        def comatch_loss():
            exp.add_tag('Lcs')
            return contrastive_loss2(un_query, un_key, temperature=args.temperature, qk_graph=Q)

        # contrastive loss
        loss_contrast = comatch_loss()

        # loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        # loss_contrast = loss_contrast.mean()

        # unsupervised classification loss
        loss_u = - torch.sum((F.log_softmax(logits_u_s0, dim=1) * probs), dim=1) * mask
        loss_u = loss_u.mean()

        def choice_(tensor, size=128):
            return tensor[torch.randperm(len(tensor))[:size]]

        # graph cs2
        def graph_cs2():
            # ./log/l.0.2110191754.log
            exp.add_tag('sup_gcs')
            memory = queue_feats
            # memory = torch.cat([un_query, un_key, queue_feats])
            # pos_memory = memory[torch.randperm(len(memory))[:128]]
            pos_memory = torch.cat([choice_(un_query),
                                    choice_(un_key),
                                    choice_(memory)])
            pos_memory = choice_(pos_memory, 256)
            # neg_memory = memory[torch.randperm(len(memory))[:128]]
            # memory = torch.cat([un_query, un_key, memory])

            anchor = batch_cosine_similarity(sup_query, pos_memory)
            positive = batch_cosine_similarity(sup_key, pos_memory)
            # negative = batch_cosine_similarity(sup_key, neg_memory)
            gqk = ys.unsqueeze(0) == ys.unsqueeze(1)
            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True,
                                     temperature=0.2,
                                     qk_graph=gqk)
            return loss

        def graph_cs3():
            # ./log/l.0.2110191754.log
            exp.add_tag('un_gcs')
            # memory = queue_feats
            memory = queue_feats
            pos_memory = torch.cat([choice_(un_query),
                                    choice_(un_key),
                                    choice_(memory)])
            pos_memory = choice_(pos_memory, 256)
            # neg_memory = memory[torch.randperm(len(memory))[:512]]

            anchor = batch_cosine_similarity(un_query, pos_memory)
            positive = batch_cosine_similarity(un_key, pos_memory)
            # negative = batch_cosine_similarity(un_key, neg_memory)

            # gqk = ys.unsqueeze(0) == ys.unsqueeze(1)
            loss = contrastive_loss2(anchor, positive,
                                     memory=None,
                                     norm=True,
                                     temperature=0.2,
                                     qk_graph=Q)
            return loss

        Lgcs1 = graph_cs2()
        Lgcs2 = graph_cs3()

        def strategy0():
            exp.add_tag('loss0')
            loss = loss_x + args.lam_u * loss_u + loss_contrast
            return loss

        def strategy1():
            exp.add_tag('loss1')
            loss = loss_x + args.lam_u * loss_u + Lgcs1 + Lgcs2 + loss_contrast
            return loss

        def strategy2():
            exp.add_tag('loss2')
            if epoch < 3:
                loss = loss_x + args.lam_u * loss_u + Lgcs1 + Lgcs2 + loss_contrast
            else:
                loss = loss_x + args.lam_u * loss_u + loss_contrast
            return loss

        def strategy3():
            exp.add_tag('loss3')

            if epoch < 3:
                loss = loss_x + args.lam_u * loss_u + Lgcs1 + Lgcs2 + loss_contrast
            else:
                loss = loss_x + args.lam_u * loss_u + Lgcs1 * 0.1 + Lgcs2 * 0.1 + loss_contrast
            return loss

        if args.s == 0:
            loss = strategy0()
        elif args.s == 1:
            loss = strategy1()
        elif args.s == 2:
            loss = strategy2()
        elif args.s == 3:
            loss = strategy3()

        with torch.no_grad():
            meter.mean.Lall = loss
            meter.mean.Lx = loss_x
            meter.mean.Lu = loss_u
            meter.mean.Lcs = loss_contrast
            meter.mean.Lgcs1 = Lgcs1
            meter.mean.Lgcs2 = Lgcs2
            meter.mean.Pm = pos_mask.float().mean()
            meter.mean.Pm = pos_mask.float().mean()
            meter.mean.Ax = (logits_x.argmax(dim=-1) == ys).float().mean()
            meter.mean.um = mask.float().mean()
            meter.mean.Au = (logits_u_w.argmax(dim=-1) == unys).float().mean()
            if mask.bool().any():
                meter.mean.Aum = (logits_u_w.argmax(dim=-1) == unys)[mask.bool()].float().mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)

        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_contrast_meter.update(loss_contrast.item())
        mask_meter.update(mask.mean().item())
        pos_meter.update(pos_mask.sum(1).float().mean().item())

        avg.update(meter)
        corr_u_lb = (lbs_u_guess == unys).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())
        log.inline(avg)
        if (it + 1) % 64 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. loss_c: {:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.3f}. num_pos: {:.1f}. LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg,
                loss_x_meter.avg, loss_contrast_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg,
                mask_meter.avg, pos_meter.avg, lr_log, t))

            epoch_start = time.time()

        if (it + 1) % 150 == 0:
            log.newline()
            evaluate(model, ema_model, dlval)
            model.train()

    return loss_x_meter.avg, loss_u_meter.avg, loss_contrast_meter.avg, mask_meter.avg, pos_meter.avg, n_correct_u_lbs_meter.avg / n_strong_aug_meter.avg, queue_feats, queue_probs, queue_ptr, prob_list


def evaluate(model, ema_model, dataloader):
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    avg = AvgMeter()
    with torch.no_grad():
        for ims, lbs in dataloader:
            meter = Meter()
            ims = ims.cuda()
            lbs = lbs.cuda()

            logits, _ = model(ims)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            meter.sum.A1 = top1
            meter.sum.A5 = top5

            if ema_model is not None:
                logits, _ = ema_model(ims)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                ema_top1_meter.update(top1.item())
                meter.sum.Ae1 = top1
                meter.sum.Ae5 = top5
            avg.update(meter)
        log.info(avg)

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
    parser.add_argument('--device', type=int, default=0,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--s', type=int, default=0,
                        help='策略')
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
    parser.add_argument('--seed', type=int, default=1,
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

    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

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

        loss_x, loss_u, loss_c, mask_mean, num_pos, guess_label_acc, queue_feats, queue_probs, queue_ptr, prob_list = \
            train_one_epoch(epoch, **train_args, queue_feats=queue_feats, queue_probs=queue_probs, queue_ptr=queue_ptr,
                            dlval=dlval)

        top1, ema_top1 = evaluate(model, ema_model, dlval)

        tb_logger.log_value('loss_x', loss_x, epoch)
        tb_logger.log_value('loss_u', loss_u, epoch)
        tb_logger.log_value('loss_c', loss_c, epoch)
        tb_logger.log_value('guess_label_acc', guess_label_acc, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('test_ema_acc', ema_top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        tb_logger.log_value('num_pos', num_pos, epoch)

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
