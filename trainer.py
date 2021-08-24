import time
import torch
import os
import math
import copy
import ipdb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import gc


def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, att_module, bws, bwt, gamma, beta, criterion, criterion_mec, optimizer, optimizer_att, itern, epoch, new_epoch_flag, src_weight, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_source = AverageMeter()
    losses = AverageMeter()
    
    # switch to training mode
    model.train()
    att_module.train()
    bws.train()
    bwt.train()

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1 # penalty hyperparameter
    weight = lam * args.scale if args.src_cls else args.scale
        
    if new_epoch_flag:
        print('The penalty weight is %.4f.' % weight)
        adjust_learning_rate(optimizer, epoch, args, lrplan='dao', alpha=10) # adjust learning rate of base model
        adjust_learning_rate(optimizer_att, epoch, args, lrplan=args.lrplan, alpha=10) # adjust learning rate of cluster centroid learner

    end = time.time()
    
    loss = 0
    
    if args.src_cls: 
        try:
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
        except StopIteration:
            train_loader_source_batch = enumerate(train_loader_source)
            (input_source, target_source, index) = train_loader_source_batch.__next__()[1]
    
        input_source_var = Variable(input_source)
        target_source = target_source.cuda(non_blocking=True)
        target_source_var = Variable(target_source)
        f_s, ca_s = model(input_source_var) # base model foward (source batch)
        loss += SrcClassifyLoss(args, ca_s, target_source, index, src_weight, lam, soft_label=args.src_soft_label) # source loss for structural regularization
        prec1 = accuracy(ca_s, target_source, topk=(1,))[0]
        top1_source.update(prec1[0], input_source.size(0))
        
    try:
        data = train_loader_target_batch.__next__()[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        data = train_loader_target_batch.__next__()[1]
    
    input_target = data[0]
    target_target = data[-2]
    target_target = target_target.cuda(non_blocking=True)    
    input_target_var = Variable(input_target)
    target_target_var = Variable(target_target)
    f_t, ca_t = model(input_target_var) # base model foward (target batch)
    loss += weight * TarClusterLoss(args, epoch, ca_t, target_target) # target loss for discriminative clustering
    
    # consistency loss 
    if args.aug_tar_agree:
        input_target_dup = data[1]
        input_target_dup_var = Variable(input_target_dup)
        ca_t_dup = model(input_target_dup_var)[-1]
        loss += weight * criterion_mec(ca_t, ca_t_dup)
    if args.gray_tar_agree:
        input_target_gray = data[-3]
        input_target_gray_var = Variable(input_target_gray)
        ca_t_gray = model(input_target_gray_var)[-1]
        loss += weight * criterion_mec(ca_t, ca_t_gray)
    
    if args.learn_embed:
        if args.src_cls:
            mu_k = att_module(torch.cat((bwt(f_t), bws(f_s)), dim=0) * gamma + beta) # cluster centroid learner forward
            prob_pred = (1 + (f_s.unsqueeze(1) - mu_k.unsqueeze(0)).pow(2).sum(2)).pow(-1) 
            loss += SrcClassifyLoss(args, prob_pred, target_source, index, src_weight, lam, softmax=args.embed_softmax, soft_label=args.src_soft_label) # source loss for structural regularization
        else:
            mu_k = att_module(bwt(f_t) * gamma + beta)
        prob_pred = (1 + (f_t.unsqueeze(1) - mu_k.unsqueeze(0)).pow(2).sum(2)).pow(-1)
        loss += weight * TarClusterLoss(args, epoch, prob_pred, target_target, softmax=args.embed_softmax) # target loss for generative clustering
            
    losses.update(loss.data.item(), input_target.size(0))
    
    # loss backward and parameter update
    model.zero_grad()
    optimizer_att.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_att.step()
    model.zero_grad()
    optimizer_att.zero_grad()
    
    del f_s, ca_s, f_t, ca_t, mu_k, prob_pred
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    batch_time.update(time.time() - end)
    if itern % args.print_freq == 0:
        print('Train - epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'S@1 {s_top1.val:.3f} ({s_top1.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               epoch, args.epochs, batch_time=batch_time,
               data_time=data_time, s_top1=top1_source, loss=losses))
        
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\nTrain - epoch: %d, top1_s acc: %.3f, loss: %.4f" % (epoch, top1_source.avg, losses.avg))
        log.close()
    
    return train_loader_source_batch, train_loader_target_batch


def TarClusterLoss(args, epoch, output, target, softmax=True): # compute clustering loss on target batch
    """
    Arguments:
        args: options.
        epoch: an integer, current epoch.
        output: a float tensor, output score vector.
        target: a long tensor, pseudo label.
        softmax: can optionally use softmax to normalize the output. Default: True.
    Returns:
        a float tensor with shape [1], computed loss.
    """
    if softmax:
        max_score = output.data.max(1, keepdim=True)[0]
        prob_p = F.softmax(output - max_score, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)

    prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
    prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    if epoch == 0 and not args.resume:
        prob_q = prob_q1
    else:
        prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
        prob_q2 /= prob_q2.sum(1, keepdim=True)
        prob_q = (1 - args.beta) * prob_q1 + args.beta * prob_q2
    
    loss = - (prob_q * (prob_p + args.eps).log()).sum(1).mean()
    
    return loss
    
def SrcClassifyLoss(args, output, target, index, src_weight, lam, softmax=True, soft_label=False): # compute classification loss on source batch
    """
    Arguments:
        args: options.
        output: a float tensor, output score vector.
        target: a long tensor, groud truth.
        index: a long tensor, instance index.
        src_weight: a float tensor, weight of source instance.
        lam: a float, change from 0 to 1 with the training.
        softmax: can optionally use softmax to normalize the output. Default: True.
        soft_label: can optionally use a version of soft label. Default: False.
    Returns:
        a float tensor with shape [1], computed loss.
    """
    if softmax:
        max_score = output.data.max(1, keepdim=True)[0]
        prob_p = F.softmax(output - max_score, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    if soft_label:
        prob_q = (1 - prob_p) * prob_q + prob_p * prob_p
    
    if args.src_mix_weight:
        src_weight = lam * src_weight[index] + (1 - lam) * torch.ones(output.size(0)).cuda()
    else:
        src_weight = src_weight[index]
    
    loss = - (src_weight * (prob_q * (prob_p + args.eps).log()).sum(1)).mean()
    
    return loss


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = {'cls': AverageMeter()}
    
    # switch to evaluate mode
    model.eval()
    
    total_vector = torch.FloatTensor(1, args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(1, args.num_classes).fill_(0)
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)[-1]
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        total_vector[0], correct_vector[0] = accuracy_for_each_class(output, target, total_vector[0], correct_vector[0])
        losses.update(loss.item(), input.size(0))
        top1['cls'].update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test on target test set - [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Prec@1 {cls.val:.3f} ({cls.avg:.3f})'
                  .format(epoch, i, len(val_loader), batch_time=batch_time, 
                  cls_loss=losses, cls=top1['cls']))
    
    print(' * Classifier: prec@1 {cls.avg:.3f}'.format(cls=top1['cls']))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n             Test on target test set - epoch: %d, cls_loss: %.4f, cls acc: %.3f" % (epoch, losses.avg, top1['cls'].avg))
    
    if args.src.find('visda') != -1:
        acc_for_each_class = 100.0 * correct_vector / total_vector[0]
        log.write("\nAcc for each class: ")
        for i in range(args.num_classes):
            log.write("class %d: %.3f, " % (i + 1, acc_for_each_class[0, i]))
        log.write("\nAvg. over all classes: %.3f" % (acc_for_each_class.mean(1)[0]))
        log.close()
        return {'cls': acc_for_each_class.mean(1)[0]}
    else:
        log.close()
        top1 = {k: v.avg for k, v in top1.items()}
        return top1

    
def validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = {'cls': AverageMeter()}
    
    # switch to evaluate mode
    model.eval()

    # compute source class centroids
    source_features = torch.cuda.FloatTensor(len(val_loader_source.dataset.imgs), model.module.feature_dim).fill_(0)
    source_targets = torch.cuda.LongTensor(len(val_loader_source.dataset.imgs)).fill_(0)
    c_src = torch.cuda.FloatTensor(args.num_classes, model.module.feature_dim).fill_(0)
    count_s = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    
    if compute_cen and args.src_cls: 
        for i, (input, target, index) in enumerate(val_loader_source):  # iterarion in the source dataset
            input_var = Variable(input)
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                feature = model(input_var)[0]
            source_features[index.cuda(), :] = feature.data.clone()
            source_targets[index.cuda()] = target.clone()
            for j in range(input.size(0)):
                if (args.cluster_method == 'spherical_kmeans'):
                    c_src[target[j]] += (feature[j] / (feature[j].norm(2) + args.eps))
                else:
                    c_src[target[j]] += feature[j]
                    count_s[target[j]] += 1
        c_src /= count_s if args.cluster_method == 'kmeans' else 1

    end = time.time()
    
    target_features = torch.cuda.FloatTensor(len(val_loader_target.dataset.imgs), model.module.feature_dim).fill_(0)
    target_targets = torch.cuda.LongTensor(len(val_loader_target.dataset.imgs)).fill_(0)
    
    # compute target cluster centroids
    c_tar = torch.cuda.FloatTensor(args.num_classes, model.module.feature_dim).fill_(0)
    count_t = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    
    total_vector = torch.FloatTensor(1, args.num_classes).fill_(0) # for computing per-class accuracy
    correct_vector = torch.FloatTensor(1, args.num_classes).fill_(0)
    
    for i, (input, target, index) in enumerate(val_loader_target):  # iterarion in the target dataset
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            feature, output = model(input_var)
            loss = criterion(output, target_var)
        
        target_features[index.cuda(), :] = feature.data.clone() # index: a tensor 
        target_targets[index.cuda()] = target.clone()
        
        idx_max_score = output.max(1)[1]
        for j in range(input.size(0)):
            if (args.cluster_method == 'spherical_kmeans'):
                c_tar[idx_max_score[j]] += (feature[j] / (feature[j].norm(2) + args.eps))
            else:
                c_tar[idx_max_score[j]] += feature[j]
                count_t[idx_max_score[j]] += 1
        
        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        total_vector[0], correct_vector[0] = accuracy_for_each_class(output, target, total_vector[0], correct_vector[0])
        losses.update(loss.item(), input.size(0))
        top1['cls'].update(prec1[0], input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test on target training set - [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'L {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'T@1 {cls.val:.3f} ({cls.avg:.3f})'.format(
                   epoch, i, len(val_loader_target), batch_time=batch_time,
                   data_time=data_time, cls_loss=losses, cls=top1['cls']))
    
    c_tar /= (count_t + args.eps) if args.cluster_method == 'kmeans' else 1
    
    print(' * Classifier: prec@1 {cls.avg:.3f}'.format(cls=top1['cls']))

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\nTest on target training set - epoch: %d, cls_loss: %.4f, cls acc: %.3f \n" % (epoch, losses.avg, top1['cls'].avg))
    
    if args.src.find('visda') != -1:
        acc_for_each_class = 100.0 * correct_vector / total_vector[0]
        log.write("\nAcc for each class: ")
        for i in range(args.num_classes):
            log.write("class %d: %.3f, " % (i + 1, acc_for_each_class[0,i]))
        log.write("\nAvg. over all classes: %.3f" % (acc_for_each_class.mean(1)[0]))
        log.close()        
        return {'cls': acc_for_each_class.mean(1)[0]}, c_src, c_tar, source_features, source_targets, target_features, target_targets
    else:
        log.close()
        top1 = {k: v.avg for k, v in top1.items()}
        return top1, c_src, c_tar, source_features, source_targets, target_features, target_targets


def source_select(source_features, source_targets, train_loader_source, cen, args):
    # compute source weights
    src_weight = 0.5 * (1 + F.cosine_similarity(source_features, cen[source_targets], dim=-1))
    
    indexes = torch.arange(0, source_features.size(0))
    if args.record_weight_rank:
        for c in range(args.num_classes):
            _, idx = src_weight[source_targets == c].sort(dim=0, descending=True)
            selected_indexes = list(np.array(indexes[source_targets == c][idx][:10000]))
            for i, idx in enumerate(selected_indexes):
                path = train_loader_source.dataset.imgs[idx][0]
                path_move_to = path.replace(args.data_path_source, os.path.join(args.log, args.src + '2' + args.tar + '_rank_src_samples/')).replace('.jpg', '_rank' + str(i + 1) + '_sim' + str(src_weight[idx].item()).replace('.', 'p')[:6] + '.jpg')
                dire_move_to = path_move_to[:path_move_to.rfind('/')]
                if not os.path.exists(dire_move_to):
                    os.makedirs(dire_move_to)
                os.system('cp ' + path + ' ' + path_move_to)
    
    del source_features, source_targets, cen
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return src_weight


def k_means(target_features, target_targets, train_loader_target, epoch, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        dist_xt_ct_temp = target_features.unsqueeze(1) - c_tar.unsqueeze(0)
        dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        idx_sim = (-1 * dist_xt_ct).topk(1, 1, True, True)[1].squeeze(1)
        
        prec1 = accuracy(-1 * dist_xt_ct, target_targets, topk=(1,))[0]
        if prec1 > best_prec:
            best_prec = prec1
        total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct, target_targets, total_vector_dist, correct_vector_dist)
        acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        print('Epoch %d, k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))        
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                log.write("class %d: %.3f" % (i + 1, acc_for_each_class_dist[i]))
            log.write("\nAvg_dist. over all classes: %.3f" % acc_for_each_class_dist.mean())
        log.close()
        
        c_tar.fill_(0)
        count = c_tar[:, 0].unsqueeze(1).clone()
        for k in range(args.num_classes):
            c_tar[k] += (target_features[idx_sim == k].sum(0))
            count[k] += (idx_sim == k).float().sum()
        c_tar /= (count + args.eps)
        
        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])
        
        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    
    del target_features, target_targets, c
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar
    
'''
# for visda due to too many data
def k_means(target_features, target_targets, train_loader_target, epoch, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        interval = 10000
        dist_xt_ct = torch.cuda.FloatTensor(target_features.size(0), args.num_classes).fill_(0)
        for split in range(int(target_features.size(0) / interval) + 1):
            dist_xt_ct_temp = target_features[split*interval:((split + 1)*interval)].unsqueeze(1) - c_tar.unsqueeze(0)
            dist_xt_ct[split*interval:((split + 1)*interval)] = dist_xt_ct_temp.pow(2).sum(2)
            del dist_xt_ct_temp
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        idx_sim = (-1 * dist_xt_ct).topk(1, 1, True, True)[1].squeeze(1)
        
        prec1 = accuracy(-1 * dist_xt_ct, target_targets, topk=(1,))[0]
        if prec1 > best_prec:
            best_prec = prec1
        total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct.data, target_targets, total_vector_dist, correct_vector_dist)
        acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        print('Epoch %d, k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))        
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                log.write("class %d: %.3f" % (i + 1, acc_for_each_class_dist[i]))
            log.write("\nAvg_dist. over all classes: %.3f" % acc_for_each_class_dist.mean())
        log.close()
        
        c_tar.fill_(0)
        count = c_tar[:, 0].unsqueeze(1).clone()
        for k in range(args.num_classes):
            c_tar[k] += (target_features[idx_sim == k].sum(0))
            count[k] += (idx_sim == k).float().sum()
        c_tar /= (count + args.eps)
        
        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])
    
    del target_features, target_targets, c
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar
'''
    
def spherical_k_means(target_features, target_targets, train_loader_target, epoch, c, args, best_prec, change_target=True):
    batch_time = AverageMeter()
    
    c_tar = c.clone()
    end = time.time()
    for itr in range(args.cluster_iter):
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - ((dist_xt_ct_temp.sum(2) / (target_features.norm(2, dim=1, keepdim=True) + args.eps)) / (c_tar.norm(2, dim=1, keepdim=True).t() + args.eps)))
        idx_sim = (-1 * dist_xt_ct).topk(1, 1, True, True)[1].squeeze(1)
        
        prec1 = accuracy(-1 * dist_xt_ct, target_targets, topk=(1,))[0]
        if prec1 > best_prec:
            best_prec = prec1
        total_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        correct_vector_dist = torch.FloatTensor(args.num_classes).fill_(0)
        total_vector_dist, correct_vector_dist = accuracy_for_each_class(-1 * dist_xt_ct, target_targets, total_vector_dist, correct_vector_dist)
        acc_for_each_class_dist = 100.0 * correct_vector_dist / (total_vector_dist + args.eps)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        print('Epoch %d, spherical k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write('\nEpoch %d, spherical k-means clustering %d, average clustering time %.3f, prec@1 %.3f' % (epoch, itr, batch_time.avg, prec1))
        if args.src.find('visda') != -1:
            log.write("\nAcc_dist for each class: ")
            for i in range(args.num_classes):
                log.write("class %d: %.3f" % (i + 1, acc_for_each_class_dist[i]))
            log.write("\nAvg_dist. over all classes: %.3f" % acc_for_each_class_dist.mean())
        log.close()

        c_tar.fill_(0)
        for k in range(args.num_classes):
            c_tar[k] += ((target_features[idx_sim == k] / (target_features[idx_sim == k].norm(2, dim=1, keepdim=True) + args.eps)).sum(0))
        
        if (itr == (args.cluster_iter - 1)) and change_target:
            for i in range(target_targets.size(0)):
                train_loader_target.dataset.tgts[i] = int(idx_sim[i])
        
        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    
    del target_features, target_targets, c
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, lrplan, alpha=10):
    """Adjust the learning rate according to epoch"""
    if lrplan == 'step':
        exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
        lr = args.lr * (args.gamma ** exp)
    elif lrplan == 'dao':
        lr = args.lr / math.pow((1 + alpha * epoch/args.epochs), 0.75)
    elif lrplan == 'exp':
        lr = args.lr * ((1 - epoch/args.epochs) ** 0.9)
    lr_pretrain = lr * 0.1
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        elif param_group['name'] == 'newly-added':
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    pred = output.max(1)[1]
    correct = pred.eq(target).float().cpu()
    for i in range(target.size(0)):
        total_vector[target[i]] += 1
        correct_vector[target[i]] += correct[i]
    
    return total_vector, correct_vector


