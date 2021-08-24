##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# The file ./opts.py stores the options
# The file ./trainer.py stores the training and test strategies
# The ./main.py should be simple
#
##############################################################################
import os
import json
import shutil
import torch
import torch.optim
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
from models.resnet import resnet # base model for the model construction
from models.blocks import SetTransformer # set transformer for the model construction
from utils.whitening import whitening_scale_shift # batch whitening for the model construction
from trainer import train  # for the training process
from trainer import validate, validate_compute_cen # for the validation/test process
from trainer import k_means, spherical_k_means # for k-means clustering and its variants
from trainer import source_select # for source sample selection
from opts import opts  # options for the project
from data.prepare_data import generate_dataloader # prepare data and dataloader
from utils.consensus_loss import MinEntropyConsensusLoss # consistency loss
from torch.autograd import Variable
import time
import ipdb
import gc

best_val_accs = {'cls': 0, 'clust': 0}
best_test_accs = {'cls': 0, 'cond_cls': 0}

def main():
    global args, best_val_accs, best_test_accs
    args = opts()
    
    # define base model    
    model = resnet(args)
    model = nn.DataParallel(model).cuda() # define multiple GPUs
    
    # define cluster centroid learner
    bws = whitening_scale_shift(batch_size=args.batch_size, num_features=model.module.feature_dim, affine=False).cuda()
    bwt = whitening_scale_shift(batch_size=args.batch_size, num_features=model.module.feature_dim, affine=False).cuda()
    gamma = torch.ones(model.module.feature_dim).cuda()
    gamma.requires_grad_(True)
    beta = torch.zeros(model.module.feature_dim).cuda()
    beta.requires_grad_(True)
    
    att_module = SetTransformer(model.module.feature_dim, args.num_classes, model.module.feature_dim, args.only_decoder)
    att_module = nn.DataParallel(att_module).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_mec = MinEntropyConsensusLoss(div=args.div).cuda()

    np.random.seed(1) # may fix data
    random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(1)
    
    # apply different learning rate to different layer
    optimizer = torch.optim.SGD([
        {'params': model.module.conv1.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.bn1.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.layer1.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.layer2.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.layer3.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.layer4.parameters(), 'name': 'pre-trained' if args.pretrained else 'newly-added'},
        {'params': model.module.fc.parameters(), 'name': 'newly-added'},
    ],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, 
                                nesterov=args.nesterov,)
    
    optimizer_att = torch.optim.Adam([
                                      {'params': gamma, 'name': 'newly-added'}, 
                                      {'params': beta, 'name': 'newly-added'}, 
                                      {'params': att_module.parameters(), 'name': 'newly-added'}, 
                                      ],
                                      lr=args.lr, weight_decay=1e-5,)
    
    # resume                                    
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_accs = checkpoint['best_val_accs']
            best_test_accs = checkpoint['best_test_accs']
            pretrained_state_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_state_dict[0])
            bws.load_state_dict(pretrained_state_dict[1])
            bwt.load_state_dict(pretrained_state_dict[2])
            att_module.load_state_dict(pretrained_state_dict[-1])
            gamma.data = checkpoint['gamma'][0].data
            beta.data = checkpoint['beta'][0].data
            print("==> loaded checkpoint '{}'(epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from does not exist', args.resume)
    
    # make log directory
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
        
    # record options
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()
    
    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()

    cudnn.benchmark = True
    
    # process data and prepare dataloaders
    train_loader_source, train_loader_target, val_loader_target, val_loader_target_t, val_loader_source = generate_dataloader(args)
    train_loader_target.dataset.tgts = list(np.array(torch.LongTensor(train_loader_target.dataset.tgts).fill_(-1))) # avoid using ground truth labels of target

    print('begin training')
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    num_itern_total = args.epochs * batch_number
    
    test_flag = False # if test, test_flag=True
    new_epoch_flag = False # if new epoch, new_epoch_flag=True
    
    src_weight = torch.cuda.FloatTensor(len(train_loader_source.dataset.tgts)).fill_(1) # initialize source weights
    
    epoch = args.start_epoch
    count_itern_each_epoch = 0
    for itern in range(args.start_epoch * batch_number, num_itern_total + 1):
        # evaluate on val/test data
        if (itern == 0) or (count_itern_each_epoch == batch_number):
            val_accs, c_s, c_t, source_features, source_targets, target_features, target_targets = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=True if itern == 0 or args.initial_cluster != 0 or args.src_soft_select else False)
            test_flag = True

            # k-means clustering or its variants
            cen = None
            if (itern == 0 and args.src_cen_first) or args.initial_cluster == 2:
                cen = c_s
            elif (itern == 0 and not args.src_cen_first) or args.initial_cluster == 1:
                cen = c_t
            if cen is not None:
                if args.cluster_method == 'kmeans':
                    clust_acc, c_t = k_means(target_features, target_targets, train_loader_target, epoch, cen, args, best_val_accs['clust'])
                elif args.cluster_method == 'spherical_kmeans':
                    clust_acc, c_t = spherical_k_means(target_features, target_targets, train_loader_target, epoch, cen, args, best_val_accs['clust'])
                val_accs['clust'] = clust_acc
                
            test_accs = validate(val_loader_target_t, model, criterion, epoch, args) if args.tar != args.tar_t else val_accs
                        
            if itern != 0:
                if args.src_soft_select:
                    src_weight = source_select(source_features, source_targets, train_loader_source, c_t, args) # select source samples
                count_itern_each_epoch = 0
                epoch += 1
            train_loader_target_batch = enumerate(train_loader_target)
            train_loader_source_batch = enumerate(train_loader_source)
            
            new_epoch_flag = True
            
            del source_features, source_targets, target_features, target_targets, c_s, c_t, cen
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        elif args.src.find('visda') != -1 and (itern + 1) % int(num_itern_total / 200) == 0:
            val_accs = validate_compute_cen(val_loader_target, val_loader_source, model, criterion, epoch, args, compute_cen=False)[0]
            test_accs = validate(val_loader_target_t, model, criterion, epoch, args) if args.tar != args.tar_t else val_accs
            test_flag = True
            
        if itern != 0 and test_flag:
            # record the best accuracy
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            for k in val_accs.keys():
                if val_accs[k] > best_val_accs[k]:
                    best_test_accs['cond_cls'] = 0 if k == 'cls' else best_test_accs['cond_cls']
                    best_val_accs[k] = val_accs[k]
                    log.write('\n                                                                        best val ' + k + ' acc: %.3f' % best_val_accs[k])
            for k in test_accs.keys():
                best_test_accs['cond_cls'] = test_accs[k] if k == 'cls' and val_accs[k] == best_val_accs[k] and test_accs[k] > best_test_accs['cond_cls'] else best_test_accs['cond_cls']
                if k in best_test_accs.keys() and test_accs[k] > best_test_accs[k]:
                    best_test_accs[k] = test_accs[k]
                    log.write('\n                                                                        best test ' + k + ' acc: %.3f' % best_test_accs[k])
            log.close()
            
            # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': [model.state_dict(), bws.state_dict(), bwt.state_dict(), att_module.state_dict()],
                'gamma': [gamma],
                'beta': [beta],
                'best_val_accs': best_val_accs,
                'best_test_accs': best_test_accs,
            }, test_accs['cls'] == best_test_accs['cls'], args)
                    
        test_flag = False
        
        if epoch == args.stop_epoch:
            break
                
        # train for one iteration
        train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model, att_module, bws, bwt, gamma, beta, criterion, criterion_mec, optimizer, optimizer_att, itern, epoch, new_epoch_flag, src_weight, args)
        
        new_epoch_flag = False        
        count_itern_each_epoch += 1
    
    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n***   best val cls acc: %.3f   ***' % best_val_accs['cls'])
    for k in best_test_accs.keys():
        log.write('\n***   best test ' + k + ' acc: %.3f   ***' % best_test_accs[k])
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()
    

def count_epoch_on_large_dataset(train_loader_target, train_loader_source, args):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    if args.src_cls:
        batch_number_s = len(train_loader_source)
        if batch_number_s > batch_number_t:
            batch_number = batch_number_s
    
    return batch_number
    

def save_checkpoint(state, is_best, args):
    filename = 'final_checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()


