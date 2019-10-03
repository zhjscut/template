import torch
from torch.autograd import Variable
import time
import os
import math
from tensorboardX import SummaryWriter
import copy 
import numpy as np

from utils import mixup_data, mixup_criterion, AverageMeter
from utils import My_criterion, My_criterion_Em
from utils import check_params, check_grad, analyze_output


batch_time_train = AverageMeter()
data_time_train = AverageMeter()
losses_C_train = AverageMeter()
top1_source_train = AverageMeter()
top5_source_train = AverageMeter()
losses_G_train = AverageMeter()
losses_total_train = AverageMeter()

batch_time_val = AverageMeter()
data_time_val = AverageMeter()
losses_source_val = AverageMeter()
top1_source_val = AverageMeter()
top5_source_val = AverageMeter()
losses_target_val = AverageMeter()
top1_target_val = AverageMeter()
top5_target_val = AverageMeter()
losses_total_val = AverageMeter()

# def train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model, criterion_y, criterion_d, optimizer_C, optimizer_G, epoch, args):
def train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, feature_extractor, class_classifier, domain_classifier, criterion_y, criterion_d, optimizer, epoch, args):
    """
    Train for one epoch. Only a batch is used in a epoch, not all the batches.
    Parameters
    ----------
    train_loader_source: torch.utils.data.DataLoader
        Used to reset train_loader_source_batches if the enumerate reach the end of iteration
    train_loader_source_batches: enumerate 
        An object whose each element contain one batch of source data
    train_loader_target: torch.utils.data.DataLoader
        Used to reset train_loader_target_batches if the enumerate reach the end of iteration
    train_loader_target_batches: enumerate
        An object whose each element contain one batch of target data
    model: pytorch model
        The model in training pipeline
    criterion_y: A certain class of loss in torch.nn
        The criterion of the label predicter model
    criterion_d: A certain class of loss in torch.nn
        The criterion of the domain classifier model        
    optimizer_C: An optimizer in a certain update principle in torch.optim
        The optimizer for classifier of the model 
    optimizer_G: An optimizer in a certain update principle in torch.optim
        The optimizer for feature extracter of the model 
    args: Namespace
        Arguments that main.py receive
    epoch: int
        The current epoch
    Return
    ------
    pred_acc1_source: float
        The top1 accuracy in this minibatch
    loss_total_train: float
        The loss in this minibatch
    """
#     model.train()
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()
    
    adjust_learning_rate(optimizer, epoch, args)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])    
    end = time.time()

    # prepare the data for the model forward and backward
    # note that DANN is used on the condition that the label of target dataset is not available
    new_epoch_flag = False
    try:
        _, (inputs_source, labels_source) = train_loader_source_batches.__next__()
        
    except StopIteration:
        if args.epoch_count_dataset == 'source':
            epoch = epoch + 1
            new_epoch_flag = True
        train_loader_source_batches = enumerate(train_loader_source)
        _, (inputs_source, labels_source) = train_loader_source_batches.__next__()
        
    try:
        _, (inputs_target, _) = train_loader_target_batches.__next__()
    except StopIteration:
        if args.epoch_count_dataset == 'target':
            epoch = epoch + 1
            new_epoch_flag = True
        train_loader_target_batches = enumerate(train_loader_target)
        _, (inputs_target, _) = train_loader_target_batches.__next__()
        
    
    if torch.cuda.is_available():
        inputs_source = inputs_source.cuda(async=True)
        labels_source = labels_source.cuda(async=True)
    inputs_source_var, labels_source_var = Variable(inputs_source), Variable(labels_source)

    if torch.cuda.is_available():
        inputs_target = inputs_target.cuda(async=True)
    inputs_target_var = Variable(inputs_target)
    data_time_train.update(time.time() - end)

    # compute the output of source domain and target domain
    feature_source = feature_extractor(inputs_source)
    feature_target = feature_extractor(inputs_target)

    # compute the class loss of feature_source
    outputs_source = class_classifier(feature_source)
    outputs_target = class_classifier(feature_target)
    loss_C = criterion_y(outputs_source, labels_source)

    # prepare domain labels
    if torch.cuda.is_available():
        source_labels = Variable(torch.zeros((inputs_source.size()[0])).type(torch.LongTensor).cuda())
        target_labels = Variable(torch.ones((inputs_target.size()[0])).type(torch.LongTensor).cuda())
    else:
        source_labels = Variable(torch.zeros((inputs_source.size()[0])).type(torch.LongTensor))
        target_labels = Variable(torch.ones((inputs_target.size()[0])).type(torch.LongTensor))
        
    # compute the domain loss of feature_source and target_feature
    p = float(epoch) / args.epochs
    constant = 2. / (1. + np.exp(-args.gamma * p)) - 1        
    preds_source = domain_classifier(feature_source, constant)
    preds_target = domain_classifier(feature_target, constant)
    domain_loss_source = criterion_d(preds_source, source_labels)
    domain_loss_target = criterion_d(preds_target, target_labels)
    loss_G = domain_loss_target + domain_loss_source
    
    loss = loss_C + loss_G
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    
    grad_mean_extractor = check_grad(feature_extractor)
    grad_mean_class = check_grad(class_classifier)
    grad_mean_domain = check_grad(domain_classifier)

#     if new_epoch_flag or epoch == 0:
    if new_epoch_flag or epoch == 0:
        with torch.no_grad():
    #         outputs_tmp = copy.deepcopy(outputs_Cst_target)
            writer = SummaryWriter(log_dir=args.log)       
            o_minimum, o_maximum, o_medium = analyze_output(outputs_target)
            writer.add_scalars('data/output_analysis', {'o_minimum': o_minimum,
                                                        'o_maximum': o_maximum,
                                                        'o_medium': o_medium
                                                       }, epoch)         
            writer.add_scalars('data/scalar_group', {'grad_mean_extractor': grad_mean_extractor,
                                                     'grad_mean_class': grad_mean_extractor,
                                                    'grad_mean_domain': grad_mean_domain}, epoch)
            writer.add_scalars('data/insight', {'loss_C': loss_C,
                                                'domain_loss': loss_G,
                                               'loss': loss
                                              }, epoch)        
            writer.close() 
        
    # measure accuracy and record loss    
    pred_acc1_source, pred_acc5_source = accuracy(outputs_source, labels_source_var, topk=(1, 5))
    losses_C_train.update(loss_C.data)
    losses_G_train.update(loss_G.data)
    loss_total_train = loss_C + loss_G
    losses_total_train.update(loss_total_train.data)
    top1_source_train.update(pred_acc1_source)
    top5_source_train.update(pred_acc5_source)
    batch_time_train.update(time.time() - end)
    
    if epoch % args.print_freq == 0:
        print('Tr epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'S@1 {top1_source.val:.3f} ({top1_source.avg:.3f})\t'
              'S@5 {top5_source.val:.3f} ({top5_source.avg:.3f})\t'
              'loss_C {loss_C_train.val:.4f} ({loss_C_train.avg:.4f})\t'
              'loss_G {loss_G_train.val:.4f} ({loss_G_train.avg:.4f})\t'.format(
              epoch, args.epochs, batch_time=batch_time_train, data_time=data_time_train,
              top1_source=top1_source_train, top5_source=top5_source_train, 
              loss_C_train=losses_C_train, loss_G_train=losses_G_train)
              )
    if epoch % args.record_freq == (args.record_freq-1):
        if not os.path.isdir(args.log):
            os.mkdir(args.log)
        with open(os.path.join(args.log, 'log.txt'), 'a+') as fp:
            fp.write('\n')
            fp.write('Tr:epoch: %d, loss_total: %4f,'
                     'top1_source acc: %3f, top5_source acc: %3f, loss_C: %4f, loss_G: %4f'
                     % (epoch, losses_total_train.avg, top1_source_train.avg, top5_source_train.avg, losses_C_train.avg, losses_G_train.avg))    

#     return pred_acc1_source, loss_total_train
#     return pred_acc1_source, loss_C, loss_G
    return train_loader_source_batches, train_loader_target_batches, epoch, pred_acc1_source, loss_C, loss_G, new_epoch_flag


# def validate(val_loader_source, val_loader_target, model, criterion_y, criterion_d, epoch, args):
def validate(val_loader_source, val_loader_target, feature_extractor, class_classifier, domain_classifier, criterion_y, criterion_d, epoch, args):
    """
    Validate on the whole validation set
    Parameters
    ----------
    val_loader_source: torch.utils.data.DataLoader
        The dataloader that contains batches of source data. And this parameter can be None, to not validate in source dataset
    val_loader_target: torch.utils.data.DataLoader 
        The dataloader that contains batches of target data
    model: pytorch model
        The model in training pipeline
    criterion_y: A certain class of loss in torch.nn
        The criterion of the label predicter model
    criterion_d: A certain class of loss in torch.nn
        The criterion of the domain classifier model  
    args: Namespace
        Arguments that main.py receive
    epoch: int
        The current epoch
    Return
    ------
    top1_target_val.avg: float
        The average top1 accuracy in validation set
    losses_total_val.avg: float
        The average loss in validation set
    """

#     model.eval()
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
#     end = time.time()
    # because the batch size of the last batch of a dataset is usually less than args.batch_size, so the real_labels and fake_labels can not simply set their size equals to args.batch_size
    with torch.no_grad():    
        # in spite of whether val_loader_source is None, the validation on target dataset has to be done
        end = time.time()
        # temporary AverageMeter compute the measures in a certain validating operation
        top1_target_val_tmp = AverageMeter()
        top5_target_val_tmp = AverageMeter()
        for i, data in enumerate(val_loader_target):
            inputs_target, labels_target = data
            if torch.cuda.is_available():
                inputs_target = inputs_target.cuda(async=True)
                labels_target = labels_target.cuda(async=True)
            inputs_target_var, labels_target_var = Variable(inputs_target), Variable(labels_target)
#             outputs_Cst_target = model(inputs_target_var)
            outputs_target = class_classifier(feature_extractor(inputs_target_var))
            
#             outputs_Cs_target, outputs_Ct_target = outputs_Cst_target[:, 0:args.num_classes_s], outputs_Cst_target[:, args.num_classes_s:]
            pred_acc1_target, pred_acc5_target = accuracy(outputs_target, labels_target_var, topk=(1, 5))
            batch_size_target = inputs_target.size(0)
            top1_target_val_tmp.update(pred_acc1_target, batch_size_target)
            top5_target_val_tmp.update(pred_acc5_target, batch_size_target)

    total_time_val = time.time() - end
    top1_target_val.update(top1_target_val_tmp.avg) # the average measure in the whole validation set is the current measure in the AverageMeter
    top5_target_val.update(top5_target_val_tmp.avg)
    # losses_total_val.update(losses_total_val_tmp.avg)
    if epoch % args.print_freq == 0:
        print('Te epoch [{0}/{1}]\t'
            'Time {total_time:.3f}\t'
            'T@1 {top1_target.val:.3f} ({top1_target.avg:.3f})\t'
            'T@5 {top5_target.val:.3f} ({top5_target.avg:.3f})\t'.format(
            epoch, args.epochs, total_time=total_time_val, 
            top1_target=top1_target_val, top5_target=top5_target_val)
            )

#     if val_loader_source:
#         print(' * Source Dataset Prec@1 {top1_source.avg:.3f} Prec@5 {top5_source.avg:.3f}'
#               .format(top1_source=top1_source_val, top5_source=top5_source_val))
    print(' * Target Dataset Prec@1 {top1_target.avg:.3f} Prec@5 {top5_target.avg:.3f}'
          .format(top1_target=top1_target_val_tmp, top5_target=top5_target_val_tmp))
    if not os.path.isdir(args.log):
        os.mkdir(args.log)
    with open(os.path.join(args.log, 'log.txt'), 'a+') as fp:
#         if val_loader_source:
#             fp.write('\n')
#             fp.write('    Test Source: epoch %d, Top1 acc: %3f, Top5 acc: %3f' 
#                     % (epoch, top1_source_val.avg, top5_source_val.avg))
        fp.write('\n')
        fp.write('    Test Target: epoch %d, Top1 acc: %3f, Top5 acc: %3f'
                    % (epoch, top1_target_val_tmp.avg, top5_target_val_tmp.avg))


    return top1_target_val_tmp.avg, 10086


def adjust_learning_rate(optimizer, epoch, args, ratio=0.1):
    """
    Adjust the learning rate according to the epoch
    Parameters
    ----------
    optimzer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model 
    epoch: int
        The current epoch
    args: Namespace
        Arguments that main.py receive
    ratio: float
        ratio of lr in different optimizers, such as pretrained layer(0.1) and scratch layer(1)
    Return
    ------
    The function has no return
    """    
    p = float(epoch) / args.epochs
    lr = args.lr / ((1 + 10 * p) ** 0.75)
    lr_pretrain = args.lr * ratio / ((1 + 10 * p) ** 0.75) # 0.001 / pow((1 + 10 * epoch / epoch_total), 0.75)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        else:
            param_group['lr'] = lr  
            
def adjust_learning_rate_v2(optimizer, epoch, args, mode='rel', value=0.1, namelist=[]):
    """
    Adjust the learning rate according to the epoch
    Parameters
    ----------
    optimzer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model 
    epoch: int
        The current epoch
    args: Namespace
        Arguments that main.py receive
    total_epochs: int
        The total epoch number
    mode: str
        Mode of setting lr, 'rel' (relative) or 'abs' (absolute)
    value: float
        In 'rel' mode, parameters multiply this variable;
        In 'abs' mode, parameters is set to this variable
    namelist: list
        If namelist is not empty, then only adjust the lr of param_groups whose name is in namelist;
        If namelist is empty (default), then adjust the lr of all param_group
    Return
    ------
    The function has no return
    """    
    select_groups = []
    if len(namelist) == 0:
        select_groups = optimizer.param_groups
    else:
        for param_group in optimizer.param_groups:
            if param_group['name'] in namelist:
#                     print(param_group['name'])
                select_groups.append(param_group)

    for param_group in select_groups:
        if mode == 'rel':
            param_group['lr'] = param_group['lr'] * value
        elif mode == 'abs':
            param_group['lr'] = value   
            
def accuracy(output, label, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    Parameters
    ----------
    output: tensor
        The output of the model, the shape is [batch_size, num_classes]
    label: tensor
        The label of the input, the shape is [batch_size, 1]
    topk: tuple, optional
        The specified list of value k, default just compute the top1_acc
    Return
    ------
    result: list
        Each element contain an accuracy value for a specified k
    """
#     print('label:', label)
    maxk = max(topk)
    batch_size = output.size(0) # The type of batch_size is int, not tensor
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = float(correct[:k].sum())
        result.append(correct_k * 100.0 / batch_size)
    
    return result            