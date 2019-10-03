import torch
import torch.nn as nn
import torchvision
import os
import shutil
from tensorboardX import SummaryWriter
from datetime import datetime 

from models.DANN import Extractor, Class_classifier, Domain_classifier
# from models.resnet import resnet
from opts import opts # The options for the project
from data.prepare_data import generate_dataloader
from utils import check_model, drop_msecond, time_delta2str


def main():
    start_time = datetime.now()
    start_time_str = datetime.strptime(drop_msecond(start_time),"%Y-%m-%d %H:%M:%S")
    args = opts()
    from trainer import train, validate
#     if args.ablation == '':
#         from trainer import train, validate
#     elif args.ablation == 'baseline':
#         from trainer_baseline import train, validate
#     elif args.ablation == 'wo_taskt':
#         from trainer_wo_taskt import train, validate
#     elif args.ablation == 'wo_Mst':
#         from trainer_wo_Mst import train, validate
#     elif args.ablation == 'wo_confusion':
#         from trainer_wo_confusion import train, validate
#     elif args.ablation == 'wo_category_confusion':
#         from trainer_wo_category_confusion import train, validate        
        
    # 将每一个epoch洗牌后的序列固定, 以使多次训练的过程中不发生较大的变化(到同一个epoch时会得到同样的模型)
    # 师兄说不固定也问题不大，他一般都没固定
#     if args.seed != 666:
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(args.seed)
#             torch.manual_seed(args.seed)
#         else:
#             torch.manual_seed(args.seed)
#     else: 
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(666)
#             torch.manual_seed(args.seed)
#         else:
#             torch.manual_seed(666)  
            
    # init models, multi GPU
#     model = nn.DataParallel(resnet(args)) # multi-GPU
    feature_extractor = nn.DataParallel(Extractor(args))
    class_classifier = nn.DataParallel(Class_classifier(2048, num_classes=args.num_classes)) # 512 for ResNet18 and 32, 2048 for ResNet50
    domain_classifier = nn.DataParallel(Domain_classifier(2048, hidden_size=128))
#     print(id(model.module)) 
#     check_model([3, 200, 200], Extractor(args))

    if torch.cuda.is_available():
#         model = model.cuda()
        feature_extractor = feature_extractor.cuda()
        class_classifier = class_classifier.cuda()
        domain_classifier = domain_classifier.cuda()
        
    # optimizer for multi gpu
    optimizer = torch.optim.SGD([
        {'params': feature_extractor.module.parameters(), 'name': 'pre-trained'},
        {'params': class_classifier.module.parameters(), 'name': 'new-added'},
        {'params': domain_classifier.module.parameters(), 'name': 'new-added'}        
    ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay, 
        nesterov=True)
    
    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])   
        else:
            raise ValueError('The file to be resumed is not exited', args.resume)
            
    train_loader_source, train_loader_target, val_loader_target = generate_dataloader(args)

    print('Begin training')
    print(len(train_loader_source), len(train_loader_target))
    train_loader_source_batches = enumerate(train_loader_source)
    train_loader_target_batches = enumerate(train_loader_target)
    if torch.cuda.is_available():
        criterion_y = nn.CrossEntropyLoss().cuda()
        criterion_d = nn.CrossEntropyLoss().cuda() # not used in this code
    else:
        criterion_y = nn.CrossEntropyLoss()
        criterion_d = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=args.log)
#     for epoch in range(args.start_epoch, args.epochs):
    epoch = args.start_epoch
    epochs_has_not_been_improved = 0
    maximum_gap = 0
    while epoch < args.epochs:
        # train for one epoch
#         pred1_acc_train, loss = train(train_loader_source, train_loader_source_batches, train_loader_target, 
#                                       train_loader_target_batches, model, criterion_y, criterion_d, optimizer_C, optimizer_G, epoch, args)
#         pred1_acc_train, loss_C, loss_G = train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model, criterion_y, criterion_d, optimizer_C, optimizer_G, epoch, args)
#         pred1_acc_train, loss_C, loss_G, new_epoch_flag = train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model, criterion_y, criterion_d, optimizer_C, optimizer_G, epoch, args)
#         train_loader_source_batches, train_loader_target_batches, epoch, pred1_acc_train, loss_C, loss_G, new_epoch_flag = train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model, criterion_y, criterion_d, optimizer_C, optimizer_G, epoch, args)
# -------------尚未更新（开始），可能会有错误-------------
# -------------尚未更新（结束），可能会有错误-------------

        train_loader_source_batches, train_loader_target_batches, epoch, pred1_acc_train, loss_C, loss_G, new_epoch_flag = train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, feature_extractor, class_classifier, domain_classifier, criterion_y, criterion_d, optimizer, epoch, args)
    
        if new_epoch_flag:     
            # 测试一下如果没有这两个语句，会不会出现异常
#             train_loader_source_batches = enumerate(train_loader_source)
#             (inputs_source, labels_source) = train_loader_source_batches.__next__()[1]

            # evaluate on the val data
            if epoch % args.test_freq == (args.test_freq-1):
#                 prec1, _ = validate(None, val_loader_target, model, criterion_y, criterion_d, epoch, args)
                prec1, _ = validate(None, val_loader_target, feature_extractor, class_classifier, domain_classifier, criterion_y, criterion_d, epoch, args)

            
                is_best = prec1 > best_prec1
                if is_best:
                    epochs_has_not_been_improved = 0
                    best_prec1 = prec1
                    with open(os.path.join(args.log, 'log.txt'), 'a') as fp:
                        fp.write('      \nTarget_T1 acc: %3f' % (best_prec1))
                else:
                    epochs_has_not_been_improved += 1
                     
                writer.add_scalars('data/scalar_group', {'pred1_acc_valid': prec1,
                                                        'best_prec1': best_prec1}, epoch)         
                
                # updating the maximum distance between current and best
                current_gap = best_prec1 - prec1
                if current_gap > maximum_gap:
                    maximum_gap = current_gap       
                    
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
#                     'model_state_dict': model.state_dict(),
                    'feature_extractor_state_dict': feature_extractor.state_dict(),
                    'class_classifier_state_dict': class_classifier.state_dict(),
                    'domain_classifier_state_dict': domain_classifier.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                    },
                    is_best, args, epoch + 1)  
                    
    writer.close()
    
    end_time = datetime.now()
    end_time_str = datetime.strptime(drop_msecond(end_time),"%Y-%m-%d %H:%M:%S")
    through_time = end_time - start_time
    through_time_str = time_delta2str(through_time)
    
    with open(os.path.join(args.result, 'overview.txt'), 'a') as fp:
        fp.write('%s: \nbest_prec1:%.2f%%, epochs_has_not_been_improved:%d, maximum distance between current and best:%.2f%%\n\
start at %s, finish at %s, it takes %s \n' % (args.log.split('/')[1], best_prec1, epochs_has_not_been_improved, maximum_gap, start_time_str, end_time_str, through_time_str))
    


if __name__ == '__main__':
    main()