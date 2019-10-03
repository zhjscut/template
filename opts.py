import argparse
import os

def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the M3SDA dataset',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataset options
#     parser.add_argument('--data_path_source', type=str, default='data/Office-31/webcam', help='Root of the source dataset (default: webcam in Office-31)')
#     parser.add_argument('--data_path_target', type=str, default='data/Office-31/amazon', help='Root of the target dataset (default: amazon in Office-31)')
#     parser.add_argument('--num_classes_s', type=int, default=1000, help='Number of classes of source dataset')
#     parser.add_argument('--num_classes_t', type=int, default=120, help='Number of classes of target dataset')
    parser.add_argument('--source_domain', type=str, default='webcam', help='Name of the source dataset (default: webcam in Office-31)')
    parser.add_argument('--target_domain', type=str, default='amazon', help='Name of the target dataset (default: amazon in Office-31)')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes of the dataset')
    parser.add_argument('--mixup', default=False, action='store_true', help='The indicator of whether use mix-up trick (default: no)')


    # optimization options
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--epoch_count_dataset', type=str, default='source', help='which dataset is used to count epoch, source or target')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of data (default: 128)')
    # parser.add_argument('--batch_size_source', type=int, default=128, help='Batch size of source data (default: 128)')
    # parser.add_argument('--batch_size_target', type=int, default=128, help='Batch size of target data (default: 128)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[5000, 7500, 9000], help='Decrease learning rate at these epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='Lr is multiplied by gamma on schedule')
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty)')
    parser.add_argument('--seed', type=int, default=666, help='The random seed for pytorch (default: 666)')
    
    # checkpoints options
    parser.add_argument('--start_epoch', type=int, default=0, help='Manual epoch number (used for restarting)')
    parser.add_argument('--resume', type=str, default='', help='Path of the checkpoint file to resume(default empty)')
    
    # architecture options
    parser.add_argument('--arch', type=str, default='resnet34', help='Model name')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Whether use pretrained model')
#     parser.add_argument('--pretrained', type=bool, default=True, help='Whether use pretrained model')
    parser.add_argument('--ablation', type=str, default='', help='Used in ablation study of model')

    # i/o options
    parser.add_argument('--log', type=str, default='log', help='Directory of log and checkpoint file')
    parser.add_argument('--result', type=str, default='log', help='Directory of overview results file')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', type=int, default=10, help='The frequency printing the result while training (default: 10 epoch)')
    parser.add_argument('--test_freq', type=int, default=1, help='The frequency printing the result while testing (default: 500 epoch)')
    parser.add_argument('--record_freq', type=int, default=1, help='The frequency recoding the result into the log file while training (default: 500 epoch)')
    args = parser.parse_args()
    if args.mixup:
        # args.log = os.path.join(args.log, str(args.ratio_s) + '_W_to_' + str(args.ratio_t) + '_A' + '_' + str(args.batch_size_target) + 'Timg_' + str(args.batch_size_source) + 'Simg_directly_DANN_mixup')
        args.log = os.path.join(args.log, str(args.ratio_s) + '_W_to_' + str(args.ratio_t) + '_A' + '_' + str(args.batch_size) + 'Timg_' + str(args.batch_size) + 'Simg_directly_DANN_mixup')
    else:
        # args.log = os.path.join(args.log, str(args.ratio_s) + '_W_to_' + str(args.ratio_t) + '_A' + '_' + str(args.batch_size_target) + 'Timg_' + str(args.batch_size_source) + 'Simg_directly_DANN')
#         args.log = os.path.join(args.log, str(args.ratio_s) + '_W_to_' + str(args.ratio_t) + '_A' + '_' + str(args.batch_size) + 'Timg_' + str(args.batch_size) + 'Simg_directly_DANN')
#         args.log = os.path.join(args.log, '_'.join(['bs'+str(args.batch_size), 'lr'+str(args.lr), 'ep'+str(args.epochs), args.arch, args.ablation, args.data_path_source.split('/')[-1][0] + '2' + args.data_path_target.split('/')[-1][0] ]) )  
        args.log = os.path.join(args.log, '_'.join(['bs'+str(args.batch_size), 'lr'+str(args.lr), 'ep'+str(args.epochs), args.arch, args.source_domain + '2' + args.target_domain ]) )  


    return args