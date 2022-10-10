import time
import argparse
import logging
import numpy as np
import torch
import pandas as pd
from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from scipy.special import softmax
import os 
from timm.utils import *

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--output_path', type=str, metavar='PATH', default='',
                    help='path to output files')
parser.add_argument('--scoreoutput_path', type=str, metavar='PATH', default='',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--crop-pct', default=1.0, type=float,
                    metavar='crop_num', help='crop_num')
parser.add_argument('--mode', default=1, type=int,
                    metavar='mode_num', help='choose train or test data')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument("--local_rank", default=0, type=int)
def main():
    setup_default_logging()

    args = parser.parse_args()
    args.rank = 0
    random_seed(args.seed, args.rank)

    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    print(args.mode)
    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

  

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    TTA = True

    class_to_index = {'aeroplane': 0,'bicycle': 1,'boat': 2,'bus': 3,'car': 4,'chair': 5,'diningtable': 6,'motorbike': 7,'sofa': 8,'train': 9}
    class_to_index_reverse = {0:'aeroplane',1:'bicycle',2:'boat',3:'bus',4:'car',5:'chair',6:'diningtable',7:'motorbike',8:'sofa',9:'train'}
    #val_nuisances = ['shape', 'pose', 'texture', 'context', 'weather','occlusion','iid']

    score_list = []
    #for nuisance in val_nuisances:
    dataset = ImageDataset("../data/phase2-cls/images")

    # path = os.path.join("../ROBINv1.1-cls-pose/ROBINv1.1-cls-pose/nuisances", nuisance+'/' + 'labels.csv')

    # train_metadata = pd.read_csv(path)
    # train_metadata['image_path'] = train_metadata.apply(lambda x: "../ROBINv1.1-cls-pose/ROBINv1.1-cls-pose/nuisances" + '/'+nuisance+'/Images/'+x['labels']+'/' + x['imgs'], axis=1)
    # train_dir = list(train_metadata['image_path'])
    # train_class = list(train_metadata['labels'])
    # samples=[]
    # for i in range(len(train_dir)):
    #     samples.append((train_dir[i],train_class[i]))
    # dataset.parser.samples = samples

    print(len(dataset))

    loader = create_loader(
        dataset,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=False,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'],
        TTA=TTA)

    model.eval()
    #crop_pct = 1.0 if test_time_pool else config['crop_pct']
    #print(crop_pct)
    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    preds_raw = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            # print(input.shape)
            input = input.cuda()

            if TTA:
                bs, ncrops, c, h, w = input.size()
                result = model(input.view(-1, c, h, w))  # fuse batch size and ncrops
                labels = result.view(bs, ncrops, -1).mean(1)  # avg over crops

            else:
                labels = model(input)

            
            preds_raw.extend(labels.to('cpu').numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))


    preds_raw = pd.DataFrame(preds_raw)

    scores = pd.DataFrame()
    if args.mode:
        scores['imgs'] = [dataset.parser.samples[i][0].split('/')[-1] for i in
                                range(len(dataset.parser.samples))]
        # scores['label'] = [class_to_index[dataset.parser.samples[i][1]] for i in
        #                         range(len(dataset.parser.samples))]
    # else:
    #     scores['imgs'] = [dataset.parser.samples[i][0].split('/')[-1].split('.')[0].split('-')[0] for i in
    #                             range(len(dataset.parser.samples))]

    scores_logit = pd.concat([scores['imgs'], preds_raw], axis=1)
    scores_logit.to_csv(args.scoreoutput_path+'final_logits.csv', index=False, header=True)  # 保存logits
    print(scores_logit.shape)

    preds_raw = softmax(preds_raw, axis=1)
    group_scores = pd.concat([scores['imgs'], preds_raw], axis=1)
    #group_scores = scores.groupby(['ObservationId']).mean().reset_index()

    submit = pd.DataFrame()
    # ObservationId,ClassId
    submit['imgs'] = group_scores['imgs']
    submit['pred'] = np.argmax(np.array(group_scores.iloc[:, 1:]), axis=1)
    # value = sum(i==0 for i in list(scores['label'] - submit['pred']))/len(scores['label'])
    # if nuisance!='iid':
    #     score_list.append(value*100)
    # print(nuisance,":",value*100)

    submit['pred'] = submit['pred'].map(class_to_index_reverse)

    
    submit.to_csv(args.output_path+'results.csv', index=False, header=True)
    # for i in range(len(val_nuisances)-1):
    #     print(val_nuisances[i], score_list[i])
    # print("Mean score:",sum(score_list)/len(score_list))

if __name__ == '__main__':
    main()