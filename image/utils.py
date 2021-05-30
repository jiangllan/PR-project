import os
import math
import torch
import shutil
import numpy as np
from metrics import getMetric


def cos_similarity(feature, csv, drop_itself=True, f1_threshold=0, mAP_threshold=0, topn=0):

    assert (f1_threshold > 0 or topn > 0) and not (f1_threshold > 0 and topn > 0)

    preds_f1 = []
    preds_mAP = []
    CHUNK = 4096

    # print('Finding similar images...')
    CTS = len(feature) // CHUNK
    if len(feature) % CHUNK != 0:
        CTS += 1

    all_distances = []
    for j in range(CTS):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(feature))
        # print('chunk', a, 'to', b)

        distances = torch.matmul(feature, feature[a:b].T).T
        distances = distances.data.cpu().numpy()
        all_distances.append(distances)

    all_distances = np.vstack(all_distances)

    if drop_itself:
        mask = 1 - np.diag(np.ones(all_distances.shape[0]))
        all_distances *= mask

    # match_result = np.zeros(all_distances.shape)

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(feature))

        distances = all_distances[a:b]

        if f1_threshold > 0:
            for k in range(b - a):
                target_number = len(csv.iloc[a + k, :].target) - 1
                IDX_f1 = np.where(distances[k, ] > f1_threshold)[0][:]
                IDX_mAP = np.argsort(distances[k, ])[::-1][:target_number]
                tmp_IDX_mAP = np.where(distances[k, IDX_mAP] > mAP_threshold)[0][:]
                IDX_mAP = IDX_mAP[tmp_IDX_mAP]
                o = csv.iloc[IDX_f1].posting_id.values
                preds_f1.append(o)
                o = csv.iloc[IDX_mAP].posting_id.values
                preds_mAP.append(o)

        if topn > 0:
            for k in range(b - a):
                IDX_f1_mAP = np.argsort(distances[k, ])[::-1][:topn]  # Top 10, except itself
                o = csv.iloc[IDX_f1_mAP].posting_id.values
                preds_f1.append(o)
                preds_mAP.append(o)
                # match_result[a + k, IDX_f1_mAP] = 1
    # import pickle
    # with open('../../log/image-only/pca/best_512_f1threshold_95e-2.pickle', 'wb') as f:
    #     pickle.dump(match_result, f)

    csv['oof_cnn_f1'] = preds_f1
    csv['oof_cnn_mAP'] = preds_mAP

    csv['f1'] = csv.apply(getMetric('f1', 'oof_cnn_f1'), axis=1)
    # print('CV score for baseline =', csv.f1.mean())
    csv['mAP'] = csv.apply(getMetric('mAP@10', 'oof_cnn_mAP'), axis=1)
    # print('CV score for baseline =', csv.mAP.mean())
    csv['mrr'] = csv.apply(getMetric('mrr', 'oof_cnn_mAP'), axis=1)
    # print('CV score for baseline =', csv.mrr.mean())
    return csv.f1.mean(), csv.mAP.mean(), csv.mrr.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        if args.warmup_epoch:
            T_warmup_epoch = args.warmup_epoch
            if epoch < T_warmup_epoch:
                lr = args.warmup_lr + (args.lr - args.warmup_lr) * ((epoch * nBatch + batch)  / (T_warmup_epoch * nBatch))
            else:
                T_total = args.epochs - args.warmup_epoch
                T_cur = epoch - args.warmup_epoch
                lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            T_total = args.epochs
            T_cur = epoch
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, args, is_best, filename):
    model_dir = args.log_dir
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    print("=> saving checkpoint '{}'".format(model_filename))
    torch.save(state, model_filename)
    if is_best:
        shutil.copy(model_filename, best_filename)
    print("=> saved checkpoint '{}'".format(model_filename))
    return

