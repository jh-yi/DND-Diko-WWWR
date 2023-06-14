# Miscellaneous functions that might be useful for pytorch


import torch
import random
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR
from torch import optim
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def top_k_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k

    [top_a, top_b, top_c, ...], each element is an accuracy over a batch. 
    
    output: (B, n_cls)
    target: (n_cls)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # (B, maxk), (B, maxk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))   # (maxk, B)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))    # 100.0 -> %
        return res


# Source: adapt from https://github.com/rowanz/neural-motifs/blob/master/lib/pytorch_misc.py
def print_para(model, is_sorted=True):
    """
    Prints parameters of a model
    For example:
        555.6M total parameters
        -----
        detector.roi_fmap.0.weight                        : [4096,25088]    (102760448) (    )
        union_vgg.1.0.weight                              : [4096,25088]    (102760448) (grad)
        roi_fmap.1.0.weight                               : 7[4096,25088]    (102760448) (grad)
        roi_fmap_obj.0.weight                             : [4096,25088]    (102760448) (grad)
        detector.roi_fmap.3.weight                        : [4096,4096]     (16777216) (    )
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    if is_sorted:
        st_sorted = sorted(st.items(), key=lambda x: -x[1][1])
    else:
        st_sorted = st.items()
    for p_name, (size, prod, p_req_grad) in st_sorted:
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))

    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


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


def freeze_para(model, freeze_name='features'):
    for n, param in model.named_parameters():
        if n.startswith(freeze_name):
            param.requires_grad = False


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def get_optim(model, cfg):
    opt = cfg.TRAIN.OPTIM
    lr = cfg.TRAIN.LR   # better for DND
    # lr = cfg.TRAIN.LR * cfg.NGPUS * cfg.bsz  # a hack
    l2 = cfg.TRAIN.WEIGHT_DECAY
    momentum = cfg.TRAIN.MOMENTUM
    if opt == 'sgd':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], weight_decay=l2, lr=lr, momentum=momentum)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=l2, lr=lr, eps=1e-3)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

    if cfg.train_set == 'train':
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.5, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=0)
    elif cfg.train_set == 'trainval':
        # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = MultiStepLR(optimizer, milestones=[20, 30, 35, 40, 45, 50], gamma=0.1)
    print("Scheduler: ", scheduler)

    return optimizer, scheduler


def load_model(cfg, model):
    # Source: https://github.com/rowanz/neural-motifs/blob/master/lib/pytorch_misc.py
    def optimistic_restore(model, state_dict):

        mismatch = False
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
                mismatch = True
            elif param.size() == own_state[name].size():
                own_state[name].copy_(param)
            else:
                print("model has {} with size {}, ckpt has {}".format(name, own_state[name].size(), param.size()))
                mismatch = True

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print("We couldn't find {} from ckpt".format(','.join(missing)))
            mismatch = True

        return not mismatch

    start_epoch = -1
    
    if len(cfg.restore_from) != 0:
        ckpt = torch.load(cfg.restore_from)
        start_epoch = ckpt['epoch']
        # optimizer.load_state_dict(ckpt['optimizer'])      # if available
        print("Loading everything from {}".format(cfg.restore_from))
  
        if not optimistic_restore(model, ckpt['state_dict']):
            print("Mismatch! Loading something from {}".format(cfg.restore_from))
        del ckpt
    else:
        print("Loading nothing. Starting from scratch")

    return model, start_epoch


def save_best_model(epoch, model, optimizer, suffix=''):
    save_path = os.path.join('results', 'best_model'+'_'+suffix+'.pth') # save mem, or 'model_{}.pth'.format(epoch))
    
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'state_dict': {k:v for k,v in model.state_dict().items() if not k.startswith('detector.')},
    }, save_path)
    print("Epo {:3}: Saving best ckpt to {}".format(epoch, save_path))
    return save_path


# Srouce: https://github.com/YanchaoYang/FDA/blob/master/utils/timer.py
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

if __name__ == "__main__":
    pass
