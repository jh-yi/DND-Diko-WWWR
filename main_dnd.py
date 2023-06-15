"""
Baseline script for CVPPA, ICCVW 2023 with the dataset challenge of DND-Diko-WWWR. 

Author: Jinhui Yi
Date: 2023.06.01
"""

from pprint import pprint
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from configs.config import cfg_from_file, project_root, get_arguments
from datasets.dnd_dataset import DND
from models.my_model import MyModel
from utils.pytorch_misc import * 
import yaml

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.cuda.synchronize()

def train_epoch(model, train_loader, cfg, optimizer, epoch_num):

    model.train()

    # init
    loss_epoch = AverageMeter()
    top1_epoch = AverageMeter()

    accumulate_step = 0
    optimizer.zero_grad()

    timer_epoch = Timer()
    timer_batch_avg = Timer()

    timer_epoch.tic()
    timer_batch_avg.tic()
    for batch_num, batch in enumerate(train_loader):

        img = batch['img'].to(device)
        labels = batch['label_idxs'].to(device)
        bsz = labels.shape[0]

        scores = model.forward(img)
        loss = F.cross_entropy(scores, labels)
        loss_epoch.update(loss, labels.shape[0])
        
        top1_batch = top_k_accuracy(scores, labels, topk=(1, ))[0]
        top1_epoch.update(top1_batch, bsz) 

        # gradient accumulation
        accumulate_step += 1
        loss /= args.acc_bsz
        loss.backward()
        if accumulate_step == args.acc_bsz:
            optimizer.step() 
            optimizer.zero_grad()
            accumulate_step = 0

        batch_time_avg = timer_batch_avg.toc(average=True)
        if batch_num % args.p_interval == 0:
            print("epo:{:2d}-{:4d}/{:4d}, exp {:4.1f}m/epo, avg {:4.3f}s/b, loss_batch: {:.3f} Acc (train-batch): {:5.2f}% Acc (train-epoch): {:5.2f}%"
                  .format(epoch_num, batch_num, len(train_loader) - 1, len(train_loader) * batch_time_avg / 60, batch_time_avg,
                          loss.item(), top1_batch.item() * 100.0, top1_epoch.avg.item() * 100.0), flush=True)
        timer_batch_avg.tic()
            
    print("Epoch {}: elapsed_time: {:.2f}m Acc (train): {:.2f}% {} loss_epo_avg: {:.3f}"
    .format(epoch_num, (timer_epoch.toc(average=False)) / 60, top1_epoch.avg.item() * 100.0, [top1_epoch.sum.item(), top1_epoch.count], loss_epoch.avg.item()), flush=True)   

def val_epoch(model, data_loader, cfg, is_final=False):

    model.eval()

    # init
    loss_epoch = AverageMeter()
    top1_epoch = AverageMeter()
    timer_epoch = Timer()
    timer_epoch.tic()
        
    results = []
    for batch_num, batch in enumerate(data_loader):

        img = batch['img'].to(device)
        bsz = img.shape[0]

        with torch.no_grad():
            scores = model.forward(img)
            preds = scores.argmax(1)

            if data_loader.dataset.split == 'val':
            # if not is_final:
                labels = batch['label_idxs'].to(device)
                loss = F.cross_entropy(scores, labels)
                loss_epoch.update(loss, bsz)
            
                top1_batch = top_k_accuracy(scores, labels, topk=(1,))[0]
                top1_epoch.update(top1_batch, bsz)

            results.append(batch['img_name'][0]+' '+str(int(preds)))

    if data_loader.dataset.split == 'val':
    # if not is_final:
        print("Val: elapsed_time: {:.2f}m Acc (val): {:.2f}% {} loss_epo_avg: {:.3f}"
        .format((timer_epoch.toc(average=False)) / 60, top1_epoch.avg.item() * 100.0, [top1_epoch.sum.item(), top1_epoch.count], loss_epoch.avg.item()), flush=True)  

    if is_final:
        res_path = os.path.join(project_root, 'codalab', 'res'+'_'+data_loader.dataset.split)
        os.makedirs(res_path, exist_ok=True)
        pred_path = os.path.join(res_path, 'predictions_{}.txt'.format(cfg.CROPTYPE))
        with open(pred_path, 'w') as f:
            f.write('\n'.join(results))
        print("Predictions saved at {}".format(pred_path))
        
    return top1_epoch.avg * 100.0


def train_model(model, train_loader, val_loader, test_loader, start_epoch, cfg):
    optimizer, scheduler = get_optim(model, cfg)

    best_acc = 0.0
    best_path = ''
    for epoch in range(start_epoch + 1, start_epoch + 1 + cfg.TRAIN.MAX_EPOCH):
        print("=================Epo {}: Training=================".format(epoch))
        train_epoch(model, train_loader, cfg, optimizer, epoch)                                     

        # separate train and val set
        if val_loader is not None: 
            print("=================Epo {}: Validating=================".format(epoch))
            cur_acc = val_epoch(model, val_loader, cfg, is_final=False)

            # save the best model
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_path = save_best_model(epoch, model, optimizer, suffix=cfg.exp_name)
            scheduler.step(cur_acc)

        # w/o val set
        else:
            best_path = save_best_model(epoch, model, optimizer, suffix=cfg.exp_name)   # actually saving current model
            scheduler.step()
            print("Current lr: ", scheduler.get_last_lr())

        if any([pg['lr'] <= cfg.TRAIN.LR * cfg.NGPUS * cfg.bsz / 999.0 for pg in optimizer.param_groups]):
            print("=================Testing=================")

            # inference
            if best_path:
                model.cpu()
                cfg.restore_from = best_path
                model, _ = load_model(cfg, model)
                model.to(device)
            _ = val_epoch(model, test_loader if cfg.codalab_pred == 'test' else val_loader, cfg, is_final=True)
            
            print("Exiting training early", flush=True)
            break

if __name__ == '__main__':
    # init
    args = get_arguments()
    assert args.cfg is not None, 'Missing cfg file'
    cfg = cfg_from_file(args.cfg)
    cfg.update(vars(args))          # change cfg according to args
    print('Called with args:')
    print(args)
    print('Using config:')
    pprint(cfg)
    set_seed(cfg.SEED)

    # dataset & dataloader
    if not args.is_test:
        train_dataset = DND(cfg, split=cfg.train_set)
        train_loader = DataLoader(train_dataset, batch_size=cfg.bsz, shuffle=True, num_workers=cfg.NWORK, drop_last=False)
        # val_dataset = DND(cfg, split='val') 
        # val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)
        val_loader = None
    test_dataset = DND(cfg, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)

    # model
    cfg.num_classes = len(test_dataset.class_to_ind)
    model = MyModel(cfg)
    # print(print_para(model), flush=True)

    # load model
    model, start_epoch = load_model(cfg, model)
    model.to(device)

    if not args.is_test:
        train_model(model, train_loader, val_loader, test_loader, start_epoch, cfg)
    else:
        print("=================Start testing!=================", flush=True)
        _ = val_epoch(model, test_loader, cfg, is_final=True)