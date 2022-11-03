# base library
import os
import torch
import visdom

# config
import configargparse
from config import get_args_parser

# dataset
from dataset.build import build_dataloader

# model
from models.build import build_model

# loss
from loss import MultiBoxLoss

# scheduler
from torch.optim.lr_scheduler import MultiStepLR

# log
from log import XLLogSaver

# train
from train import train_one_epoch

# test
from test import test_and_eval

# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

# distributed
import torch.multiprocessing as mp
from utils import init_for_distributed, resume


def main_worker(rank, opts):
    # 1. config
    print(opts)

    # 2. distributed
    if opts.distributed:
        init_for_distributed(rank, opts)

    # 3. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 4. visdom
    vis = visdom.Visdom(port=opts.visdom_port)

    # 5. data loader
    train_loader, test_loader = build_dataloader(opts)

    # 6. network
    model = build_model(opts)

    # 7. loss
    criterion = MultiBoxLoss(alpha=10.)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[120, 150], gamma=0.1)  # 115, 144

    # 10. logger
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP', 'val_loss'))

    # 11. resume
    model, optimizer, scheduler = resume(opts, model, optimizer, scheduler)

    # set best results
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):

        # 11. train
        train_one_epoch(epoch=epoch,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        opts=opts)

        # 12. test
        result_best = test_and_eval(epoch=epoch,
                                    vis=vis,
                                    test_loader=test_loader,
                                    model=model,
                                    criterion=criterion,
                                    opts=opts,
                                    xl_log_saver=xl_log_saver,
                                    result_best=result_best,
                                    is_load=False)

        scheduler.step()


if __name__ == "__main__":

    parser = configargparse.ArgumentParser('SSD training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)



