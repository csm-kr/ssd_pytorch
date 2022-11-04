import time
import os
import torch
from tqdm import tqdm


def train_one_epoch(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):

    print('Training of epoch [{}]'.format(epoch))

    # device
    device = torch.device(f'cuda:{int(opts.gpu_ids[opts.rank])}')
    tic = time.time()
    model.train()

    for idx, data in enumerate(tqdm(train_loader)):

        images = data[0]
        boxes = data[1]
        labels = data[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        anchors = model.module.anchors.to(device)

        pred = model(images)
        loss, (cls_loss, loc_loss) = criterion(pred, boxes, labels, anchors)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        toc = time.time() - tic

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % opts.vis_step == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Cls_loss: {cls:.4f}\t'
                  'Loc_loss: {loc:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          cls=cls_loss,
                          loc=loc_loss,
                          lr=lr,
                          time=toc))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 3)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, cls_loss, loc_loss]).unsqueeze(0).cpu(),
                         win='train_loss_' + opts.name,
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='train_loss_' + opts.name,
                                   legend=['Total Loss', 'cls Loss', 'loc Loss']))

    # epoch 마다 저장 안함
    if opts.rank == 0:
        save_path = os.path.join(opts.log_dir, opts.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        # checkpoint = {'epoch': epoch,
        #               'model_state_dict': model.state_dict(),
        #               'optimizer_state_dict': optimizer.state_dict(),
        #               'scheduler_state_dict': scheduler.state_dict()}
        #
        # torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))



