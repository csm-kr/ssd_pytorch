import time
import os
import torch


def train(epoch, device, vis, train_loader, model, criterion, optimizer, scheduler, opts):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    for idx, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        loc, cls = model(images)
        loss, (loc, cls) = criterion(loc, cls, boxes, labels)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time() - tic

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % 100 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Cls_loss: {cls:.4f}\t'
                  'Loc_loss: {loc:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          cls=cls,
                          loc=loc,
                          lr=lr,
                          time=toc))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 3)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, cls, loc]).unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'Cls Loss', 'Loc Loss']))

    # # 각 epoch 마다 저장
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))



