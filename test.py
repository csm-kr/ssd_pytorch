import os
import time
import torch
import numpy as np
from tqdm import tqdm
from evaluation.evaluator import Evaluator


@torch.no_grad()
def test_and_eval(opts, epoch, vis, device, loader, model, criterion, optimizer=None, scheduler=None, xl_log_saver=None,
                  result_best=None, is_load=False):
    if opts.rank == 0:
        # 0. evaluator
        evaluator = Evaluator(opts=opts)  # opts.data_type : voc or coco

        # 1. device
        # device = torch.device(f'cuda:{int(opts.gpu_ids[opts.rank])}')

        # 2. load .pth
        checkpoint = None
        if is_load:
            checkpoint = torch.load(f=os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch)),
                                    map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        tic = time.time()
        sum_loss = []

        print(f'{opts.data_type} dataset evaluation...')

        for idx, data in enumerate(tqdm(loader)):

            images = data[0]
            boxes = data[1]
            labels = data[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            anchors = model.module.anchors.to(device)

            # ---------- loss ----------
            pred = model(images)
            loss, (cls_loss, loc_loss) = criterion(pred, boxes, labels, anchors)
            sum_loss.append(loss.item())

            # ---------- predict ----------
            pred_boxes, pred_labels, pred_scores = model.module.predict(pred, anchors, opts)

            if opts.data_type == 'voc':

                info = data[3][0]  # [{}]
                info = (pred_boxes, pred_labels, pred_scores, info['name'], info['original_wh'])

            elif opts.data_type == 'coco':

                img_id = loader.dataset.img_id[idx]
                img_info = loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)
            toc = time.time()

            # ---------- print ----------
            if idx % opts.test_vis_step == 0 or idx == len(loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(loader.dataset)
        mean_loss = np.array(sum_loss).mean()
        print("mAP : ", mAP)
        print("mean Loss : ", mean_loss)
        print("Eval Time : {:.4f}".format(time.time() - tic))
        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss_' + opts.name,
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test_loss_{}'.format(opts.name),
                               legend=['test Loss', 'mAP']))

        if xl_log_saver is not None:
            xl_log_saver.insert_each_epoch(contents=(epoch, mAP, mean_loss))

        # save best.pth.tar
        if result_best is not None and optimizer is not None and scheduler is not None:
            if result_best['mAP'] < mAP:
                print("update best model from {:.4f} to {:.4f}".format(result_best['mAP'], mAP))
                result_best['epoch'] = epoch
                result_best['mAP'] = mAP
                if checkpoint is None:
                    checkpoint = {'epoch': epoch,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))
        return


def test_worker(rank, opts):

    from datasets.build import build_dataloader
    from models.build import build_model
    from losses.build import build_loss

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = None

    # 4. dataloader
    _, test_loader = build_dataloader(opts)

    # 5. network
    model = build_model(opts)

    # 6. loss
    criterion = build_loss(opts)

    # 7. loss
    test_and_eval(opts=opts,
                  epoch=opts.test_epoch,
                  vis=vis,
                  device=device,
                  loader=test_loader,
                  model=model,
                  criterion=criterion,
                  is_load=True)


if __name__ == "__main__":
    import configargparse
    from config import get_args_parser

    parser = configargparse.ArgumentParser('SSD test', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    test_worker(0, opts)




