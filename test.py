import time
import os
from utils import detect_objects
import torch
from voc_eval import voc_eval
from config import device


def test(epoch, device, vis, test_loader, model, criterion, opts, priors_cxcy=None, eval=False):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    # Lists to store detected and true boxes, labels, scores
    det_additional = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    with torch.no_grad():

        for idx, (images, boxes, labels, difficulties, additional_info) in enumerate(test_loader):
            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            additional_info = additional_info[0]  # img_name, img_width, img_height
            # ---------- loss ----------
            predicted_locs, predicted_scores = model(images)
            loss, (loc, cls) = criterion(predicted_locs, predicted_scores, boxes, labels)
            # loss = torch.zeros()

            sum_loss += loss.item()

            # ---------- eval ----------
            if eval:
                det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(priors_cxcy,
                                                                                     predicted_locs,
                                                                                     predicted_scores,
                                                                                     min_score=opts.conf_thres,
                                                                                     max_overlap=0.45,
                                                                                     top_k=200)

                det_additional.append(additional_info)                          # 4952 len list # [3]
                # batch 를 풀어주는 부분
                det_boxes.append(det_boxes_batch[0].cpu())                               # 4952 len list # [obj, 4]
                det_labels.append(det_labels_batch[0].cpu())                             # 4952 len list # [obj]
                det_scores.append(det_scores_batch[0].cpu())                             # 4952 len list # [obj]

            toc = time.time() - tic
            # ---------- print ----------
            # for each steps
            if idx % 1000 == 0:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc))

        window_test_root ="D:\Data\VOC_ROOT\TEST\VOC2007\Annotations"
        ubuntu_test_root ="/home/cvmlserver3/Sungmin/data/VOC_ROOT/TEST/VOC2007/Annotations"
        if opts.os_type == 'window':
            root = window_test_root
        elif opts.os_type == 'ubuntu':
            root = ubuntu_test_root
        mAP = voc_eval(root, det_additional, det_boxes, det_scores, det_labels)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


if __name__ == "__main__":

    from dataset.voc_dataset import VOC_Dataset
    from loss import MultiBoxLoss
    from model import VGG, SSD
    import visdom
    from test import test
    from anchor_boxes import create_anchor_boxes
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='ssd_vgg_16')
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--conf_thres', type=float, default=0.01)
    test_opts = parser.parse_args()
    print(test_opts)

    # 1. epoch
    epoch = 186

    # 2. device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. data set
    window_root = "D:\Data\VOC_ROOT"
    ubuntu_root = "/home/cvmlserver3/Sungmin/data/VOC_ROOT"
    test_set = VOC_Dataset(root=window_root, split='TEST')
    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=4)
    # 6. network
    model = SSD(VGG(pretrained=True)).to(device)
    priors_cxcy = create_anchor_boxes()  # cx, cy, w, h - [8732, 4]

    # 7. loss
    criterion = MultiBoxLoss(priors_cxcy=priors_cxcy)

    test(epoch=epoch,
         device=device,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         opts=test_opts,
         priors_cxcy=priors_cxcy,
         eval=True)







