import time
import os
from utils import detect_objects
import torch
from voc_eval import voc_eval
from pycocotools.cocoeval import COCOeval
from config import device
import tempfile
import json


def test(epoch, device, vis, test_loader, model, criterion, opts, priors_cxcy=None, eval=False):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    # for VOC evaluation
    # Lists to store detected and true boxes, labels, scores of whole test data
    # test_dataset 에 대하여 다 넣는 list
    det_additional = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    # for COCO evaluation
    results = []
    image_ids = []

    with torch.no_grad():

        # for idx, (images, boxes, labels, difficulties, additional_info) in enumerate(test_loader):
        for idx, datas in enumerate(test_loader):
            images = datas[0]
            boxes = datas[1]
            labels = datas[2]
            # difficulties = datas[3] 안쓰임

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            predicted_locs, predicted_scores = model(images)
            loss, (loc, cls) = criterion(predicted_locs, predicted_scores, boxes, labels)
            # loss = torch.zeros()

            sum_loss += loss.item()

            # ---------- eval ----------
            if eval:
                pred_boxes, pred_labels, pred_scores = detect_objects(priors_cxcy,
                                                                      predicted_locs,
                                                                      predicted_scores,
                                                                      min_score=opts.conf_thres,
                                                                      max_overlap=0.45,
                                                                      top_k=200)

                if is_coco:
                    # --- for COCO ---
                    # step 0
                    # coco eval parameter 2개 + image_ids 만들기
                    image_id = test_loader.dataset.image_ids[idx]
                    image_ids.append(image_id)

                    # step 1
                    # pred_boxes 를 x1, y1, x2, y2 to x1, y1, w, h coordinate 로 다시 바꿔줌
                    pred_boxes[:, 2] -= pred_boxes[:, 0]
                    pred_boxes[:, 3] -= pred_boxes[:, 1]

                    image_info = test_loader.dataset.coco.loadImgs(image_id)[0]
                    w = image_info['width']
                    h = image_info['height']

                    # re-scaling
                    pred_boxes[:, 0] *= w
                    pred_boxes[:, 2] *= w
                    pred_boxes[:, 1] *= h
                    pred_boxes[:, 3] *= h

                    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels,
                                                                pred_scores):  # pred object 의 갯수
                        if int(pred_label) == 0:
                            continue

                        coco_result = {
                            'image_id': image_id,
                            'category_id': test_loader.dataset.label_to_coco_label(int(pred_label - 1)),
                            # 0 : background 하나씩 줄이는게 맞음
                            'score': float(pred_score),
                            'bbox': pred_box.tolist(),
                        }
                        results.append(coco_result)

                else:

                    # --- for VOC --- (68 ~ 71)
                    additional_info = datas[4]
                    additional_info = additional_info[0]                # img_name, img_width, img_height --> diff name
                    det_additional.append(additional_info)                           # 4952 len list # [3] - name, w, h
                    det_boxes.append(pred_boxes.cpu())                               # 4952 len list # [obj, 4]
                    det_labels.append(pred_labels.cpu())                             # 4952 len list # [obj]
                    det_scores.append(pred_scores.cpu())                             # 4952 len list # [obj]

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

        if is_coco:

            # --- for COCO ---
            _, tmp = tempfile.mkstemp()
            json.dump(results, open(tmp, 'w'))

            # json.dump(results, open('{}_bbox_results.json'.format(test_loader.dataset.set_name), 'w'))
            cocoGt = test_loader.dataset.coco
            cocoDt = cocoGt.loadRes(tmp)
            # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

            coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP = coco_eval.stats[0]  # 0: AP, 1: .5 AP, 2: .75 AP, 3

        else:
            # --- for VOC --- (117~118)
            test_root = os.path.join(opts.data_root, 'TEST', 'VOC2007', 'Annotations')
            mAP = voc_eval(test_root, det_additional, det_boxes, det_scores, det_labels)

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
    parser.add_argument('--save_file_name', type=str, default='ssd_vgg_16_coco')
    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--data_root', type=str, default='D:\Data\VOC_ROOT')
    # "/home/cvmlserver3/Sungmin/data/VOC_ROOT"
    test_opts = parser.parse_args()
    print(test_opts)

    # 1. epoch
    epoch = 18

    # 2. device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. data set
    # test_set = VOC_Dataset(root=test_opts.data_root, split='TEST')
    from dataset.coco_dataset import COCO_Dataset
    test_set = COCO_Dataset(root_dir='D:\Data\coco', set_name='val2017', split='TEST')

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=4)
    # 6. network
    model = SSD(VGG(pretrained=True), n_classes=81).to(device)
    priors_cxcy = create_anchor_boxes()  # cx, cy, w, h - [8732, 4]

    # 7. loss
    criterion = MultiBoxLoss(priors_cxcy=priors_cxcy)

    test(epoch=epoch,
         device=device,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         priors_cxcy=priors_cxcy,
         eval=True,
         opts=test_opts,
         )







