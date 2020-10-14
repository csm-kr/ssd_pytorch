from torchvision import transforms
from PIL import Image
from model import SSD, VGG
from anchor_boxes import create_anchor_boxes
import torch
import os
import glob
import cv2
import time
from utils import detect_objects, rev_label_map, voc_color_array
from config import device
import argparse

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def save_det_txt_for_mAP(file_name, boxes, labels, scores):
    '''
    file name 을 mAP 에 넣을 수 있도록 만드는 부분
    :param file_name:
    :param bbox:
    :param label:
    :param score:
    :return:
    '''
    if not os.path.isdir('./pred'):
        os.mkdir('./pred')
    f = open(os.path.join("./pred", file_name + '.txt'), 'w')

    for idx, box in enumerate(boxes):
        if labels[idx] == 'background':
            continue
        class_name = labels[idx]
        data = class_name + \
               " " + str(scores[idx].item()) + \
               " " + str(box[0].item()) + \
               " " + str(box[1].item()) + \
               " " + str(box[2].item()) + \
               " " + str(box[3].item()) + "\n"
        f.write(data)
    f.close()


def demo(original_image, model, min_score, max_overlap, top_k, priors_cxcy=None):
    """

    :param original_image:
    :param model:
    :param min_score:
    :param max_overlap:
    :param top_k:
    :param priors_cxcy:
    :return:
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    tic = time.time()
    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # in demo using cpu is faster than gpu
    priors_cxcy = priors_cxcy.to('cpu')
    predicted_locs = predicted_locs.to('cpu')
    predicted_scores = predicted_scores.to('cpu')

    det_boxes, det_labels, det_scores = detect_objects(priors_cxcy,
                                                       predicted_locs,
                                                       predicted_scores,
                                                       min_score=min_score,
                                                       max_overlap=max_overlap,
                                                       top_k=top_k,
                                                       n_classes=21)
                                                       # 내부에서 cuda 로 올립니다.
    # detection time
    detection_time = time.time() - tic

    # 배치를 벗겨내는 부분
    det_boxes = det_boxes.to('cpu')
    det_scores = det_scores.to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    # 소수를 실제 bbox 좌표로 변경하는 부분
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels.to('cpu').tolist()]

    return det_boxes, det_labels, det_scores, detection_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='ssd_vgg_16')
    parser.add_argument('--conf_thres', type=float, default=0.35)
    # parser.add_argument('--img_path', type=str, default='D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages')
    parser.add_argument('--img_path', type=str, default='D:\Data\coco\images\\val2017')
    parser.add_argument('--visualization', type=bool, default=True)
    demo_opts = parser.parse_args()
    print(demo_opts)

    visualization = demo_opts.visualization
    model = SSD(VGG(pretrained=True))

    # use custom training pth file
    epoch = 2
    checkpoint = torch.load(os.path.join(demo_opts.save_path, demo_opts.save_file_name) + '.{}.pth.tar'.format(epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    # anchor box
    priors_cxcy = create_anchor_boxes()  # cx, cy, w, h - [8732, 4] 오름차순 *

    model = model.to(device)
    model.eval()

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # voc test
    img_paths = glob.glob(os.path.join(demo_opts.img_path, '*.jpg'))

    # ubuntu voc test
    # img_path = "/home/cvmlserver3/Sungmin/data/VOC_ROOT/TEST/VOC2007/JPEGImages"
    # img_paths = glob.glob(os.path.join(img_path, '*.jpg'))

    # rtts test
    # img_path = 'C:\\Users\csm81\Desktop\RTTS_mAP\RTTS\RTTS\JPEGImages'
    # img_paths = glob.glob(os.path.join(img_path, '*.png'))

    tic = time.time()
    total_time = 0

    print(len(img_paths))
    with torch.no_grad():
        for i, img_path in enumerate(img_paths):

            # for each a image, outputs are boxes and labels.
            img = Image.open(img_path, mode='r').convert('RGB')
            pred_boxes, pred_labels, pred_scores, det_time = demo(img, model=model, min_score=demo_opts.conf_thres,
                                                                  max_overlap=0.45, top_k=200,
                                                                  priors_cxcy=priors_cxcy)

            # FIXME save for mAP??? --> ./pred
            name = os.path.basename(img_path).split('.')[0]  # .replace('.jpg', '.txt')
            save_det_txt_for_mAP(file_name=name, boxes=pred_boxes, labels=pred_labels, scores=pred_scores)

            total_time += det_time
            if i % 100 == 0:
                print("[{}/{}]".format(i, len(img_paths)))
                print("fps : {:.4f}".format(i / total_time))

            if visualization:

                voc_labels_array = list(rev_label_map.values())
                img = cv2.imread(img_path)
                for i in range(len(pred_boxes)):

                    x_min = pred_boxes[i][0]
                    y_min = pred_boxes[i][1]
                    x_max = pred_boxes[i][2]
                    y_max = pred_boxes[i][3]

                    cv2.rectangle(img,
                                  pt1=(x_min, y_min),
                                  pt2=(x_max, y_max),
                                  color=voc_color_array[voc_labels_array.index(pred_labels[i])].tolist(),
                                  thickness=2)

                    text_size = cv2.getTextSize(text=pred_labels[i],
                                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                thickness=1)[0]

                    # text box
                    cv2.rectangle(img,
                                  pt1=(x_min, y_min),
                                  pt2=(x_min + text_size[0], y_min + text_size[1] + 3),
                                  color=voc_color_array[voc_labels_array.index(pred_labels[i])].tolist(),
                                  thickness=-1)

                    # text
                    cv2.putText(img,
                                text=pred_labels[i],
                                org=(x_min + 10, y_min + 10),
                                fontFace=0,
                                fontScale=0.4,
                                color=(255, 255, 255))

                cv2.imshow('input', img)
                cv2.waitKey(0)

        print("fps : {:.4f}".format(len(img_paths) / total_time))




