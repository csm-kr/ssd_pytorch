from torchvision import transforms
from PIL import Image
from model import SSD, VGG
from anchor_boxes import create_anchor_boxes
import torch
import os
import glob
import cv2
import time
from utils import detect_objects, rev_label_map
from config import device

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def save_det_txt_for_mAP(file_name, bbox, cls, score):
    '''
    file name 을 mAP 에 넣을 수 있도록 만드는 부분
    :param file_name:
    :param bbox:
    :param cls:
    :param score:
    :return:
    '''

    score = score[0]
    if not os.path.isdir('./pred'):
        os.mkdir('./pred')
    f = open(os.path.join("./pred", file_name + '.txt'), 'w')
    for idx, t in enumerate(bbox):
        if cls[idx] == 'background':
            continue
        class_name = cls[idx]
        data = class_name + \
               " " + str(score[idx].item()) + \
               " " + str(t[0].item()) + \
               " " + str(t[1].item()) + \
               " " + str(t[2].item()) + \
               " " + str(t[3].item()) + "\n"
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

    det_boxes, det_labels, det_scores = detect_objects(priors_cxcy, predicted_locs, predicted_scores,
                                                       min_score=min_score,
                                                       max_overlap=max_overlap,
                                                       top_k=top_k,
                                                       n_classes=21)
                                                       # 내부에서 cuda 로 올립니다.
    # detection time
    detection_time = time.time() - tic

    # 배치를 벗겨내는 부분
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    # 소수를 실제 bbox 좌표로 변경하는 부분
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    return det_boxes, det_labels, det_scores, detection_time


if __name__ == '__main__':

    visualization = True
    model = SSD(VGG(pretrained=True))

    # use custom training pth file
    epoch = 59
    checkpoint = torch.load(os.path.join('./saves', 'ssd_vgg_16') + '.{}.pth.tar'.format(epoch))
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
    img_path = 'D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages'
    img_paths = glob.glob(os.path.join(img_path, '*.jpg'))

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
            boxes, labels, scores, det_time = demo(img, model=model, min_score=0.35, max_overlap=0.45, top_k=200,
                                                   priors_cxcy=priors_cxcy)

            # FIXME save for mAP??? --> ./pred
            name = os.path.basename(img_path).split('.')[0]  # .replace('.jpg', '.txt')
            save_det_txt_for_mAP(file_name=name, bbox=boxes, cls=labels, score=scores)

            total_time += det_time
            if i % 100 == 0:
                print("[{}/{}]".format(i, len(img_paths)))
                print("fps : {:.4f}".format(i / total_time))

            if visualization:

                # for plt
                # image = Image.open(img_path)
                # image_np = np.array(image)
                # plt.figure('result')
                # plt.imshow(image_np)
                # scores = scores[0]  # score is list of tensors
                #
                # for i in range(len(boxes)):
                #     print(labels[i])
                #     plt.text(x=boxes[i][0],
                #              y=boxes[i][1],
                #              s=labels[i],
                #              # s=voc_labels_array[int(labels[i].item()) - 1] + str(scores[i].item()),
                #              fontsize=10,
                #              bbox=dict(facecolor='red', alpha=0.5))
                #
                #     plt.gca().add_patch(Rectangle(xy=(boxes[i][0], boxes[i][1]),
                #                                   width=boxes[i][2] - boxes[i][0],
                #                                   height=boxes[i][3] - boxes[i][1],
                #                                   linewidth=1, edgecolor='r', facecolor='none'))
                #
                # plt.show()


                img = cv2.imread(img_path)
                scores = scores[0]  # score is list of tensors
                for i in range(len(boxes)):
                    cv2.rectangle(img,
                                  pt1=(boxes[i][0], boxes[i][1]),
                                  pt2=(boxes[i][2], boxes[i][3]),
                                  color=(0, 0, 255),
                                  thickness=2)

                    cv2.putText(img,
                                text=labels[i],
                                org=(boxes[i][0] + 10, boxes[i][1] + 10),
                                fontFace=0, fontScale=0.7,
                                color=(0, 255, 0))



                cv2.imshow('input', img)
                cv2.waitKey(0)

        print("fps : {:.4f}".format(len(img_paths) / total_time))




