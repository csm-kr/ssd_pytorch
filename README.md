# SSD detection

re-implementation of SSD detection : https://arxiv.org/abs/1512.02325

### Training Setting

```
- batch size : 32
- optimizer : SGD
- epoch : 200 
- initial learning rate 0.001 to (0.0001/0.00001)
- weight decay : 5e-4
- momentum : 0.9
- scheduler : step LR [120, 150]
```

### Results

- voc

|methods     |  Training Dataset   |   Testing Dataset  | Resolution |     AP50        | Time | Fps  |
|------------|---------------------|--------------------|------------| ----------------|------|------|
|papers      |2007 + 2012          |  2007              | 300 x 300  |      74.3       |      |  46  |
|this repo   |2007 + 2012          |  2007              | 300 x 300  |   75.58(+1.28)  |      |  42  |


- coco

|methods     | Training Dataset   |    Testing Dataset     | Resolution | AP        |AP50     |AP75    |Time | Fps  |
|------------|--------------------| ---------------------- | ---------- | --------- |---------|--------| ----| ---- |
|papers      | COCOtrain2017      |  COCO test-dev         | 300 x 300  |  23.2     |41.2     |23.4    |-    | -    |
|ours        | COCOtrain2017      |  COCOval2017(minival)  | 300 x 300  |  22.0     |37.7     |22.6    |-    | -    |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.066
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.534
```

### Start Guide

- train

```
python main.py --config ./configs/ssd_coco_train.txt
python main.py --config ./configs/ssd_voc_train.txt
```

- test
```
test.py --config ./configs/ssd_coco_test.txt
test.py --config ./configs/ssd_voc_test.txt
```

- demo
```
test.py --config ./configs/ssd_coco_demo.txt
test.py --config ./configs/ssd_voc_demo.txt
```


