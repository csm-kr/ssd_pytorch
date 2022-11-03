# SSD detection

re-implementation of SSD detection : https://arxiv.org/abs/1512.02325

### Training Setting

```
- batch size : 32
- optimizer : SGD
- epoch : 100 
- initial learning rate 0.001
- weight decay : 5e-4
- momentum : 0.9
- scheduler : MultiStepLR 1e-3(~30), 1e-4(~60), 1e-5(~100)
```

### Results

- voc

|methods     |  Training Dataset   |   Testing Dataset  | Resolution |     AP50        |Time | Fps  |
|------------|---------------------|--------------------|------------| ----------------|-----|------|
|papers      |2007 + 2012          |  2007              | 300 x 300  |      74.3       |     |  46  |
|this repo   |2007 + 2012          |  2007              | 300 x 300  |   75.81(+2.61)  |     |      |


- coco

|methods     | Training Dataset   |    Testing Dataset     | Resolution | AP        |AP50     |AP75    |Time | Fps  |
|------------|--------------------| ---------------------- | ---------- | --------- |---------|--------| ----| ---- |
|papers      | COCOtrain2017      |  COCO test-dev         | 300 x 300  |  23.2     |41.2     |23.4    |-    | -    |
|ours        | COCOtrain2017      |  COCOval2017(minival)  | 300 x 300  |  -        |-        |-       |-    | -    |

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


