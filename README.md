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

```
https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
voc -  120000
coco - 400000
100 epoch :  50000 iters [ 80, 160 ]
200 epoch : 100000 iters

voc : 16551
batch : 32  
517.218 ~ 518 iters per epoch, 2 epochs == 1K


so 10s epoch is 5K and 100 epochs 50K, 200s epoch 100K (to 1e-4)
120 epochs is 60K, 40 epochs is 20 K

120

coco : 117,266
batch : 32
3665 iters per epoch, 1 epoch == 3~4K
160, 40, 40
45 epoch, 10 epoch, 10 epoch 

100 epoch : 400000 iters
200 epoch : 800000 iters
```

### Results

- voc

|methods     |  Training Dataset   |   Testing Dataset  | Resolution |     AP50        | Time | Fps  |
|------------|---------------------|--------------------|------------| ----------------|------|------|
|papers      |2007 + 2012          |  2007              | 300 x 300  |      74.3       |      |  46  |
|this repo   |2007 + 2012          |  2007              | 300 x 300  |   75.58(+1.28)  |      |      |


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


