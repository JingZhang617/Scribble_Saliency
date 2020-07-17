# Scribble_Saliency (CVPR2020)
Weakly-Supervised Salient Object Detection via Scribble Annotations

![alt text](./overview.png)

# Setup
Install Pytorch

# Trained Model

Please download the trained model and put it in "models"

https://drive.google.com/file/d/19mco_WjMAK7OKDMklxTrzot7wWhfSsr1/view?usp=sharing

# Train Model

1) Prepare data for training (We provided the related data in: https://drive.google.com/file/d/15uasGpd6fRUtpwo21LovFtzZBUh0zHF0/view?usp=sharing. Please download it and put it in the "data" folder)

      a) We have scribble dataset (1: foreground, 2: background, 0: unknown), raw RGB images, gray images and edge map from:https://github.com/yun-liu/rcf.

      b) Convert scribble data to "gt" and "mask" with matlab code: generate_gt_mask_from_scribble.m, where gt contains forergound scribble(s), and mask contains both foreground and background scribble(s).

      c) Convert RGB image to gray image with matlab code: convert_rgb2gray.m

2) Run ./train.py

# Test Model

1) Modify the testing image path accordingly.

2) Run ./test.py


# Scribble Dataset (S-DUTS Dataset)

![alt text](./scribble_show.png)

We manually labeled the benchmark saliency dataset DUTS with scribble, and provided three versions of scribble annotations with thin scribbles and wider scribbles (salient foreground region: 1, background region: 2, unknown pixels: 0):

1) thin scribbles:

https://drive.google.com/open?id=10fGhQBN5VQqeSyQDKAO5_P2_w9Nn5_w_

2) wider scribbles:

https://drive.google.com/open?id=1umNUJaU8pNlA4pIbV5MSDKHcKEYXPlRU

We also labeled the fixation prediction dataset Salicon (the 10K training training dataset) with scribble for further research on weakly supervised salient object detection and fully supervised fixation prediction.

3) scribble labeling of Salicon training dataset:

https://drive.google.com/open?id=1NhEdBl7pas0us_BvWsQVll_QtJJVh_JR

# Our Results:
![alt text](./results.png)

![alt text](./E_F_measure.png)

We provide saliency maps of our model on seven benchmark saliency dataset (DUT, DUTS, ECSSD, HKU-IS, PASCAL-S, SOD, THUR) as below:

https://drive.google.com/file/d/1njRCKDk89SX-um4aYN7vUV8ex05sI9ir/view?usp=sharing

# Benchmark Testing Dataset (DUT, DUTS, ECSSD, HKU-IS, PASCAL-S, SOD, THUR):

https://drive.google.com/open?id=11rPRBzqxdRz0zHYax995uvzQsZmTR4A7

# Our Bib:

Please cite our paper if necessary:
```
@inproceedings{jing2020weakly,
  title={Weakly-Supervised Salient Object Detection via Scribble Annotations},
  author={Zhang, Jing and Yu, Xin and Li, Aixuan and Song, Peipei and Liu, Bowen and Dai, Yuchao},
  booktitle=cvpr,
  year={2020}
}
```

# Contact

Please drop me an email for further problems or discussion: zjnwpu@gmail.com

