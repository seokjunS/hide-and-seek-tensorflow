# hide-and-seek-tensorflow

Tensorflow implementation of [Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization, ICCV 2017](https://arxiv.org/pdf/1704.04232.pdf)

## Description

Implement CAM and HAS in Tensorflow. Unlike the original paper, it is only tested on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/).

### Prerequisites

All codes are tested at Linux environment only.

* Python 2.7
* NumPy
* Scipy
* Scipy.misc
* Matplotlib
* Tensorflow 1.2 (or 1.3)
* [imgaug](https://github.com/aleju/imgaug)


### Installing

First clone the repository.

```
git clone https://github.com/seokjunS/hide-and-seek-tensorflow
cd hide-and-seek-tensorflow
```

Prepare `data` directory.

```
mkdir data
```

Downlaod Tiny ImageNet dataset at `data` and unzip it.

```
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
cd ..
```

Generate data. (labels.txt, train.tfrecord, valid.tfrecord)

```
python src/tiny_imagenet.py gen_data
ls -l data
```


## Running the models

Explain how to run various models.
Basic models used in paper is AlexnetGAP (at `src/alexnet_gap`) and GooglenetGAP (at `src/googlenet_gap`).
For intensive experiment, I implemented variation of models including,
* SmallAlexnetGAP - `src/small_alexnet_gap.py`: Resized Alexnet in order to feed 64x64 images.
* AlexnetGMP - `src/alexnet_gmp.py`: Replace GAP to GMP.
* DropAlexnetGAP - `src/drop_alexnet_gap.py`: Add dropout after mean normalization.
* Drop2AlexnetGAP - `src/drop2_alexnet_gap.py`: Add dropout before mean normalization.
* CustomnetGAP - `src/customnet_gap`: Introducing fire module of SqueezeNet.


### Training
Following command is an example of training AlexnetGAP with hiding 16 patches.
```
python src/train.py --method AlexnetGAP \
                    --train_dir models/alexnet_gap \
                    --max_epoch 200 \
                    --batch_size 128 \
                    --do_hide 16
```

To feed 64x64 image to original model (without resize) set `--without_resize True` option.

Two types of augmentation is implemented and can be set as '--do_augmentation 1' or '--do_augmentation 2' (only for AlexnetGAP).
* Augmentation 1
```
data = iaa.Sequential([
            iaa.Fliplr(0.25),
            iaa.Flipud(0.25),
            iaa.Sometimes(0.25, iaa.Affine(
              rotate=(-180, 180)
            )),
            iaa.Sometimes(0.2, iaa.Affine(
              translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)}
            ))
          ]).augment_images(data)
```
* Augmentation 2
```
data = iaa.Sequential([
            iaa.Fliplr(0.25),
            iaa.Flipud(0.25),
            iaa.Sometimes(0.25, iaa.Affine(
              rotate=(-180, 180)
            )),
            iaa.Sometimes(0.2, iaa.Affine(
              translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}
            )),
            iaa.Sometimes(0.2, iaa.OneOf([
              iaa.CoarseDropout(0.2, size_percent=(0.05, 0.1)),
              iaa.CoarseSalt(0.2, size_percent=(0.05, 0.1)),
              iaa.CoarsePepper(0.2, size_percent=(0.05, 0.1)),
              iaa.CoarseSaltAndPepper(0.2, size_percent=(0.05, 0.1))
            ]))
          ]).augment_images(data)
```

Please use `--help` command for options and descriptions.


### Evaluation
After training, to evaluate and visualize, please use `src/test.py`. Test is done in various localization thresholds {0.2, 0.3, 0.4, 0.5}

Following command is an example of evaluating trained AlexnetGAP.
```
python src/test.py --method AlexnetGAP \
                   --checkpoint models/alexnet_gap/ckpt/model-200
```

To visualize several well localized object, please set `--do_vis Ture` option.
To do a multi-crop test (10 crops: 4 corners + center, also for flipped one), please set `--do_multi_crop` option.


## Sample Results

### Result Table
Tests are done without multi-crop test and reported values are the highest one among four localization thresholds.

![alt text](https://github.com/seokjunS/hide-and-seek-tensorflow/blob/master/resource/res_table.png?raw=true)

### Visualizations

![alt text](https://github.com/seokjunS/hide-and-seek-tensorflow/blob/master/resource/4.png?raw=true)
![alt text](https://github.com/seokjunS/hide-and-seek-tensorflow/blob/master/resource/20.png?raw=true)
![alt text](https://github.com/seokjunS/hide-and-seek-tensorflow/blob/master/resource/64.png?raw=true)
![alt text](https://github.com/seokjunS/hide-and-seek-tensorflow/blob/master/resource/370.png?raw=true)


## Contact

* **Seokjun Seo** - [seokjunS](dane2522@gmail.com)
