# libtorch_examples
This repo contains tutorials to use libtorch (Pytorch C++ API) for computer vision applications. 

Follow the codes in this order:
1. image_classification_pretrained 
2. image_classification_transfer_learning

There will be other tutorials for object detection and image segmentation.

## How to build?

First, download and unzip the pre-built version of libtorch:
```
cd ~
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
Then, install torchvision:
```
pip3 install torchvision
```
Then go inside a directory (e.g. image_classification_pretrained) and build the executable:
```
cd image_classification_pretrained
python3 save_jit.py
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/Pytorch/libtorch/ ..
make
./classify ../traced_resnet_model.pt ../data/imageNetLabels.txt ../data/panda.jpg
```

## Appendix
For more details read my posts [here](http://imrid.net/?p=4403) abd [here](http://imrid.net/?p=4414).
