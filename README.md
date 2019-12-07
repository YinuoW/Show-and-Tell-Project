# ECE285_projectA
Pytorch implementation of image captioning based on the paper: [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)

## Code Organization
The files can be divided into three parts:
1. Demo file
- demo.ipynb: Generate caption for test image.
2. Main model files
- model.py: Provides CNN and RNN models for train and test.
- train.py: Train CNN and RNN model.
3. Helper files:
- resize.py: Preprocess the images to make the image sizes uniform.
- load_data.py: Creates the CoCoDataset and a DataLoader for it.
- build_vocab.py:  Build vocabulary dictionary.

## Setup
1. Download COCO dataset:
```
mkdir data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P ./data/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P ./data/
unzip ./data/captions_train-val2014.zip -d ./data/
unzip ./data/train2014.zip -d ./data/
unzip ./data/val2014.zip -d ./data/ 
```
2. Set up COCO API:
```
pip install --user pycocotools
```
3. other settings:
- Python 3
- nltk
- numpy
- scikit-image
- matplotlib
- tqdm


## Model training
```
python training.py
```

## Reference
1. [TensorFlow Implementation of show and tell] (https://github.com/nikhilmaram/Show_and_Tell)
2. [Show and Tell: A Neural Image Caption Generator] (https://arxiv.org/pdf/1411.4555.pdf)
3. [PyTorch Implementation of show and tell] (https://github.com/ntrang086/image_captioning)