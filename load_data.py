import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import random
import json
from tqdm import tqdm


class CocoDataset(data.Dataset):
    def __init__(self, mode, transform, folder, annotation, vocab, batch_size,
                 start_word='<start>', end_word='<end>'):
        """For our dataset we need to initialize several variables.
            mode: mode can be either "train" or "val"
            transform: transformer of the image
            folder: the directory of images.
            annotation: coco provided annotation file.
            vocab: if there is no preexist vocabs, create a new one; othereise, load vocabs.
        """
        assert(mode == "train" or mode == "val")
        self.dataset = json.load(open(annotation, 'r'))
        self.mode = mode
        self.transform = transform
        self.folder = folder
        self.coco = COCO(annotation)
        self.ann_ids = list(self.coco.anns.keys())
        self.vocabulary = vocabulary(vocab)
        self.get_index()
        self.start_word = start_word
        self.end_word = end_word

    def __getitem__(self, index):
        # return image and caption as output
        coco = COCO(annotation)
        vocab = self.vocab
        curr_id = self.ann_ids[index]
        caption = coco.anns[curr_id]['caption']
        img_id = coco.anns[curr_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.folder, path)).convert('RGB')
        caption = caption2id(self, index, img_id)
        
        return image, caption
        
    def get_index(self):
        # generate index for annotations and images
        anns, imgs = {}, {}
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        if 'annotations' in self.dataset:
            for anns in self.dataset['annotations']:
                anns[ann['id']] = ann
        self.imgs = imgs
        self.anns = anns
    
    def caption2id(self, index, img_id):
        # convert captions to ids
        caption = coco.anns[curr_id]['caption']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.folder, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert captions to ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocabulary(self.start_word))
        caption.extend([vocab(token) for token in tokens])
        caption.append(self.vocabulary(self.end_word))
        caption = torch.Tensor(caption).long()
        return caption
        
    def __len__(self):
        # get the total number of ids
        return len(self.coco.anns.keys())


def collate_fn(data_batch):
    """Creates mini-batch tensors from the list of tuples produced by COCOdataset.
    This method is to generate custom merging captions.
    Args:
        data_batch: list of tuple (image, caption) where image = (3, 224, 224) and its caption
    The function will return batched images, captions, and lengths tuples.
    """
    images, captions = zip(*data_batch)
    
    # Merge images
    images = torch.cat(images, 0)

    # Merge captions
    lengths = otrch.Tensor([len(x) for x in captions]).long()
    tokens = torch.zeros(len(captions), lengths.max()).long()
    _cur_ind = 0
    for i, cap in enumerate(captions):
        end = lengths[i]
        tokens[_cur_ind, :end] = cap[:end] 
        _cur_ind += 1
    
    return images, tokens, lengths


def get_loader(mode, transform, folder, vocab, batch_size, shuffle, num_workers, 
              start_word='<start>', end_word='<end>'):
    """get torch.utils.data.DataLoader for coco dataset."""
    # COCO caption dataset
    dset = CocoDataset(mode=mode, transform=transform, folder=folder,
                       annotation=annotation, vocab=vocab, start_word, end_word)
    
    # Data loader for COCO dataset
    # Each iteration, it will produce (imgs, captions, sizes)
    data_loader = torch.utils.data.DataLoader(dataset=dset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader

def get_train_val_loaders(mode, transform, folder, vocab, batch_size, 
                          shuffle, num_workers, start_word='<start>', end_word='<end>'):
    train_loader = get_loader("train", transform, folder, vocab, batch_size, shuffle, num_workers)
    val_loader = get_loader("val", transform, folder, vocab, batch_size, shuffle, num_workers)
    
    return train_loader, val_loader