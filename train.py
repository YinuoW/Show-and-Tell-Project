import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


def train_func(model_path, crop_img_size, vocab_sen_path,
         image_path, caption_img_path, embed_layer_size, hidden_layer_size,layer_num,
         epoch_num, batch_size, learning_rate):
    '''
    this is used to train the model
    :param model_path: model path
    :param crop_img_size: the size of cropped picture
    :param vocab_sen_path: vocab path
    :param image_path: image path
    :param caption_img_path: caption picture path
    :param embed_layer_size: size of embed layer
    :param hidden_layer_size: size of hidden layer
    :param layer_num: number of layers
    :param epoch_num: number of epochs
    :param batch_size: size of batch
    :param learning_rate: learning rate
    :return:
    '''
    # check the path exist or not
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    transform = transforms.Compose([
        transforms.RandomCrop(crop_img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with open(vocab_sen_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_path, caption_img_path, vocab,
                             transform, batch_size,
                             shuffle=True, num_workers=2)

    encoder = EncoderCNN(embed_layer_size)
    decoder = DecoderRNN(embed_layer_size, hidden_layer_size, len(vocab), layer_num)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train
    total_step = len(data_loader)
    for epoch in range(epoch_num):
        for i, (images, captions, lengths) in enumerate(data_loader):

            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d]'
                      % (epoch, epoch_num, i, total_step))

            if (i + 1) % 100 == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(model_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    model_path = './models/'
    crop_img_size = 224
    vocab_sen_path = './data/vocab.pkl'
    image_path = './data/resized2014'
    caption_img_path = './data/annotations/captions_train2014.json'
    embed_layer_size = 256
    hidden_layer_size = 512
    layer_num = 1
    epoch_num = 5
    batch_size = 128
    learning_rate = 0.001
    train_func(model_path, crop_img_size, vocab_sen_path, image_path, caption_img_path
               , embed_layer_size, hidden_layer_size,layer_num,
               epoch_num, batch_size, learning_rate)
