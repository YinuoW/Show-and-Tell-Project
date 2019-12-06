import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        # disable grad for resnet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        # batchNorm for CNN
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_len=20):
        '''
        :param embed_size: dimension of word embedding vectors
        :param hidden_size: dimension of lstm hidden states
        :param vocab_size: the size of vocabulary wrapper
        :param num_layers: number of layers in lstm
        :param max_seq_len: maximum sentence length
        '''
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        # load LSTM and set hyper-parameters
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.max_seq_len = max_seq_len
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states):
        '''
        :param features: extracted image features from CNN
        :param states: lstm state
        '''
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_len):                                     
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.linear(hiddens.squeeze(1))            
            # Get the index of the vocabulary that is of highest probability
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            # increase the dimension from 2 to 3
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.cat(sampled_ids, 0)
        return sampled_ids.squeeze()
