import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocab(object):
    '''
    This is the calss of vocabulary
    wrod_idx_dict: a dictionary where word is key and index is value
    idx_word_dict: a dictionary where index is key and word is value
    count: record the current index of word
    '''
    def __init__(self):
        self.word_idx_dict = {}
        self.idx_word_dict = {}
        self.count = 0
        self.append('<pad>')
        self.append('<start>')
        self.append('<end>')
        self.append('<unk>')

    def append(self, word):
        # append the word to the Vocab object
        is_contained = word in self.word_idx_dict
        if (is_contained == False):
            self.word_idx_dict[word] = self.count
            self.idx_word_dict[self.count] = word
            self.count += 1

    def __call__(self, word):
        # get the index of a word
        if word in self.word_idx_dict:
            return self.word_idx_dict[word]
        else:
            return self.word_idx_dict['<unk>']

    def __len__(self):
        # get the length of the object
        return len(self.word_idx_dict)


def build_vocab(json, threshold):
    '''
    This function takes a json file and threshold number
    and returns the corresponding vocabulary object
    '''
    counts = count_vocab(json)
    words = list()
    for word, count in counts.items():
        if count >= threshold:
            words.append(word)
    vocab = Vocab()
    for word in words:
        vocab.append(word)
    return vocab


def count_vocab(json):
    # helper function to get counts for each word
    coco = COCO(json)
    counts = Counter()
    idxs = coco.anns.keys()
    idxs = list(idxs)
    return iter_count(idxs, coco, counts)


def iter_count(idxs, coco, counts):
    # helper function to get counts for each word in iterative way
    for i in range(len(idxs)):
        idx = idxs[i]
        caption = str(coco.anns[idx]['caption'])
        caption = caption.lower()
        tokens = nltk.tokenize.word_tokenize(caption)
        counts.update(tokens)
    return counts

def main():
    caption_dir = './data/annotations/captions_train2014.json'
    vocab_dir = './data/vocab2.pkl'
    threshold = 4
    vocab = build_vocab(caption_dir, threshold=threshold)
    with open(vocab_dir, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        
if __name__ == '__main__':
    main()
