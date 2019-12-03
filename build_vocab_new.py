import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocab(object):
    def __init__(self):
        self.word_idx_dict = {}
        self.idx_word_dict = {}
        self.count = 0
        self.append('<pad>')
        self.append('<start>')
        self.append('<end>')
        self.append('<unk>')

    def append(self, word):
        is_contained = word in self.word_idx_dict
        if (is_contained == False):
            self.word_idx_dict[word] = self.count
            self.idx_word_dict[self.count] = word
            self.count += 1

    def __call__(self, word):
        if word in self.word_idx_dict:
            return self.word_idx_dict[word]
        else:
            return self.word_idx_dict['<unk>']

    def __len__(self):
        return len(self.word_idx_dict)


def build_vocab(json, threshold):
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
    coco = COCO(json)
    counts = Counter()
    idxs = coco.anns.keys()
    idxs = list(idxs)
    return iter_count(idxs, coco, counts)


def iter_count(idxs, coco, counts):
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