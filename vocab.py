#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
vocab.py: Vocabulary Generation
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import torch
from typing import List


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing the language terms.
    """
    def __init__(self, datapath_vocab = "data/finished_files/vocab"):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """

        self.datapath_vocab = datapath_vocab

        self.word2id = dict()
        self.load_vocab()

        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    
    def load_vocab(self):
        ''' Builds the word2id dict from the vocab file '''

        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<s>'] = 1 # Start Token
        self.word2id['</s>'] = 2    # End Token
        self.word2id['<unk>'] = 3   # Unknown Token

        count = 4

        with open(self.datapath_vocab, 'r'):
            for line in self.datapath_vocab:
                word = line.split()[0]
                self.word2id[word] = count
                count += 1 

        
    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tensor, i.e. CPU or GPU
        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = self.pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)


    def pad_sents(self, sents, pad_token):
        """ Pad list of sentences according to the longest sentence in the batch.
        @param sents (list[list[str]]): list of sentences, where each sentence
                                        is represented as a list of words
        @param pad_token (str): padding token
        @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, such that
            each sentences in the batch now has equal length.
        """
        sents_padded = []

        ### YOUR CODE HERE (~6 Lines)
        max_length = max((len(sent) for sent in sents))
        for sent in sents:
            s = sent[:]+[pad_token]*(max_length-len(sent))
            sents_padded.append(s)
        ### END YOUR CODE

        return sents_padded

    # @staticmethod
    # def from_corpus(corpus, size, freq_cutoff=2):
    #     """ Given a corpus construct a Vocab Entry.
    #     @param corpus (list[str]): corpus of text produced by read_corpus function
    #     @param size (int): # of words in vocabulary
    #     @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
    #     @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
    #     """
    #     vocab_entry = VocabEntry()
    #     word_freq = Counter(chain(*corpus))
    #     valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
    #     print('number of word types: {}, number of word types w/ frequency >= {}: {}'
    #           .format(len(word_freq), freq_cutoff, len(valid_words)))
    #     top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
    #     for word in top_k_words:
    #         vocab_entry.add(word)
    #     return vocab_entry



if __name__ == '__main__':
    args = docopt(__doc__)
    vocab_entry = VocabEntry()

