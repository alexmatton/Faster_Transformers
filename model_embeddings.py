#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts words to their embeddings.
    """
    def __init__(self, embed_size, vocab_entry):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab_entry (VocabEntry): Vocabulary object containing language
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        pad_token_idx = vocab_entry.word2id['<pad>']
        self.embed = nn.Embedding(len(vocab_entry), embed_size, padding_idx = pad_token_idx)



