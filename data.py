import numpy as np
import os
import struct
from torch.utils.data import Dataset
from tensorflow.core.example import example_pb2
from fairseq.tasks import FairseqTask
from fairseq.data import data_utils, Dictionary
import torch
from fairseq.models.transformer import base_architecture


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    # taken from https://github.com/pytorch/fairseq/blob/master/fairseq/data/language_pair_dataset.py
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('article', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['article'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = merge('summary', left_pad=left_pad_target)
    target = target.index_select(0, sort_order)
    ntokens = sum(len(s['summary']) for s in samples)

    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'summary',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class SummaryDataset(Dataset):
    '''
    '''

    def __init__(self, datapath, dictionary, max_article_size=10000, max_summary_size=10000, max_elements=None):
        self.datapath = datapath
        self.dictionary = dictionary

        self.max_article_size = max_article_size
        self.max_summary_size = max_summary_size
        self.max_elements= max_elements

        self.articles = []
        self.summaries = []
        self.articles_len = []
        self.summaries_len = []

        self.preprocess()

    def preprocess(self):
        ''' Import the dataset from the binary files.
        Code taken and adapted from: https://github.com/abisee/pointer-generator/blob/master/data.py '''

        filelist = os.listdir(self.datapath)  # get the list of datafiles
        filelist = [os.path.join(self.datapath, f) for f in filelist]
        filelist.sort()
        assert filelist, ('Error: Empty filelist at %s' %
                          self.datapath)  # check filelist isn't empty

        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break  # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack(
                    '%ds' % str_len, reader.read(str_len))[0]
                tf_example = example_pb2.Example.FromString(example_str)

                examples = []
                for key in tf_example.features.feature:
                    examples.append(
                        '%s' % (tf_example.features.feature[key].bytes_list.value[0]))

                examples[0] = examples[0][2:-1].split()[:self.max_article_size]
                examples[1] = [w for w in examples[1][2:-1].split()
                               if (w != self.dictionary.eos_word and w != '<s>')]
                examples[1] = examples[1][:self.max_summary_size]
                self.articles.append(examples[0])
                self.summaries.append(examples[1])
        self.articles_len = np.array([len(a) for a in self.articles], dtype='long')
        self.summaries_len = np.array([len(s) for s in self.summaries], dtype='long')

        if self.max_elements is not None:
            self.articles_len = self.articles_len[:self.max_elements]
            self.summaries_len = self.summaries_len[:self.max_elements]
            self.articles = self.articles[:self.max_elements]
            self.summaries = self.summaries[:self.max_elements]

    def tokenize(self, text):
        return torch.LongTensor([self.dictionary.index(sym) for sym in text] + [self.dictionary.eos_index])

    def ordered_indices(self, shuffle=False):
        """Return an ordered list of indices. Batches will be constructed based
                on this order."""
        if shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        indices = indices[np.argsort(self.summaries_len[indices], kind='mergesort')]
        return indices[np.argsort(self.articles_len[indices], kind='mergesort')]

    def __getitem__(self, index):
        article, summary = self.articles[index], self.summaries[index]
        article, summary = self.tokenize(article), self.tokenize(summary)
        item = {'id': index, 'article': article, 'summary': summary}
        return item

    def __len__(self):
        return len(self.articles)


##TODO maybe integrate totally into FairSeq for later use

class SummarizationTask(FairseqTask):

    ##TODO finish this and design in in the same way as  https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/language_modeling.py
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


def transformer_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


if __name__ == "__main__":
    dataset = SummaryDataset(datapath='datasets/cnn_debug/train', dictionary=Dictionary.load('datasets/vocab'))
    print(dataset[3])
