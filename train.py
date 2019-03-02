import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import SummaryDataset, SummarizationTask, collate
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models.transformer import transformer_vaswani_wmt_en_de_big, TransformerModel


def train(dataloader, model, criterion, optimizer):
    '''

    :param dataloader:
    :param model:
    :param criterion:
    :param optimizer:
    :return:
    '''
    for batch_idx, batch in enumerate(dataloader):
        #
        # TODO continue here
        import pdb;
        pdb.set_trace()




def main():
    # data setup

    args = parser.parse_args()
    dictionary = Dictionary.load(args.vocab_path)
    train_dataset = SummaryDataset(os.path.join(args.data_path, 'train'), dictionary=dictionary)
    val_dataset = SummaryDataset(os.path.join(args.data_path, 'val'), dictionary=dictionary)
    test_dataset = SummaryDataset(os.path.join(args.data_path, 'test'), dictionary=dictionary)

    # TODO maybe change the sampler to group texts of similar lengths

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=100 if args.debug else None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, \
                                  num_workers=args.num_workers,
                                  collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                     dictionary.eos_index))

    summarization_task = SummarizationTask(args, dictionary)
    transformer_vaswani_wmt_en_de_big(args)
    model = TransformerModel.build_model(args, summarization_task)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train(train_dataloader, model, criterion, optimizer)


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="datasets/cnn_debug")
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)

if __name__ == "__main__":
    main()
