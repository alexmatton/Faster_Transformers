import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import SummaryDataset
from torch.utils.data import DataLoader, RandomSampler
from fairseq.models.transformer import transformer_vaswani_wmt_en_de_big, TransformerModel
from fairseq.data import dictionary


def train(dataloader, model, criterion, optimizer):
    for idx, (text, summary) in enumerate(dataloader):
        import pdb;
        pdb.set_trace()
        y = model(text)
        loss = criterion(y, model)
        loss.backward()
        optimizer.step()


def main():
    args = parser.parse_args()

    dic = dictionary.Dictionary.load(args.vocab)
    train_dataset = SummaryDataset(os.path.join(args.data_path, 'train'))
    val_dataset = SummaryDataset(os.path.join(args.data_path, 'val'))
    test_dataset = SummaryDataset(os.path.join(args.data_path, 'test'))

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=100 if args.debug else None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, \
                                  num_workers=args.num_workers)

    transformer_vaswani_wmt_en_de_big(args)
    model = TransformerModel.build_model(args,)
    import pdb;
    pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.params(), lr=args.lr, momentum=args.momentum,
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
