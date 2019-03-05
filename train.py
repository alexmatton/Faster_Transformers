import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import SummaryDataset, SummarizationTask, collate, transformer_small
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerModel
import time
import datetime


def train_epoch(dataloader, model, criterion, optimizer, device, pad_index, save_dir, epoch, log_interval=100,
                save=True,dictionary=None):
    print("-" * 10 + "epoch " + str(epoch) + "-" * 10)
    total_loss = 0.0
    total_tokens = 0.0
    total_correct = 0.0

    last_logged_tokens = 0.0
    last_logged_loss = 0.0
    last_log_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        src_tokens = batch['net_input']['src_tokens'].to(device)
        src_lengths = batch['net_input']['src_lengths'].to(device)
        prev_output_tokens = batch['net_input']['prev_output_tokens'].to(device)
        target = batch['target'].to(device)
        target_mask = (target != pad_index).float()
        total_tokens += batch['ntokens']

        output = model(src_tokens, src_lengths, prev_output_tokens)
        preds = torch.argmax(output[0], dim=-1)


        # TODO continue here
        total_correct += ((preds == target).float() * target_mask).sum().item()
        # total_correct += ((preds == target)).sum().item()

        loss = (criterion(output[0].view(-1, output[0].size(-1)), target.view(-1)) * target_mask.view(-1)).sum()
        total_loss += loss.item()
        loss = loss / batch['ntokens']
        loss.backward()
        optimizer.step()
        # TODO maybe lr_scheduler.step()
        if epoch >200:
            import pdb;
            pdb.set_trace()

        if (batch_idx % log_interval == 0) and (batch_idx > 0):
            loss_to_log = (total_loss - last_logged_loss) / (total_tokens - last_logged_tokens)
            speed_to_log = log_interval / (time.time() - last_log_time)
            last_log_time = time.time()
            last_logged_loss = total_loss
            last_logged_tokens = total_tokens

            print("train | epoch {} | batch {}/{} | loss {:.3f}| {:.3f} batch/s".format(epoch, batch_idx,
                                                                                        len(dataloader),
                                                                                        loss_to_log,
                                                                                        speed_to_log))
        # TODO validation loss

    print("EPOCH {} | train loss {:.3f} | train acc {:.7f}".format(epoch, total_loss / total_tokens,
                                                                   total_correct / total_tokens))

    if save:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_transformer_epoch_{}.pt'.format(epoch)))


def main():
    # TODO max article size
    # data setup

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dictionary = Dictionary.load(args.vocab_path)
    train_dataset = SummaryDataset(os.path.join(args.data_path, 'train'), dictionary=dictionary,
                                   max_article_size=args.max_source_positions,
                                   max_summary_size=args.max_target_positions, max_elements=1 if args.debug else None)

    # TODO maybe change the sampler to group texts of similar lengths

    train_sampler = RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, \
                                  num_workers=args.num_workers,
                                  collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                     dictionary.eos_index))

    summarization_task = SummarizationTask(args, dictionary)
    transformer_small(args)
    model = TransformerModel.build_model(args, summarization_task).to(args.device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.flag == "":
        args.flag = 'train_transformer_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
    if not os.path.isdir(os.path.join(args.save_dir, args.flag)):
        os.makedirs(os.path.join(args.save_dir, args.flag))
    for epoch in range(args.n_epochs):
        train_epoch(train_dataloader, model, criterion, optimizer, args.device, dictionary.pad_index,
                    save_dir=os.path.join(args.save_dir, args.flag), epoch=epoch, save=args.save,dictionary=dictionary)


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="datasets/cnn_debug")
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--save", type=int, default=0)
parser.add_argument("--flag", type=str, default="")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default='sgd')

parser.add_argument("--momentum", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--log_interval", type=str, help='log every k batch', default=100)
parser.add_argument("--max_source_positions", type=int, default=400)
parser.add_argument("--max_target_positions", type=int, default=100)

parser.add_argument("--seed", type=int, default=1111)

if __name__ == "__main__":
    main()
