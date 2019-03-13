import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import SummaryDataset, SummarizationTask, collate
from models import transformer_small, light_conv_small
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models import transformer
from fairseq.models import lstm
from fairseq.models import lightconv
from LocalTransformerModel import LocalTransformerModel
import time
import datetime
import tensorboardX


def validate(dataloader, model, criterion, device, pad_index, epoch):
    model.eval()
    print("-" * 10 + "evaluation - epoch " + str(epoch) + "-" * 10)
    total_loss = 0.0
    total_tokens = 0.0
    total_correct = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src_tokens = batch['net_input']['src_tokens'].to(device)
            src_lengths = batch['net_input']['src_lengths'].to(device)
            prev_output_tokens = batch['net_input']['prev_output_tokens'].to(device)
            target = batch['target'].to(device)
            target_mask = (target != pad_index).float()
            total_tokens += batch['ntokens']

            output = model(src_tokens, src_lengths, prev_output_tokens)
            preds = torch.argmax(output[0], dim=-1)

            total_correct += ((preds == target).float() * target_mask).sum().item()

            loss = (criterion(output[0].view(-1, output[0].size(-1)), target.view(-1)) * target_mask.view(-1)).sum()
            total_loss += loss.item()

            # TODO validation loss
    val_loss = total_loss / total_tokens
    print("EPOCH {} | val loss {:.3f} | val acc {:.7f}".format(epoch, total_loss / total_tokens,
                                                               total_correct / total_tokens))
    return val_loss


def train(dataloaders, model, criterion, optimizer, lr_scheduler, device, pad_index, save_dir, n_epochs,
          log_interval=100,
          save=True, debug = False):
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        lr_scheduler.step(epoch)
        model.train()
        print("-" * 10 + "epoch " + str(epoch) + "-" * 10)
        total_loss = 0.0
        total_tokens = 0.0
        total_correct = 0.0

        last_logged_tokens = 0.0
        last_logged_loss = 0.0
        last_log_time = time.time()

        print("Learning rate:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        for batch_idx, batch in enumerate(dataloaders['train']):

            src_tokens = batch['net_input']['src_tokens'].to(device)
            src_lengths = batch['net_input']['src_lengths'].to(device)
            prev_output_tokens = batch['net_input']['prev_output_tokens'].to(device)
            target = batch['target'].to(device)
            target_mask = (target != pad_index).float()
            total_tokens += batch['ntokens']
            output = model(src_tokens, src_lengths, prev_output_tokens)
            preds = torch.argmax(output[0], dim=-1)
            total_correct += ((preds == target).float() * target_mask).sum().item()

            loss = (criterion(output[0].view(-1, output[0].size(-1)), target.view(-1)) * target_mask.view(-1)).sum()
            total_loss += loss.item()
            loss = loss / target_mask.sum()
            loss.backward()
            optimizer.step()

            if (batch_idx % log_interval == 0) and (batch_idx > 0):
                loss_to_log = (total_loss - last_logged_loss) / (total_tokens - last_logged_tokens)
                speed_to_log = log_interval / (time.time() - last_log_time)
                last_log_time = time.time()
                last_logged_loss = total_loss
                last_logged_tokens = total_tokens

                print("train | epoch {} | batch {}/{} | loss {:.3f}| {:.3f} batch/s".format(epoch, batch_idx,
                                                                                            len(dataloaders['train']),
                                                                                            loss_to_log,
                                                                                            speed_to_log))
        val_loss = validate(dataloaders['val'], model, criterion, device, pad_index, epoch)

        if save and debug and epoch % 200 == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'debug_model.pt'))

        if (val_loss < best_val_loss) and save and not debug:
            print("saved model, epoch {}, val loss {:.3f}".format(epoch, val_loss))
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        print("EPOCH {} | train loss {:.3f} | train acc {:.7f}".format(epoch, total_loss / total_tokens,
                                                                       total_correct / total_tokens))


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dictionary = Dictionary.load(args.vocab_path)
    train_dataset = SummaryDataset(os.path.join(args.data_path, 'train'), dictionary=dictionary,
                                   max_article_size=args.max_source_positions,
                                   max_summary_size=args.max_target_positions, max_elements=20 if args.debug else None)
    val_dataset = SummaryDataset(os.path.join(args.data_path, 'val'), dictionary=dictionary,
                                 max_article_size=args.max_source_positions,
                                 max_summary_size=args.max_target_positions, max_elements=20 if args.debug else None)

    # TODO maybe change the sampler to group texts of similar lengths

    train_sampler = RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, \
                                  num_workers=args.num_workers,
                                  collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                     dictionary.eos_index))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                   dictionary.eos_index))

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    summarization_task = SummarizationTask(args, dictionary)
    if args.model == 'transformer':
        transformer_small(args)
        model = transformer.TransformerModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'lstm':
        lstm.base_architecture(args)
        args.criterion = None
        model = lstm.LSTMModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'lightconv':
        args.encoder_conv_type = 'lightweight'
        args.decoder_conv_type = 'lightweight'
        args.weight_softmax = True
        light_conv_small(args)
        model = lightconv.LightConvModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'localtransformer':
        transformer_small(args)
        model = LocalTransformerModel.build_model(args, summarization_task).to(args.device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exponential_decay)

    if args.flag == "":
        args.flag = 'train_transformer_{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    if not os.path.isdir(os.path.join(args.save_dir, args.flag)):
        os.makedirs(os.path.join(args.save_dir, args.flag))

    print("Launching training with: \noptimizer: {}\n lr: {}\n \
exponential_decay: {}\n momentum: {}\n weight_decay: {}\n batch_size: {}\n"
            .format(args.optimizer, args.lr, args.exponential_decay, 
            args.momentum, args.weight_decay, args.batch_size))

    train(dataloaders, model, criterion, optimizer, lr_scheduler, args.device, dictionary.pad_index,
          save_dir=os.path.join(args.save_dir, args.flag), n_epochs=args.n_epochs, save=args.save, debug = args.debug)


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="datasets/cnn_full")
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--save", type=int, default=0)
parser.add_argument("--flag", type=str, default="") #for name of saved model, if "" then takes into account the date
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument('--exponential_decay', type=float, default=0.9)
parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default='sgd')
parser.add_argument("--kernel_size", type=int, default=10) 

parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--log_interval", type=str, help='log every k batch', default=100)
parser.add_argument("--model", type=str, choices=['transformer', 'lstm', 'lightconv', 'localtransformer'], 
                default='transformer')
parser.add_argument("--max_source_positions", type=int, default=400)
parser.add_argument("--max_target_positions", type=int, default=100)

parser.add_argument("--seed", type=int, default=1111)

if __name__ == "__main__":
    main()
