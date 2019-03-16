import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import DummySummaryDataset, SummarizationTask, collate
from models import transformer_small, light_conv_small
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models import transformer
from fairseq.models import lstm
from fairseq.models import lightconv
from LocalTransformerModel import LocalTransformerModel
import time


def run_pass(model, dataloader, criterion, optimizer, pad_index, device, forward=True, log_interval=100):
    model.train()
    a = time.time()
    last_log_time = a
    total_sents = 0
    last_logged_sents = 0
    with torch.set_grad_enabled(not forward):
        for batch_idx, batch in enumerate(dataloader):
            src_tokens = batch['net_input']['src_tokens'].to(device)
            src_lengths = batch['net_input']['src_lengths'].to(device)
            prev_output_tokens = batch['net_input']['prev_output_tokens'].to(device)
            target = batch['target'].to(device)
            target_mask = (target != pad_index).float()
            total_sents += len(src_lengths)
            if batch_idx == 0:
                print(src_tokens.shape)
            output = model(src_tokens, src_lengths, prev_output_tokens)
            if not forward:
                loss = (criterion(output[0].view(-1, output[0].size(-1)), target.view(-1)) * target_mask.view(-1)).sum()
                loss.backward()
                optimizer.step()

            if (batch_idx % log_interval == 0) and (batch_idx > 0):
                speed_to_log = (total_sents - last_logged_sents) / (time.time() - last_log_time)
                last_log_time = time.time()
                last_logged_sents = total_sents

                print("{} | {:.3f} sent/s".format("forward only" if forward else "forward + backward", speed_to_log))
    total_time = time.time() - a
    return total_sents, total_time


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dictionary = Dictionary.load(args.vocab_path)
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
    else:
        assert args.model == 'localtransformer'
        transformer_small(args)
        model = LocalTransformerModel.build_model(args, summarization_task).to(args.device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-5, momentum=0.0,
                                weight_decay=0.0)
    speeds = []
    len_articles = [a for a in range(args.len_step, args.max_len_article, args.len_step)]
    for len_article in len_articles:
        dataset = DummySummaryDataset(args.total_sents, len_article, args.len_summaries, dictionary)

        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                   dictionary.eos_index))

        print("MODEL {} ARTICLE LEN {} SUMMARY LEN {}".format(args.model, len_article, args.len_summaries))

        total_sents, total_time = run_pass(model, dataloader, criterion, optimizer, dictionary.pad_index, args.device,
                                           args.forward)
        print("MODEL {} SPEED {}".format(args.model, total_sents / total_time))
        speeds.append(total_sents / total_time)
    if not os.path.isdir(os.path.join(args.save_dir, args.model)):
        os.makedirs(os.path.join(args.save_dir, args.model))
    np.save(os.path.join(args.save_dir, args.model, 'len_articles.npy'), np.array(len_articles))
    np.save(os.path.join(args.save_dir, args.model, 'speeds.npy'), np.array(speeds))


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--save_dir", type=str, default="checkpoints/speed_analysis")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--model", type=str, choices=['transformer', 'lstm', 'lightconv', 'localtransformer'],
                    default='transformer')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--kernel_size", type=int, default=15)

parser.add_argument("--max_len_article", type=int, default=4000)
parser.add_argument("--len_step", type=int, default=50)
parser.add_argument("--len_summaries", type=int, default=50)
parser.add_argument("--total_sents", type=int, default=4000)
parser.add_argument("--forward", type=int, default=1)

if __name__ == "__main__":
    main()
