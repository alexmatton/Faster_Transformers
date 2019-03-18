import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import DummySummaryDataset, SummarizationTask, collate
from models import transformer_small, light_conv_small
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models import transformer, lightconv, lstm, transformer_conv, transformer_mc

import time


def run_pass(model, dataloader, pad_index, device, encoder_only = False, log_interval=100):

    model.train()
    total_sents = 0
    last_logged_sents = 0

    with torch.no_grad():

        last_log_time = time.time()
        time_start = last_log_time

        for batch_idx, batch in enumerate(dataloader):
            
            src_tokens = batch['net_input']['src_tokens'].to(device)
            src_lengths = batch['net_input']['src_lengths'].to(device)
            prev_output_tokens = batch['net_input']['prev_output_tokens'].to(device)
            total_sents += len(src_lengths)
            if batch_idx == 0:
                print(src_tokens.shape)

            if encoder_only:
                output = model.encoder(src_tokens, src_lengths)
            else:
                output = model(src_tokens, src_lengths, prev_output_tokens)

            if (batch_idx % log_interval == 0) and (batch_idx > 0):
                current_time = time.time()
                speed_to_log = (total_sents - last_logged_sents) / (current_time - last_log_time)
                last_log_time = current_time
                last_logged_sents = total_sents

                print("{} | {:.3f} sent/s or {:.5f} s/sent".format("whole model" if not encoder_only
                                                                else "decoder only",
                                                                speed_to_log, 1/speed_to_log))
                
    return total_sents, time.time()-time_start


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dictionary = Dictionary.load(args.vocab_path)
    summarization_task = SummarizationTask(args, dictionary)

    if args.model == 'transformer':
        args.local_transformer = False
        # transformer.base_architecture(args)
        transformer.transformer_small(args)
        model = transformer.TransformerModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'lstm':
        lstm.base_architecture(args)
        args.criterion = None
        model = lstm.LSTMModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'lightconv':
        args.encoder_conv_type = 'lightweight'
        args.decoder_conv_type = 'lightweight'
        args.weight_softmax = True
        lightconv.lightconv_small(args)
        model = lightconv.LightConvModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'localtransformer':
        args.local_transformer = True
        # transformer.base_architecture(args)
        transformer.transformer_small(args)
        model = transformer.TransformerModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'transformer_conv':
        # args.local_transformer = True
        # transformer.base_architecture(args)
        transformer_conv.transformer_conv_small(args)
        model = transformer_conv.TransformerConvModel.build_model(args, summarization_task).to(args.device)
    elif args.model == 'transformer_mc':
        # args.local_transformer = True
        # transformer.base_architecture(args)
        transformer_mc.transformer_mc_small(args)
        model = transformer_mc.TransformerMCModel.build_model(args, summarization_task).to(args.device)

    total_speeds = []
    len_articles = [a for a in range(args.min_len_article, args.max_len_article+args.len_step, args.len_step)]
    for len_article in len_articles:
        dataset = DummySummaryDataset(args.total_sents, len_article, args.len_summaries, dictionary)

        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                   dictionary.eos_index))

        print("MODEL {} ARTICLE LEN {} SUMMARY LEN {}".format(args.model, len_article, args.len_summaries))

        total_sents, total_time = \
            run_pass(model, dataloader, dictionary.pad_index, args.device, args.encoder_only)
        print("MODEL {} SPEED {} T/SENT {}".format(args.model, total_sents / total_time, total_time / total_sents))
        print()
        total_speeds.append(total_sents / total_time)

    if not os.path.isdir(os.path.join(args.save_dir, args.model)):
        os.makedirs(os.path.join(args.save_dir, args.model))
    
    filename = "full_model_" if not args.encoder_only else "encoder_only_"
    filename += str(args.min_len_article) + "_" + str(args.max_len_article) + "_"
    filename += 'total_speeds.npy'
    np.save(os.path.join(args.save_dir, args.model, filename), np.array(list(zip(len_articles, total_speeds))))


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--save_dir", type=str, default="checkpoints/speed_analysis")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--model",type=str, choices=['transformer', 'lstm', 'lightconv', 'localtransformer', 
                        'transformer_conv', 'transformer_mc'],default='transformer')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--kernel_size", type=int, default=4)  # for LocalTransformer or transformer_conv
parser.add_argument("--deconv", action='store_true')  # for LocalTransformer or transformer_conv

parser.add_argument("--min_len_article", type=int, default=1000)
parser.add_argument("--max_len_article", type=int, default=2000)
parser.add_argument("--len_step", type=int, default=50)
parser.add_argument("--len_summaries", type=int, default=100)
parser.add_argument("--total_sents", type=int, default=2000)
parser.add_argument("--forward", type=int, default=1)

parser.add_argument("--encoder_only", action='store_true')

if __name__ == "__main__":
    main()
