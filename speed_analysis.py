import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from data import DummySummaryDataset, SummarizationTask, collate
from models import transformer_small, light_conv_small
from torch.utils.data import DataLoader, RandomSampler
from fairseq.data import Dictionary
from fairseq.models import transformer, lightconv, lstm, transformer_conv

import time


def run_pass(model, dataloader, criterion, optimizer, pad_index, device, forward=True, log_interval=50):
    model.train()
    total_sents = 0
    total_time = 0
    total_time_encoder = 0
    total_time_decoder = 0

    last_logged_sents = 0
    last_log_total_time = 0
    last_log_total_time_encoder = 0
    last_log_total_time_decoder = 0

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

            time_before_model = time.process_time()
            output_encoder = model.encoder(src_tokens, src_lengths)
            time_after_encoder = time.process_time()
            # print("time encod", time_after_encoder-time_before_model)

            output = model.decoder(prev_output_tokens, output_encoder)

            # output = model(src_tokens, src_lengths, prev_output_tokens)
            if not forward:
                loss = (criterion(output[0].view(-1, output[0].size(-1)), target.view(-1)) * target_mask.view(-1)).sum()
                loss.backward()
                optimizer.step()

            current_time = time.process_time()

            # print("time decod", current_time - time_after_encoder)

            total_time += current_time - time_before_model
            total_time_encoder += time_after_encoder - time_before_model
            total_time_decoder += current_time - time_after_encoder

            if (batch_idx % log_interval == 0) and (batch_idx > 0):
                speed_to_log = (total_sents - last_logged_sents) / (total_time - last_log_total_time)
                speed_to_log_encoder = (total_sents - last_logged_sents) / \
                                        (total_time_encoder - last_log_total_time_encoder)
                speed_to_log_decoder = (total_sents - last_logged_sents) / \
                                        (total_time_decoder - last_log_total_time_decoder)
                last_log_total_time = total_time
                last_log_total_time_encoder = total_time_encoder
                last_log_total_time_decoder = total_time_decoder
                last_logged_sents = total_sents

                print("{} | {:.3f} sent/s".format("forward only" if forward else "forward + backward", speed_to_log))
                print("{} | {:.3f} sent/s".format("encoder only", speed_to_log_encoder))
                print("{} | {:.3f} sent/s".format("decoder only", speed_to_log_decoder))
                print()
    return total_sents, total_time, total_time_decoder, total_time_decoder


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

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-5, momentum=0.0,
                                weight_decay=0.0)
    total_speeds = []
    encoder_speeds = []
    decoder_speeds = []
    len_articles = [a for a in range(args.min_len_article, args.max_len_article, args.len_step)]
    for len_article in len_articles:
        dataset = DummySummaryDataset(args.total_sents, len_article, args.len_summaries, dictionary)

        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                   dictionary.eos_index))

        print("MODEL {} ARTICLE LEN {} SUMMARY LEN {}".format(args.model, len_article, args.len_summaries))

        total_sents, total_time, total_time_encoder, total_time_decoder = \
            run_pass(model, dataloader, criterion, optimizer, dictionary.pad_index, args.device, args.forward)
        print("MODEL {} SPEED {}".format(args.model, total_sents / total_time))
        print()
        total_speeds.append(total_sents / total_time)
        encoder_speeds.append(total_sents / total_time_encoder)
        decoder_speeds.append(total_sents / total_time_decoder)

    if not os.path.isdir(os.path.join(args.save_dir, args.model)):
        os.makedirs(os.path.join(args.save_dir, args.model))
    
    np.save(os.path.join(args.save_dir, args.model, 'len_articles.npy'), np.array(len_articles))
    np.save(os.path.join(args.save_dir, args.model, 'total_speeds.npy'), np.array(total_speeds))
    np.save(os.path.join(args.save_dir, args.model, 'encoder_speeds.npy'), np.array(encoder_speeds))
    np.save(os.path.join(args.save_dir, args.model, 'decoder_speeds.npy'), np.array(decoder_speeds))


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--save_dir", type=str, default="checkpoints/speed_analysis")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--model",type=str, choices=['transformer', 'lstm', 'lightconv', 'localtransformer', 
                        'transformer_conv'],default='transformer')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--kernel_size", type=int, default=10)  # for LocalTransformer or transformer_conv
parser.add_argument("--deconv", action='store_true')  # for LocalTransformer or transformer_conv

parser.add_argument("--min_len_article", type=int, default=1000)
parser.add_argument("--max_len_article", type=int, default=2000)
parser.add_argument("--len_step", type=int, default=50)
parser.add_argument("--len_summaries", type=int, default=100)
parser.add_argument("--total_sents", type=int, default=2000)
parser.add_argument("--forward", type=int, default=1)

if __name__ == "__main__":
    main()
