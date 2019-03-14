import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from collections import defaultdict
from data import SummaryDataset, SummarizationTask, collate
from models import transformer_small, light_conv_small
from torch.utils.data import DataLoader, SequentialSampler

from fairseq.data import Dictionary, TruncatedDictionary
from fairseq.models import transformer
from fairseq.models import lstm
from fairseq.models import lightconv
import compute_rouge

from fairseq.sequence_generator import SequenceGenerator


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dictionary = TruncatedDictionary(Dictionary.load(args.vocab_path), args.max_vocab_size)

    test_dataset = SummaryDataset(os.path.join(args.data_path, 'test'), dictionary=dictionary,
                                  max_article_size=args.max_source_positions,
                                  max_summary_size=args.max_target_positions,
                                  max_elements=10 if args.debug else None)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler, \
                                 num_workers=args.num_workers,
                                 collate_fn=lambda samples: collate(samples, dictionary.pad_index,
                                                                    dictionary.eos_index))

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

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    generator = SequenceGenerator([model], dictionary, beam_size=args.beam_size,
                                  maxlen=args.max_target_positions)

    avg_rouge_score = defaultdict(float)

    for batch_idx, batch in enumerate(test_dataloader):
        src_tokens = batch['net_input']['src_tokens'].to(args.device)
        src_lengths = batch['net_input']['src_lengths'].to(args.device)

        references = batch['target']
        references = [remove_special_tokens(ref, dictionary) for ref in references]
        references = [dictionary.string(ref) for ref in references]

        encoder_input = {'src_tokens': src_tokens, 'src_lengths': src_lengths}
        print("generation")
        hypos = generator.generate(encoder_input)
        print("Done")

        hypotheses = [hypo[0]['tokens'] for hypo in hypos]
        assert len(hypotheses) == src_tokens.size()[0]  # = size of the batch
        hypotheses = [remove_special_tokens(hypo, dictionary) for hypo in hypotheses]
        hypotheses = [dictionary.string(hyp) for hyp in hypotheses]

        if args.verbose:
            print("\nComparison references/hypotheses:")
            for ref, hypo in zip(references, hypotheses):
                print(ref)
                print(hypo)
                print()

        avg_rouge_score_batch = compute_rouge.compute_score(references, hypotheses)
        print("rouge for this batch:", avg_rouge_score_batch)

        compute_rouge.update(avg_rouge_score, batch_idx * args.batch_size,
                             avg_rouge_score_batch, len(hypotheses))

    return avg_rouge_score


def remove_special_tokens(token_tensor, dictionary):
    return [token.item() for token in token_tensor if (token.item() != dictionary.pad_index
                                                       and token != dictionary.eos_index)]


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="datasets/cnn_full")
parser.add_argument("--vocab_path", type=str, default="datasets/vocab")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--kernel_size", type=int, default=10)  # for LocalTransformer
parser.add_argument("--max_source_positions", type=int, default=400)
parser.add_argument("--max_target_positions", type=int, default=100)
parser.add_argument("--max_vocab_size", type=int, default=20000)
parser.add_argument("--beam_size", type=int, default=4)
parser.add_argument("--model", type=str, choices=['transformer', 'lstm', 'lightconv'], default='transformer')

#for local transformer only, choose whether local attention should be used in the decoder self-attention layer
parser.add_argument("--use_local_decoder", action='store_true')

parser.add_argument("--seed", type=int, default=1111)

if __name__ == "__main__":
    main()
