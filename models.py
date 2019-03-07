from fairseq.models import transformer
from fairseq.models import lightconv


def transformer_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    transformer.base_architecture(args)


def light_conv_small(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.encoder_conv_dim = getattr(args, 'encoder_conv_dim', args.encoder_embed_dim)
    args.decoder_conv_dim = getattr(args, 'decoder_conv_dim', args.decoder_embed_dim)

    args.encoder_kernel_size_list = getattr(args, 'encoder_kernel_size_list', [3, 15, 31])
    args.decoder_kernel_size_list = getattr(args, 'decoder_kernel_size_list', [3, 15, 31])
    if len(args.encoder_kernel_size_list) == 1:
        args.encoder_kernel_size_list = args.encoder_kernel_size_list * args.encoder_layers
    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.decoder_layers
    assert len(args.encoder_kernel_size_list) == args.encoder_layers, "encoder_kernel_size_list doesn't match encoder_layers"
    assert len(args.decoder_kernel_size_list) == args.decoder_layers, "decoder_kernel_size_list doesn't match decoder_layers"
    args.encoder_glu = getattr(args, 'encoder_glu', True)
    args.decoder_glu = getattr(args, 'decoder_glu', True)
    args.input_dropout = getattr(args, 'input_dropout', 0.1)
    args.weight_dropout = getattr(args, 'weight_dropout', args.attention_dropout)