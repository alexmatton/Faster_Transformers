import os
import argparse
import struct
from tensorflow.core.example import example_pb2


def parse_bin(datapath,max_input_tokens,max_output_tokens):
    articles = []
    summaries = []

    filelist = os.listdir(datapath)  # get the list of datafiles
    filelist = [os.path.join(datapath, f) for f in filelist]
    filelist.sort()
    assert filelist, ('Error: Empty filelist at %s' %
                      datapath)  # check filelist isn't empty

    max_input_tokens = int(max_input_tokens)
    max_output_tokens = int(max_output_tokens)

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

            articles.append(examples[0][2:-1])
            summaries.append(examples[1][2:-1])
    articles = [clean(art,max_input_tokens) for art in articles]
    summaries = [clean(sum,max_output_tokens) for sum in summaries]

    articles_2=[]
    summaries_2 = []
    for i in range(len(articles)):
        if len(articles[i])>5 and len(summaries[i])>5:
            articles_2.append(articles[i])
            summaries_2.append(summaries[i])


    return articles_2, summaries_2


def clean(text,max_tokens):
    text = text.split(" ")
    text = [s for s in text if s not in ["<s>", "</s>", "\n"]]
    if len(text)==0:
        print("empty article")
    text = text[:max_tokens]
    text = " ".join(text) + "\n"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='../datasets/cnn_full')
    parser.add_argument('--target_dir', default='../datasets/cnn_full_txt')
    parser.add_argument('--max_source_tokens', default=399)
    parser.add_argument('--max_target_tokens', default=99)

    args = parser.parse_args()

    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    for set in ['train', 'val', 'test']:
        articles, summaries = parse_bin(
            os.path.join(args.source_dir, set), args.max_source_tokens, args.max_target_tokens)
        print("Retrieved articles and summaries for {}".format(set))
        tgt_set = 'valid' if set == 'val' else set
        with open(os.path.join(args.target_dir, '{}.src-tgt.src'.format(tgt_set)), 'w') as f:
            f.writelines(articles)
        with open(os.path.join(args.target_dir, '{}.src-tgt.tgt'.format(tgt_set)), 'w') as f:
            f.writelines(summaries)
        print("Wrote articles and summaries for {}".format(set))
