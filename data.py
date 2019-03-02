import numpy as np
import os
import struct
from torch.utils.data import Dataset
from tensorflow.core.example import example_pb2
from fairseq.tasks import FairseqTask

class SummaryDataset(Dataset):
    '''
    '''

    def __init__(self, datapath):
        self._datapath = datapath
        self._articles = []  # _articles[i][0] = full text, _articles[i][1] = given summary
        self._preprocess()

    def _preprocess(self):
        ''' Import the dataset from the binary files.

        Code taken and adapted from: https://github.com/abisee/pointer-generator/blob/master/data.py'''

        filelist = os.listdir(self._datapath)  # get the list of datafiles
        filelist = [os.path.join(self._datapath,f) for f in filelist]
        filelist.sort()
        assert filelist, ('Error: Empty filelist at %s' %
                          self._datapath)  # check filelist isn't empty

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
                examples[0] = examples[0][2:-1]
                examples[1] = examples[1][2:-1]
                self._articles.append(examples)

    def __getitem__(self, index):
        return self._articles[index][0],self._articles[index][1]

    def __len__(self):
        return len(self._articles)

class SummarizationTask(FairseqTask):

    ##TODO finish this and design in in the same way as  https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/language_modeling.py
    def __init__(self,args):
        super().__init__(args)
        self.source_


#
# if __name__ == "__main__":
#     number_article = 0
#     dataset = SummaryDataset(r"data/chunked/test_0*.bin")
#
#     print(dataset[number_article])
#     print("Number of examples:", len(dataset))
#
#     for example in dataset:
#         assert len(example) == 2
