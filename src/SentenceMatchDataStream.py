import numpy as np
import re
import sys

if sys.version_info < (3,0):
    py = 'py2'
else:
    py = 'py3'


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # starting point of each batch

def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val


def read_all_instances(inpath, infeatpath, word_vocab=None, label_vocab=None, char_vocab=None, max_sent_length=100,
                       max_char_per_word=10, max_sent_char_length=200, isLower=True):
    instances = []
    assert len(open(inpath, 'rt').readlines()) == len(open(infeatpath, 'rt').readlines())
    infile = open(inpath, 'rt')
    infeatfile = open(infeatpath, 'rt')
    idx = -1
    for (line, featline) in zip(infile, infeatfile):
        idx += 1
        if py == 'py3': line = line.strip()
        else: line = line.strip().decode('utf-8')
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = items[1].strip()
        sentence2 = items[2].strip()
        cur_ID = "{}".format(idx)
        if len(items)>=4: cur_ID = items[3]
        if isLower:
            sentence1 = sentence1.lower()
            sentence2 = sentence2.lower()
        if label_vocab is not None:
            label_id = label_vocab.getIndex(label)
            if label_id >= label_vocab.vocab_size: label_id = 0
        else:
            label_id = int(label)
        word_idx_1 = word_vocab.to_index_sequence(sentence1)
        word_idx_2 = word_vocab.to_index_sequence(sentence2)
        if char_vocab is not None:
            char_idx_1 = char_vocab.to_char_index_sequence(sentence1)
            char_idx_2 = char_vocab.to_char_index_sequence(sentence2)
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1, max_char_per_word=max_char_per_word)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2, max_char_per_word=max_char_per_word)
        else:
            char_idx_1 = None
            char_idx_2 = None
            char_matrix_idx_1 = None
            char_matrix_idx_2 = None
        
        if len(word_idx_1) > max_sent_length:
            word_idx_1 = word_idx_1[:max_sent_length]
            if char_vocab is not None: char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
        if len(word_idx_2) > max_sent_length:
            word_idx_2 = word_idx_2[:max_sent_length]
            if char_vocab is not None: char_matrix_idx_2 = char_matrix_idx_2[:max_sent_length]
        if char_vocab is not None: 
            if len(char_idx_1) > max_sent_char_length: char_idx_1 = char_idx_1[:max_sent_char_length]
            if len(char_idx_2) > max_sent_char_length: char_idx_2 = char_idx_2[:max_sent_char_length]

        if py == 'py3': featline = featline.strip()
        else: featline = featline.strip().decode('utf-8')
        if featline.startswith('-'): continue
        featitems = re.split("\t", featline)

        instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2, char_idx_1, char_idx_2, featitems, cur_ID))
    infile.close()
    infeatfile.close()
    return instances

class SentenceMatchDataStream(object):
    def __init__(self, inpath, infeatpath, word_vocab=None, char_vocab=None, label_vocab=None,
                 isShuffle=False, isLoop=False, isSort=True, options=None):
        instances = read_all_instances(inpath, infeatpath, word_vocab=word_vocab, label_vocab=label_vocab,
                    char_vocab=char_vocab, max_sent_length=options.max_sent_length, max_char_per_word=options.max_char_per_word,
                    max_sent_char_length=options.max_sent_char_length, isLower=options.isLower)

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[4]), len(instance[5]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, options.batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in range(batch_start, batch_end):
                cur_instances.append(instances[i])
            if options.global_padding:
                cur_batch = InstanceBatch(cur_instances, with_char=options.with_char, max_sent_length=options.max_sent_length, max_sent_char_length=options.max_sent_char_length)
            else:
                cur_batch = InstanceBatch(cur_instances, with_char=options.with_char, max_sent_length=0, max_sent_char_length=0)
            self.batches.append(cur_batch)

        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
        # print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def shuffle(self):
        if self.isShuffle: np.random.shuffle(self.index_array)

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[self.index_array[i]]


class InstanceBatch(object):
    def __init__(self, instances, with_char=False, max_sent_length=0, max_sent_char_length=0):
        self.instances = instances
        self.batch_size = len(instances)
        self.question_len = max_sent_length
        self.passage_len = max_sent_length
        
        self.label_truth = []  # [batch_size]
        self.question_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_question_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
        self.passage_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_passage_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        self.in_question_passage_features = []  # tf.placeholder(tf.float32, [None, None]) # [batch_size, features_dim]

        if with_char:
            self.question_char_len = max_sent_char_length
            self.passage_char_len = max_sent_char_length

            self.question_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.in_question_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, question_len, q_char_len]
            self.passage_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            self.in_passage_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, passage_len, p_char_len]

            self.question_sent_char_lengths = []  # tf.placeholder(tf.int32, [None])
            self.in_question_sent_chars = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.passage_sent_char_lengths = []  # tf.placeholder(tf.int32, [None])
            self.in_passage_sent_chars = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        for (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2, char_idx_1, char_idx_2, featitems, cur_ID) in instances:
            self.label_truth.append(label_id)
            
            cur_question_length = len(word_idx_1)
            cur_passage_length = len(word_idx_2)
            if self.question_len < cur_question_length: self.question_len = cur_question_length
            if self.passage_len < cur_passage_length: self.passage_len = cur_passage_length
            self.question_lengths.append(cur_question_length)
            self.in_question_words.append(word_idx_1)
            self.passage_lengths.append(cur_passage_length)
            self.in_passage_words.append(word_idx_2)
           
            self.in_question_passage_features.append(featitems)

            if with_char:
                self.in_question_chars.append(char_matrix_idx_1)
                self.in_passage_chars.append(char_matrix_idx_2)
                self.question_char_lengths.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                self.passage_char_lengths.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])

                cur_question_char_length = len(char_idx_1)
                cur_passage_char_length = len(char_idx_2)
                if self.question_char_len < cur_question_char_length: self.question_char_len = cur_question_char_length
                if self.passage_char_len < cur_passage_char_length: self.passage_char_len = cur_passage_char_length
                self.question_sent_char_lengths.append(cur_question_char_length)
                self.in_question_sent_chars.append(char_idx_1)
                self.passage_sent_char_lengths.append(cur_passage_char_length)
                self.in_passage_sent_chars.append(char_idx_2)

        # padding all value into np arrays
        self.label_truth = np.array(self.label_truth, dtype=np.int32)
        self.question_lengths = np.array(self.question_lengths, dtype=np.int32)
        self.in_question_words = pad_2d_vals(self.in_question_words, self.batch_size, self.question_len, dtype=np.int32)
        self.passage_lengths = np.array(self.passage_lengths, dtype=np.int32)
        self.in_passage_words = pad_2d_vals(self.in_passage_words, self.batch_size, self.passage_len, dtype=np.int32)
        self.in_question_passage_features = np.array(self.in_question_passage_features, dtype=np.float)
        if with_char:
            max_char_length1 = np.max([np.max(aa) for aa in self.question_char_lengths])
            self.in_question_chars = pad_3d_vals(self.in_question_chars, self.batch_size,  self.question_len,
                                                   max_char_length1, dtype=np.int32)
            max_char_length2 = np.max([np.max(aa) for aa in self.passage_char_lengths])
            self.in_passage_chars = pad_3d_vals(self.in_passage_chars, self.batch_size,  self.passage_len,
                                                max_char_length2, dtype=np.int32)

            self.question_char_lengths = pad_2d_vals(self.question_char_lengths, self.batch_size,  self.question_len)
            self.passage_char_lengths = pad_2d_vals(self.passage_char_lengths, self.batch_size,  self.passage_len)

            self.question_sent_char_lengths = np.array(self.question_sent_char_lengths, dtype=np.int32)
            self.in_question_sent_chars = pad_2d_vals(self.in_question_sent_chars, self.batch_size, self.question_char_len, dtype=np.int32)
            self.passage_sent_char_lengths = np.array(self.passage_sent_char_lengths, dtype=np.int32)
            self.in_passage_sent_chars = pad_2d_vals(self.in_passage_sent_chars, self.batch_size, self.passage_char_len, dtype=np.int32)
