import tensorflow as tf
layers = tf.keras.layers
import numpy as np
import argparse
import logging
execfile('/home/gray/scripts/data.py')
tf.enable_eager_execution()
FLAGS=None

class LanguageModelRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,
                 seq_length, hidden_size, 
                 mask_zero=True, use_cudnn=False ):
        super(LanguageModelRNN, self).__init__()
        self.optimizer=optimizer
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim, 
                                          input_length=seq_length,
                                          mask_zero=mask_zero)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(hidden_size, return_sequences=True)
        else:
            self.recurrent = layers.GRU(hidden_size, return_sequences=True)
        self.token_logits = layers.TimeDistributed(
            layers.Dense(vocab_size, activation='linear'))
        logging.info('Language Model initialized.')
    def call(self, x):
        x = self.embedding(x)
        xs = self.recurrent(x)
        logits = self.token_logits(xs)
        return logits
    
    def train(self, dataset, optimizer):
        # Trains on iterable in dataset
        for (batch, (data, labels)) in enumerate(dataset):
            if batch % 80 == 0:
                print()
            print('.', end='')
            # labels are of shape (batch, seqlength)
            # but need them to be of (batch, seqlength, vocab) for TimeDistrib
            labels = tf.keras.utils.to_categorical(labels, self.vocab_size)
            with tf.GradientTape() as tape:
                logits = self(data)
                loss = softmax_cross_entropy(labels, logits)
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(grads, self.variables), 
                                      global_step=\
                                      tf.train.get_or_create_global_step())
            if batch % 80 == 0:
                print(loss)



def softmax_cross_entropy(labels, logits, **kwargs):
    loss = tf.losses.softmax_cross_entropy(labels, logits, **kwargs)
    tf.contrib.summary.scalar(loss, 'cross_entropy_loss')
    return loss



class UniRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,
                 seq_length, hidden_size, n_classes, use_cudnn=True):
        super(UniRNN, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, 
                                          input_length=seq_length, mask_zero=True)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(hidden_size, return_sequences=True)
        else:
            self.recurrent = layers.GRU(hidden_size, return_sequences=True)
        
        self.token_logits = layers.TimeDistributed(
            layers.Dense(vocab_size, activation='linear'))
        self.class_logits = layers.Dense(n_classes, activation='linear')

    def call(self, x, pretraining=False):
        x = self.embedding(x)
        print(x)
        xs = self.recurrent(x)
        if pretraining:
            xs = self.token_logits(xs)
        else:
            x = self.class_logits(x[-1])
        return x



def main():
    device = '/gpu:0' if not FLAGS.no_gpu else '/cpu:0'
    data_holder = data.make_dataset(FLAGS.dataset_name,
                                seq_length=FLAGS.max_seq_length,
                                vocab_size=FLAGS.max_vocab_size,
                                char_level=FLAGS.char_level,
                                data_dir=FLAGS.data_dir,
                                cache_path=FLAGS.cache_path)
    dataset = data_holder.dataset
    logger.info('Loaded dataset {} with shape {}'
                .format(FLAGS.dataset_name, dataset.output_shapes))
    if FLAGS.model == 'LanguageModelRNN':
        model = LanguageModelRNN(FLAGS.

    





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/',
        metavar='DIR',
        help=('Directory to store/fetch input data.'
              'Default = "./data/"'))
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/',
        metavar='DIR')
    parser.add_argument(
        '--log-dir',
        type=str
        default=None,
        metavar='DIR')
    parser.add_argument(
        '--model',
        type=str,
        default=None)
    parser.add_argument(
        '--cache-path',
        type=str,
        default=None)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--RNN-sizes',
        type=int,
        nargs='*',
        default=[128])
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=64)
    parser.add_argument(
        '--char-level',
        action='store_true',
        default=False)
    parser.add_argument(
        '--max-vocab-size',
        type=int,
        default=1000)
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default='32')
    parser.add_argument(
        '--no-mask',
        action='store_true',
        default=False)
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='op_spam')
    parser.add_argument(
        '--combine-reviews',
        action='store_true',
        default=False)

    FLAGS, unparsed = parser.parse_known_args()
    logging.info('Parsed flags:{}'.format(str(FLAGS))
    main()


