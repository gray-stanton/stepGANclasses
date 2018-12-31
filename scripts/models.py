
import time
import tensorflow as tf
import numpy as np
import argparse
import logging
import data
import os
tf.enable_eager_execution()
layers = tf.keras.layers
FLAGS = None

class LanguageModelRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,  hidden_size,
                 mask_zero=True, use_cudnn=False):
        super(LanguageModelRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          mask_zero=mask_zero)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(hidden_size,
                                             return_sequences=True,
                                             stateful = True)
        else:
            self.recurrent = layers.GRU(hidden_size,
                                        return_sequences=True,
                                        stateful=True)
        self.token_logits = layers.TimeDistributed(
            layers.Dense(vocab_size, activation='linear'))
        logging.info('Language Model initialized.')
    def call(self, x):
        x = self.embedding(x)
        xs = self.recurrent(x)
        logits = self.token_logits(xs)
        return logits
    
    def train(self, dataset, optimizer, global_step):
        # Trains on iterable in dataset
        for (batch, (data, labels)) in enumerate(dataset):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                FLAGS.log_interval, global_step=global_step):
                
                tf.assign_add(global_step, 1)
                # labels are of shape (batch, seqlength)
                # but need them to be of (batch, seqlength, vocab) for TimeDistrib
                labels = tf.keras.utils.to_categorical(labels, self.vocab_size)
                with tf.GradientTape() as tape:
                    logits = self(data)
                    loss = softmax_cross_entropy(labels, logits)
                grads = tape.gradient(loss, self.variables)
                optimizer.apply_gradients(zip(grads, self.variables),
                                          global_step=global_step)
                tf.contrib.summary.scalar('tr_perplexity', tf.exp(loss))
                self.recurrent.reset_states() # Batches are shuffled
    def evaluate(self, dataset):
        mean_loss = 0
        for (batch, (data, labels)) in enumerate(dataset):
            with tf.contrib.summary.always_record_summaries():
                labels = tf.keras.utils.to_categorical(labels, self.vocab_size)
                logits = self(data)
                loss = tf.losses.softmax_cross_entropy(labels, logits)
                mean_loss = ((mean_loss * (batch+1)) + loss)/(batch+2)
                tf.contrib.summary.scalar('val_cross_ent_loss', loss)
                tf.contrib.summary.scalar('val_perplexity', tf.exp(loss))
        return mean_loss
    





def softmax_cross_entropy(labels, logits, **kwargs):
    loss = tf.losses.softmax_cross_entropy(labels, logits, **kwargs)
    tf.contrib.summary.scalar('tr_cross_entropy_loss', loss)
    return loss



class UniRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, 
                 hidden_size, n_classes,
                 mask_zero=True, use_cudnn=False):
        super(UniRNN, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          mask_zero=True)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(hidden_size,
                                             return_sequences=False)
        else:
            self.recurrent = layers.GRU(hidden_size,
                                        return_sequences=False)
        self.class_logits = layers.Dense(self.n_classes, activation='linear')

    def pretrained(self, language_model):
        self.embedding.set_weights(language_model.embedding.get_weights())
        self.recurrent.set_weights(language_model.recurrent.get_weights())
        return
 
    def call(self, x):
        x = self.embedding(x)
        xs = self.recurrent(x)
        o = self.class_logits(xs)
        return o
    
    def train(self, dataset, optimizer, global_step):
        # Trains on iterable in dataset
        acc = tf.contrib.eager.metrics.Accuracy('tr_acc')
        for (batch, (data, labels)) in enumerate(dataset):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                FLAGS.log_interval, global_step=global_step):
                
                tf.assign_add(global_step, 1)
                # labels are of shape (batch, seqlength)
                # but need them to be of (batch, seqlength, vocab) for TimeDistrib
                labels = tf.keras.utils.to_categorical(labels, self.n_classes)
                with tf.GradientTape() as tape:
                    logits = self(data)
                    loss = softmax_cross_entropy(labels, logits)
                grads = tape.gradient(loss, self.variables)
                optimizer.apply_gradients(zip(grads, self.variables),
                                          global_step=global_step)
                acc(tf.argmax(labels, 1), tf.argmax(logits, 1))
                acc.result()

    def evaluate(self, dataset):
        mean_loss = 0
        acc = tf.contrib.eager.metrics.Accuracy('val_acc')
        for (batch, (data, labels)) in enumerate(dataset):
            with tf.contrib.summary.always_record_summaries():
                labels = tf.keras.utils.to_categorical(labels, self.n_classes)
                logits = self(data)
                loss = tf.losses.softmax_cross_entropy(labels, logits)
                mean_loss = ((mean_loss * (batch+1)) + loss)/(batch+2)
                tf.contrib.summary.scalar('val_cross_ent_loss', loss)
                acc(tf.argmax(labels, 1), tf.argmax(logits, 1))
                acc.result()
        return mean_loss



def _split(ds, train_frac, valid_frac, test_frac, size):
    ds = ds.shuffle(10000)
    train_ds = ds.take(int(train_frac * size))
    test_ds =  ds.skip(int(train_frac * size))
    val_ds = test_ds.skip(int(valid_frac * size))
    test_ds = test_ds.take((test_frac * size))
    return train_ds, val_ds, test_ds

def main():
    device = '/gpu:0' if not FLAGS.no_gpu else '/cpu:0'
    full_dataset, tokenizer, size = data.make_dataset(FLAGS.dataset_name,
                                               FLAGS.dataset_type,
                                               FLAGS.data_dir,
                                               FLAGS.seq_length)
    # Dataset processing
    train_dataset, valid_dataset, test_dataset = _split(full_dataset, 
                                                        FLAGS.train_frac,
                                                        FLAGS.valid_frac,
                                                        FLAGS.test_frac, 
                                                        size)
    train_dataset =(train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
                    .prefetch(2*FLAGS.batch_size))
    valid_dataset =(valid_dataset.batch(FLAGS.batch_size, drop_remainder=True)
                    .prefetch(2*FLAGS.batch_size))
    test_dataset = (test_dataset.batch(FLAGS.batch_size, drop_remainder=True)
                    .prefetch(2*FLAGS.batch_size))

    logging.info('Loaded dataset {}_{} with shape {}'
                .format(FLAGS.dataset_name, FLAGS.dataset_type,
                        train_dataset.output_shapes))
    logging.info('Dataset size {}'.format(size))
    for x, y in train_dataset.take(1):
        logging.info('x:{} ; y:{}'.format(x.numpy(), y.numpy()))
    if FLAGS.pretrain:
        lm = LanguageModelRNN(tokenizer.vocab_size, FLAGS.embed_dim,
                              FLAGS.RNN_sizes[0],
                              (not FLAGS.no_mask), False)
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.pretrain_lm_dir)
        lm_ckpt = tf.train.Checkpoint(model = lm)
        lm_ckpt.restore(latest_ckpt)

    if FLAGS.model == 'LanguageModelRNN':
        model = LanguageModelRNN(tokenizer.vocab_size, FLAGS.embed_dim, 
                                 FLAGS.RNN_sizes[0],
                                 (not FLAGS.no_mask), False)
    elif FLAGS.model == 'UniRNN':
        model = UniRNN(tokenizer.vocab_size, FLAGS.embed_dim,
                       FLAGS.RNN_sizes[0], 2,
                       (not FLAGS.no_mask), False)
    else:
        raise NotImplementedError
    
    # Create optimizer
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    # Initialize summaries and checkpointing
    summary_writer = tf.contrib.summary.create_file_writer(
        FLAGS.log_dir, flush_millis=1000)
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if latest_ckpt:
        print('Using latest checkpoint at ' + latest_ckpt)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(latest_ckpt)
    
    
    # Create global TF ops
    global_step = tf.train.get_or_create_global_step()
    # Train loop
    if FLAGS.mode == 'train':
        with tf.device(device):
            for epoch in range(FLAGS.epochs):
                start = time.time()
                with summary_writer.as_default():
                    model.train(train_dataset, optimizer, global_step)
                    val_loss = model.evaluate(valid_dataset)
                    print(val_loss)
                end = time.time()
                checkpoint.save(checkpoint_prefix)
                logging.info('\nTrain time for epoch #%d (step %d): %f seconds' % 
                            (checkpoint.save_counter.numpy(),
                             global_step.numpy(),
                             end - start))

    elif FLAGS.mode == 'sample':
        seed_tokens = tokenizer.tokenize(FLAGS.sample_seed_text)
        seed = tf.constant(np.array(seed_tokens, dtype=np.int16))
        seed = tf.expand_dims(seed, 0)
        print(seed.shape)
        inp = seed
        tokens = seed_tokens
        for i in range(FLAGS.sample_length):
            logits = model(inp)
            logits = tf.squeeze(logits, 0)
            logits = logits/FLAGS.temperature
            sample_tok =tf.multinomial(logits, num_samples=1)[-1,0].numpy()
            tokens.append(sample_tok)
            inp = tf.expand_dims([sample_tok], 0)
        print(tokenizer.untokenize(tokens))










if __name__=='__main__':
    logging.basicConfig(filename='./model.log',level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default = 'train')
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
        type=str,
        default=None,
        metavar='DIR')
    parser.add_argument(
        '--model',
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
        '--seq-length',
        type=int,
        default='32')
    parser.add_argument(
        '--no-mask',
        action='store_true',
        default=False)
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='OSChar')
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='LM')
    parser.add_argument(
        '--epochs',
        default=1,
        type=int,
        metavar='N')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int)
    parser.add_argument(
        '--log-interval',
        default=10,
        type=int)
    parser.add_argument(
        '--train-frac',
        default = 0.7,
        type=float)
    parser.add_argument(
        '--valid-frac',
        default = 0.15,
        type=float)
    parser.add_argument(
        '--test-frac',
        default = 0.15,
        type=float)
    parser.add_argument(
        '--sample-seed-text',
        default = 'A',
        type = str)
    parser.add_argument(
        '--sample-length',
        default = 50,
        type = int)
    parser.add_argument(
        '--temperature',
        default = 1,
        type = float)
    parser.add_argument(
        '--pretrain',
        action='store_true',
        default=False)
    parser.add_argument(
        '--pretrained-lm-path',
        type=str)

    FLAGS, unparsed = parser.parse_known_args()
    logging.info('Parsed flags:{}'.format(str(FLAGS)))
    main()
