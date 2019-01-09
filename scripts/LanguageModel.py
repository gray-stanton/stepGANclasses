import time
import os
import tensorflow as tf
import model_utils
import data
import numpy as np
layers = tf.keras.layers
FLAGS = None

def create_hparams(args):
    hparams = tf.contrib.training.HParams(
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        embedding_dim=64,
        rnn_size=128,
        use_cudnn=False)
    if args:
        hparams.parse(args)
    return hparams

def main():
    # Parse
    parser = model_utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    # Setup model_dir
    if FLAGS.model_name is None:
        model_name = "LanguageModel"
    else:
        model_name = FLAGS.model_name

    model_dir = os.path.abspath(FLAGS.base_dir) + '/{}/'.format(model_name)
    if not os.path.exists(model_dir):
        model_utils.setup_model_dir(model_dir, create_base=True)
    if FLAGS.no_restore:
        model_utils.remove_history(model_dir)
        model_utils.setup_model_dir(model_dir, create_base=False)
    # Start logging
    logger = model_utils.get_logger(model_name, model_dir)
    logger.info("Started constructing {}".format(model_name))
    logger.info("Parsed args {}".format(FLAGS))
    if FLAGS.no_restore:
        logger.info('Not restoring, deleted history.')

    # Get Dataset
    logger.info("Getting dataset {}".format(FLAGS.dataset_name))
    full_dataset, tokenizer, size = data.make_dataset(FLAGS.dataset_name,
                                                      FLAGS.dataset_type,
                                                      FLAGS.data_dir,
                                                      FLAGS.seq_length)
    # Create model
    hparams = create_hparams(FLAGS.hparams)
    lm = LanguageModel(tokenizer.vocab_size,
                       hparams.embedding_dim,
                       hparams.rnn_size,
                       hparams.use_cudnn)
    optimizer = tf.train.AdamOptimizer(hparams.lr,
                                       hparams.beta1,
                                       hparams.beta2,
                                       hparams.epsilon)
    epoch_count = tf.Variable(1, 'epoch_count')
    global_step = tf.train.get_or_create_global_step()
    logger.info("Model created")
    # Create checkpointing
    checkpoint_dir = os.path.abspath(model_dir + 'ckpts/' + FLAGS.run_name)
    logger.info("Checkpoints at {}".format(checkpoint_dir))
    checkpoint_prefix = checkpoint_dir + '/ckpt'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, 
                                     lm=lm,
                                     epoch_count=epoch_count,
                                     global_step=global_step)
    if not FLAGS.no_restore:
        if not FLAGS.load_checkpoint is None:
            load_checkpoint = FLAGS.load_checkpoint
        else:
            load_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            logger.info("Loading latest checkpoint...")
        logger.info("Loading checkpoint {}".format(load_checkpoint))
        checkpoint.restore(load_checkpoint)
    
    # Create summary writer
    summary_dir = model_dir + 'log/' + FLAGS.run_name + '/'
    summary_writer = tf.contrib.summary.create_file_writer(
        summary_dir, flush_millis=1000)


    # Training
    if FLAGS.mode == "train":
        logger.info("Beginning training...")
        device = '/gpu:0' if not FLAGS.no_gpu else '/cpu:0'
        # Get training Dataset
        logger.info("Full dataset size: {}".format(int(size)))
        logger.info("Train dataset size: {}".format(
            int(size * FLAGS.use_frac * FLAGS.train_frac)))
        train_dataset, valid_dataset = model_utils.split(full_dataset,
                                                         size,
                                                         FLAGS.use_frac,
                                                         FLAGS.train_frac)
        train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
        valid_dataset = valid_dataset.batch(FLAGS.batch_size, drop_remainder=True)
        train_dataset = (tf.data.experimental
                            .prefetch_to_device(device)(train_dataset))
        valid_dataset = (tf.data.experimental
                            .prefetch_to_device(device)(valid_dataset))
        # Train loop
        train_losses = []
        val_losses = []
        patience_count = 0
        for epoch in range(FLAGS.epochs):
            cur_epoch = epoch_count.numpy() + epoch
            logger.info("Starting epoch {}...".format(cur_epoch))
            start = time.time()
            with summary_writer.as_default():
                train_loss = lm.train(train_dataset, optimizer,
                                      global_step, FLAGS.log_interval)
                logger.info("Epoch {} complete: train loss = {:0.03f}".format(
                    cur_epoch, train_loss))
                logger.info("Validating...")
                val_loss = lm.evaluate(valid_dataset)
                logger.info("Validation loss = {:0.03f}".format(val_loss))
            time_elapsed = time.time() - start
            logger.info("Took {:0.01f} seconds".format(time_elapsed))
            # Checkpoint
            if FLAGS.early_stopping:
                if not val_losses or val_loss < min(val_losses) - FLAGS.es_delta:
                    logger.info("Checkpointing...")
                    checkpoint.save(checkpoint_prefix)
                elif patience_count + 1 > FLAGS.patience:
                    logger.info("Early stopping reached")
                    break
                else:
                    patience_count+=1
            else:
                logger.info("Checkpointing...")
                checkpoint.save(checkpoint_prefix)


    elif FLAGS.mode == "eval":
        logger.info("Beginning evaluation...")
        device = '/gpu:0' if not FLAGS.no_gpu else '/cpu:0'
        with summary_writer.as_default():
            val_loss = lm.evaluate(full_dataset)
            logger.info("Validation loss: {:0.02f}".format(val_loss))

    elif FLAGS.mode == "generate":
        # Generate samples
        logger.info("Generating samples...")
        for _ in range(FLAGS.num_samples):
            tokens = tokenizer.tokenize(FLAGS.seed_text)
            inp = tf.constant(np.array(tokens, dtype=np.int16))
            inp = tf.expand_dims(inp, 0)
            _, state = lm.call_with_state(inp[:, 0:-1]) # Setup state
            cur_token = tokens[-1]
            done = False
            while not done:
                inp = tf.constant(np.array([cur_token], dtype=np.int16))
                inp = tf.expand_dims(inp, 0)
                logits, state = lm.call_with_state(inp, state)
                logits = tf.squeeze(logits, 0)
                logits = logits/FLAGS.temperature
                cur_token = tf.multinomial(logits, num_samples=1)[-1,0].numpy()
                tokens.append(cur_token)
                if len(tokens) > FLAGS.sample_length:
                    done = True
            logger.info("{}".format(tokenizer.untokenize(tokens)))
            lm.recurrent.reset_states()
                    

class LanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,  rnn_size, use_cudnn=False):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        mask_zero = not use_cudnn # CUDNN doesn't support masking
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          mask_zero=mask_zero)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(rnn_size,
                                             return_sequences=True,
                                             return_state = True,
                                             stateful = True)
        else:
            self.recurrent = layers.GRU(rnn_size,
                                        return_sequences=True,
                                        return_state = True,
                                        stateful=True)
        self.token_logits = layers.TimeDistributed(
            layers.Dense(vocab_size, activation='linear'))
    def call(self, x):
        x = self.embedding(x)
        xs, _ = self.recurrent(x)
        logits = self.token_logits(xs)
        return logits

    def call_with_state(self, x, initial_state = None):
        x = self.embedding(x)
        xs, state = self.recurrent(x, initial_state)
        logits = self.token_logits(xs)
        return logits, state
    
    def train(self, dataset, optimizer, global_step, log_interval):
        # Trains on iterable in dataset
        total_loss = 0
        batch_num = 1
        for (data, labels) in dataset:
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                log_interval, global_step=global_step):
                
                tf.assign_add(global_step, 1)
                # labels are of shape (batch, seqlength)
                # but need them to be of (batch, seqlength, vocab) for TimeDistrib
                labels = tf.keras.utils.to_categorical(labels, self.vocab_size)
                with tf.GradientTape() as tape:
                    logits = self(data)
                    loss = tf.losses.softmax_cross_entropy(labels, logits)
                grads = tape.gradient(loss, self.variables)
                optimizer.apply_gradients(zip(grads, self.variables),
                                          global_step=global_step)
                tf.contrib.summary.scalar('tr_cross_entropy', loss)
                tf.contrib.summary.scalar('tr_perplexity', tf.exp(loss))
                total_loss += loss
                batch_num += 1
                self.recurrent.reset_states() # Batches are shuffled
        return total_loss/batch_num
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


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    tf.enable_eager_execution()
    main()
    

