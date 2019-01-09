import time
import os
import tensorflow as tf
import model_utils
import data
import numpy as np
import LanguageModel
import pickle
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
        adv_size = 8.0,
        use_vat=False,
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
        model_name = "UniRNN"
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
    with open('/home/gray/code/seqgan-opinion-spam/data/happydb/happytok.pkl',
              'rb') as f:
        tokenizer = pickle.load(f)
    #full_dataset, tokenizer, size = data.make_dataset(FLAGS.dataset_name,
    #                                                  FLAGS.dataset_type,
    #                                                  FLAGS.data_dir,
    #                                                  FLAGS.seq_length)
    # Create model
    hparams = create_hparams(FLAGS.hparams)
    model = UniRNN(tokenizer.vocab_size,
                       hparams.embedding_dim,
                       hparams.rnn_size,
                       FLAGS.n_classes,
                       hparams.use_vat,
                       hparams.adv_size,
                       hparams.use_cudnn)
    model_optimizer = tf.train.AdamOptimizer(hparams.lr,
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
    checkpoint = tf.train.Checkpoint(model_optimizer=model_optimizer, 
                                     model=model,
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
    
    # Upload pretrained embeddings and RNN if using
    lm = LanguageModel.LanguageModel(tokenizer.vocab_size,
                       hparams.embedding_dim,
                       hparams.rnn_size,
                       hparams.use_cudnn)
    if not FLAGS.pretrained_lm_dir is None:
        logger.info('Loading pretrained LM at {}'.format(
            FLAGS.pretrained_lm_dir))
        # TODO Tensorflow is parsing doubled slashes...
        pretrain_ckpt_path = os.path.normpath(FLAGS.pretrained_lm_dir)
        print(pretrain_ckpt_path)
        lm_checkpoint = tf.train.Checkpoint(lm=lm)
        path = tf.train.latest_checkpoint(pretrain_ckpt_path)
        checkpoint.restore(path)
        logger.info('Loaded {}'.format(path))
        model.embedding.set_weights(lm.embedding.get_weights())
        model.recurrent.set_weights(lm.recurrent.get_weights())


    # Create summary writer
    summary_dir = model_dir + 'log/' + FLAGS.run_name + '/'
    summary_writer = tf.contrib.summary.create_file_writer(
        summary_dir, flush_millis=1000)


    # Training
    if FLAGS.mode == "train":
        logger.info("Beginning training...")
        device = '/gpu:0' if not FLAGS.no_gpu else '/cpu:0'
        # Get training Dataset
        #logger.info("Full dataset size: {}".format(int(size)))
        #logger.info("Train dataset size: {}".format(
        #    int(size * FLAGS.use_frac * FLAGS.train_frac)))
        #TODO fix this
        trainX =np.load(
            '/home/gray/code/seqgan-opinion-spam/data/happydb/moments_tr_data.npy')
        trainY = np.load(
            '/home/gray/code/seqgan-opinion-spam/data/happydb/moments_tr_labels.npy')
        trainY = np.expand_dims(trainY, 1)
        valX = np.load(
            '/home/gray/code/seqgan-opinion-spam/data/happydb/moments_val_data.npy')
        valY = np.load(
            '/home/gray/code/seqgan-opinion-spam/data/happydb/moments_val_labels.npy')
        valY = np.expand_dims(valY, 1)
        train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valX, valY))

        
        #train_dataset, valid_dataset = model_utils.split(full_dataset,
        #                                                 size,
        #                                                 FLAGS.use_frac,
        #                                                 FLAGS.train_frac)
        train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
        valid_dataset = valid_dataset.batch(FLAGS.batch_size, drop_remainder=True)
        #train_dataset = (tf.data.experimental
        #                    .prefetch_to_device(device)(train_dataset))
        #valid_dataset = (tf.data.experimental
        #                    .prefetch_to_device(device)(valid_dataset))
        # Train loop
        train_losses = []
        val_losses = []
        min_val_loss = 1e8
        patience_count = 0
        for epoch in range(FLAGS.epochs):
            cur_epoch = epoch_count.numpy() + epoch
            logger.info("Starting epoch {}...".format(cur_epoch))
            start = time.time()
            with summary_writer.as_default():
                train_loss = model.train(train_dataset, model_optimizer,
                                      global_step, FLAGS.log_interval)
                logger.info("Epoch {} complete: train loss = {:0.03f}".format(
                    cur_epoch, train_loss))
                logger.info("Validating...")
                val_loss = model.evaluate(valid_dataset)
                logger.info("Validation loss = {:0.03f}".format(val_loss))
            time_elapsed = time.time() - start
            logger.info("Took {:0.01f} seconds".format(time_elapsed))
            # Checkpoint
            if FLAGS.early_stopping:
                if  val_loss < (min_val_loss - FLAGS.es_delta):
                    logger.info("Checkpointing...")
                    checkpoint.save(checkpoint_prefix)
                    patience_count = 0
                    min_val_loss=val_loss
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
            val_loss = model.evaluate(full_dataset)
            logger.info("Validation loss: {:0.02f}".format(val_loss))


class UniRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,
                 rnn_size, n_classes, use_vat, adv_size, use_cudnn=False):
        super(UniRNN, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.use_vat = use_vat
        self.adv_size = adv_size
        mask_zero = not use_cudnn # CUDNN doesn't support masking
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          mask_zero=mask_zero)
        if use_cudnn:
            self.recurrent = layers.CuDNNGRU(rnn_size,
                                             return_sequences=False,
                                             return_state = False,
                                             stateful = False)
        else:
            self.recurrent = layers.GRU(rnn_size,
                                        return_sequences=False,
                                        return_state = False,
                                        stateful=False,
                                        recurrent_dropout=0.3)
        self.class_logits = layers.Dense(self.n_classes, activation='linear')

    def call(self, x):
        x = self.embedding(x)
        xs  = self.recurrent(x)
        logits = self.class_logits(xs)
        return logits
    def call_post_embedding(self, embedded_x):
        xs = self.recurrent(embedded_x)
        logits = self.class_logits(xs)
        return logits
    # VAT code adapted from https://github.com/takerum/vat_tf/blob/master/vat.py
    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = tf.random_normal(shape=tf.shape(x))
        for _ in range(1):
            #  original paper found 1 power iteration sufficient
            d = 1e6 * get_normalized_vector(d) #small finite difference
            logit_p = logit
            with tf.GradientTape() as tape:
                logit_m = self.call_post_embedding(x + d)
                dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tape.gradient(dist, [d])[0]

        return self.adv_size * get_normalized_vector(d)



    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit = tf.stop_gradient(logit)
        logit_p = logit
        logit_m = self.call_post_embedding(x + r_vadv)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss


    
    def train(self, dataset, optimizer, global_step, log_interval):
        # Trains on iterable in dataset
        total_loss = 0
        batch_num = 1
        acc = tf.contrib.eager.metrics.Accuracy('tr_acc')
        for (data, labels) in dataset:
            with tf.contrib.summary.always_record_summaries():
                
                tf.assign_add(global_step, 1)
                
                # For SSL, unlabeled examples passed with "label" -1
                # Mask those out for supervised softmax loss
                # Create fake labels as zero for computation.
                unlabeled = tf.equal(labels, -1)
                labeled_weights = tf.squeeze(
                    1.0 - tf.cast(unlabeled, dtype='float32'), 1)
                fake_labels = tf.where(unlabeled,
                                       tf.zeros(labels.shape, dtype='int16'),
                                       labels)
    
                labels = tf.keras.utils.to_categorical(labels, self.n_classes)
                with tf.GradientTape() as tape:
                    logits = self(data)
                    # compute softmax for only labeled examples
                    cross_ent_loss = tf.losses.softmax_cross_entropy(labels,
                                                            logits,
                                                            weights=labeled_weights)
                    loss = cross_ent_loss
                    # for all examples, labeled and unlabeled, compute VAT loss
                    # if using
                    if self.use_vat:
                        # For discrete input, use VAT after embedding.
                        post_embedding = self.embedding(data)
                        vat_loss = self.virtual_adversarial_loss(post_embedding,
                                                              logits)
                        loss += vat_loss


                grads = tape.gradient(loss, self.variables)
                optimizer.apply_gradients(zip(grads, self.variables),
                                          global_step=global_step)
                tf.contrib.summary.scalar('tr_cross_entropy', cross_ent_loss)
                if self.use_vat:
                    tf.contrib.summary.scalar('tr_vat_loss', vat_loss)
                    tf.contrib.summary.scalar('tr_tot_loss', loss)
                acc(tf.argmax(labels, 1), tf.argmax(logits, 1))
                acc.result()
                total_loss += loss
                batch_num += 1
        return total_loss/batch_num

    def evaluate(self, dataset):
        total_loss = 0
        batch_num = 1
        acc = tf.contrib.eager.metrics.Accuracy('val_acc')
        for (batch, (data, labels)) in enumerate(dataset):
            with tf.contrib.summary.always_record_summaries():
                labels = tf.keras.utils.to_categorical(labels, self.n_classes)
                logits = self(data)
                loss = tf.losses.softmax_cross_entropy(labels, logits)
                tf.contrib.summary.scalar('val_cross_entropy', loss)
                acc(tf.argmax(labels, 1), tf.argmax(logits, 1))
                acc.result()
                total_loss += loss
                batch_num += 1
        return total_loss/batch_num


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp 

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    tf.enable_eager_execution()
    main()


