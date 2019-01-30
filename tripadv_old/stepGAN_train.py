import tensorflow as tf
import texar as tx
import importlib
import numpy as np
import logging
import os
import sys
import time
from tensorflow.python import debug as tf_debug

config = importlib.import_module('stepGAN_config_tripadvisor')

class Generator(tf.keras.Model):
    """Generator wrapper for checkpointing"""
    def __init__(self, vocab_size, decoder_config):
        super(Generator, self).__init__()
        self.decoder = tx.modules.BasicRNNDecoder(vocab_size=vocab_size,
                                                  hparams=decoder_config)


class RNNDiscriminator(tf.keras.Model):
    """Discriminator wrapper"""
    def __init__(self, disc_config):
        super(RNNDiscriminator, self).__init__()
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams = disc_config['encoder'])
            

class RNNClassifier(tf.keras.Model):
    def __init__(self, class_config):
        super(RNNClassifier, self).__init__()
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams = class_config['encoder'])

class RNNDiversifier(tf.keras.Model):
    def __init__(self, vocab_size, div_config):
        super(RNNDiversifier, self).__init__()
        div_config['encoder']['output_layer']['layer_size'] = vocab_size
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams = div_config['encoder'])

class Embedder(tf.keras.Model):
    def __init__(self,vocab_size, emb_config):
        super(Embedder, self).__init__()
        self.embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                        hparams=emb_config)

class RNNCritic(tf.keras.Model):
    def __init__(self, crit_config):
        super(RNNCritic, self).__init__()
        self.rec = tf.keras.layers.CuDNNGRU(**crit_config['rec'], return_sequences=True)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(**cric_config['dense']))
        
    def call(x):
        pass

def get_logger(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s")
    fh = logging.FileHandler("{0}/log.txt".format(log_dir))
    fh.setLevel(logging.DEBUG)
    logger = logging.getLogger("StepGAN")
    logger.addHandler(fh)
    return logger


class fakelogger():
    def __init__(self, f):
        self.logfile = f
    def debug(self, m):
        with open(self.logfile, 'a') as f:
            f.write(m + '\n')



## MODIFIED TO ALLOW CLASS EMBEDDING APPENDING
class ContextSoftmaxEmbeddingHelper(tf.contrib.seq2seq.Helper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).
    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.
    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, context, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
        if isinstance(embedding, tx.modules.EmbedderBase):
            embedding = embedding.embedding

        if callable(embedding):
            raise ValueError("`embedding` must be an embedding tensor or an "
                             "instance of subclass of `EmbedderBase`.")
        else:
            self._embedding = embedding
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        self.context = context
        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._start_inputs = tf.concat([self._start_inputs, self.context], axis=-1)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_dist = tf.nn.softmax(outputs / self._tau)
        sampler = tf.distributions.Categorical(logits=sample_dist)
        sample_ids = sampler.sample()
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = self._embedding_fn(sample_ids)
        ## Modified
        next_inputs = tf.concat([next_inputs, self.context], axis=-1)
        return (finished, next_inputs, state)


class Helper(object):
  """Interface for implementing sampling in seq2seq decoders.
  Helper instances are used by `BasicDecoder`.
  """

  def batch_size(self):
    """Batch size of tensor returned by `sample`.
    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  def sample_ids_shape(self):
    """Shape of tensor returned by `sample`, excluding the batch dimension.
    Returns a `TensorShape`.
    """
    raise NotImplementedError("sample_ids_shape has not been implemented")

  def sample_ids_dtype(self):
    """DType of tensor returned by `sample`.
    Returns a DType.
    """
    raise NotImplementedError("sample_ids_dtype has not been implemented")

  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass

class ContextGreedyEmbeddingHelper(Helper):
  """A helper for use during inference.
  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    if isinstance(embedding, tx.modules.EmbedderBase):
        embedding = embedding.embedding

    if callable(embedding):
        raise ValueError("`embedding` must be an embedding tensor or an "
                         "instance of subclass of `EmbedderBase`.")
    else:
        self._embedding = embedding
        self._embedding_fn = (
            lambda ids: tf.nn.embedding_lookup(embedding, ids))
    self.context = context
    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = tf.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = self._embedding_fn(self._start_tokens)
    self._start_inputs = tf.concat([self._start_inputs, self.context], axis = -1)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    finished = tf.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    sample_ids = tf.argmax(outputs, axis=-1, output_type=dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""
    del time, outputs  # unused by next_inputs_fn
    finished = tf.equal(sample_ids, self._end_token)
    all_finished = tf.reduce_all(finished)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: tf.concat([self._embedding_fn(sample_ids), self.context], axis=-1))
    return (finished, next_inputs, state)


class ContextSampleEmbeddingHelper(ContextGreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token, 
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(ContextSampleEmbeddingHelper, self).__init__(
        embedding, context, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.context = context

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature
    
    sample_id_sampler = tf.distributions.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    #p = tf.print(sample_ids)
    p2 = tf.print(sample_ids.shape)
    return sample_ids



















def print_out_array(header_names, value_lists, logger, final_line=None):
    header_format_string = ''.join(['{:<13}'] * len(header_names))
    logger.debug(header_format_string.format(*header_names))
    nvalues = len(value_lists[0])
    for i in range(nvalues):
        vals = [v[i] for v in value_lists if type(v[i]) != list]
        format_string = []
        for v in vals:
            if type(v) == np.str_:
                format_string.append('{:<13} ')
            else:
                format_string.append('{:<12.3f} ')
        format_string = ''.join(format_string)
        logger.debug(format_string.format(*vals))
    if final_line is not None:
        logger.debug(final_line)
        
    




def get_dataset(train_lm_only, hparams):
    if not train_lm_only:
        return tx.data.MultiAlignedData(hparams)
    else:
        return tx.data.MonoTextData(hparams)


def get_vocab(train_lm_only, d):
    if train_lm_only:
        return d.vocab
    else:
        return d.vocab('x')
    
    

def main(_):
    # Setup
    g = tf.Graph()
    with g.as_default():
        logger = get_logger(config.log_dir)
        global_step = tf.train.get_or_create_global_step()
        
        # Get data
        logger.info("Constructing graph...")
        train_data = get_dataset(config.train_lm_only, config.train_data)
        val_data = get_dataset(config.train_lm_only, config.val_data)
        test_data = get_dataset(config.train_lm_only, config.test_data)
        if config.use_unsup > 0:
            unsup_data = tx.data.MonoTextData(config.unsup_data)
            unsup_iterator = tx.data.DataIterator(unsup_data)
        iterator = tx.data.TrainTestDataIterator(train=train_data,
                                                 val=val_data,
                                                 test=test_data)
        data_batch = iterator.get_next()
        vocab = get_vocab(config.train_lm_only, train_data)
        vocab_size = vocab.size

        # Inputs
        label_inp = data_batch['x_text_ids']
        label_seq_lengths = data_batch['x_length'] - 1
        if config.use_unsup > 0:
            unsup_batch = unsup_iterator.get_next()
            unsup_inp = unsup_batch['text_ids']
            unsup_seq_lengths = unsup_batch['length']
            inp = tf.concat([label_inp, unsup_inp], axis=0)
            seq_lengths = tf.concat([label_seq_lengths, unsup_seq_lengths], axis=0)
        else:
            seq_lengths = label_seq_lengths
            inp = label_inp

        batch_size = tf.shape(inp)[0]
        padded_lengths = tf.shape(inp)[1]
        if not config.train_lm_only:
            data_labels = data_batch['label']
        else:
            data_labels = tf.placeholder(dtype = tf.int32)
        one_hot_labels= tf.one_hot(data_labels, 2, axis=-1)
        
        logger.info("Building model components...")
        # Embedding
        emb_model = Embedder(vocab_size, config.emb_hparams)
        embedder = emb_model.embedder
        # Generator
        gen_model = Generator(vocab_size, config.g_decoder_hparams)
        g_decoder = gen_model.decoder
        initial_state = g_decoder.zero_state(batch_size = batch_size,
                                             dtype=tf.float32)
        
        # Discriminator
        disc_model = RNNDiscriminator(config.disc_hparams)
        discriminator = disc_model.encoder

        # Classifier
        clas_model = RNNClassifier(config.clas_hparams)
        classifier = clas_model.encoder

        # Diversifier
        div_model = RNNDiversifier(vocab_size, config.div_hparams) # identical to generator
        diversifier = div_model.encoder
        
        # Critics
        disc_crit_layer = tf.layers.Dense(**config.disc_crit_hparams)
        disc_crit = tf.keras.layers.TimeDistributed(disc_crit_layer)
        clas_crit_layer = tf.layers.Dense(**config.clas_crit_hparams)
        clas_crit = tf.keras.layers.TimeDistributed(clas_crit_layer)

        logger.info("Creating Generator MLE training subgraph...")
        # Pre-train Generator subgraph
        with g.name_scope('gen_mle'):
            x = inp[:, 0:(tf.shape(inp)[1] -2)]
            x_emb = embedder(x)
            y = inp[:, 1:(tf.shape(inp)[1])-1]
            y_onehot = tf.one_hot(y, vocab_size)
            x_labeled_emb = embedder(label_inp)
            
            context_size = config.noise_size + 1
            zero_context = tf.zeros((batch_size, context_size))
            tiled_context = tf.reshape(
                tf.tile(zero_context, [1, x.shape[1]]), [-1, x.shape[1], context_size])
            x_emb_context = tf.concat([x_emb, tiled_context], axis = -1)
            
            outputs_mle, _, _ = g_decoder(
                initial_state=initial_state, 
                decoding_strategy='train_greedy',
                embedding=None,
                inputs=x_emb_context,
                sequence_length=seq_lengths - 1)
            
            logits_mle = outputs_mle.logits

            observed_logits = tf.reduce_sum(
                tf.multiply(logits_mle, y_onehot), axis = -1) #elementwise

            loss_mle_full = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=y,
                logits=logits_mle,
                sequence_length=seq_lengths,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False
                )
            loss_mle = tf.reduce_mean(loss_mle_full)
            g_variables = tx.utils.collect_trainable_variables([embedder, g_decoder])
            mle_train_op = tx.core.get_train_op(loss_mle,
                                                variables = g_variables,
                                                global_step = global_step,
                                                increment_global_step=True,
                                                hparams=config.g_opt_mle_hparams)
            mean_max_logit_mle = tf.reduce_mean(tf.reduce_max(logits_mle, axis = -1))
            mean_min_logit_mle = tf.reduce_mean(tf.reduce_min(logits_mle, axis = -1))
            mean_logit_mle = tf.reduce_mean(logits_mle)
            median_logit_mle = tf.reduce_mean(
                tf.contrib.distributions.percentile(logits_mle, q=50, axis=-1))
            logit_sd_mle = tf.sqrt(tf.reduce_mean(tf.square(logits_mle)) - tf.square(mean_logit_mle))
            tf.summary.scalar('mean_logit_mle', mean_logit_mle)
            tf.summary.scalar("mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("mean_min_logit_mle", mean_min_logit_mle)
            tf.summary.scalar('median_logit_mle', median_logit_mle)
            tf.summary.scalar('loss_mle', loss_mle)
            tf.summary.scalar('logit_sd_mle', logit_sd_mle)
            tf.summary.scalar('perplexity', tf.exp(loss_mle))
            mle_summaries = tf.summary.merge_all(scope='gen_mle')
        # MLE Validate summaries
        with g.name_scope('val_mle_summaries'):
            tf.summary.scalar('val_logit_sd_mle', logit_sd_mle)
            tf.summary.scalar("val_mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("val_mean_min_logit_mle", mean_min_logit_mle)
            tf.summary.scalar('val_mean_logit_mle', mean_logit_mle)
            tf.summary.scalar('val_loss_mle', loss_mle)
            tf.summary.scalar('val_perplexity', tf.exp(loss_mle))
            val_mle_summaries = tf.summary.merge_all(scope='val_mle_summaries')



        # Generate subgraph
        with g.name_scope('gen_sample'):
            logger.info("Creating token sequence sampling subgraph...")
            start_tokens = tf.cast(tf.fill([batch_size], 
                                   vocab.bos_token_id),
                                   dtype=tf.int32)
            random_context = tf.random.normal([batch_size, config.noise_size ])
            class_prior = tf.distributions.Bernoulli(probs=config.prior_prob)
            random_classes = class_prior.sample((batch_size, 1))
            random_vector = tf.concat([random_context, 
                                       tf.cast(random_classes, tf.float32)], 
                                      axis=-1)
            random_class_onehots = tf.one_hot(random_classes, 2, axis=-1)
            end_token = vocab.eos_token_id
            softmax_temperature = tf.constant(config.sampling_temperature, dtype=tf.float32)
            context_helper = ContextSampleEmbeddingHelper(
                embedder, random_vector, start_tokens, end_token, softmax_temperature)

            gen_outputs, _, gen_lengths = g_decoder(
                helper = context_helper,
                initial_state = initial_state,
                max_decoding_length = config.max_decoding_length_infer)
            gen_logits = gen_outputs.logits
            # Inefficient, use tf.gather
            gen_sample_ids = gen_outputs.sample_id
            observed_gen_logits = tf.reduce_sum(
                tf.math.multiply(
                    tf.one_hot(gen_sample_ids, vocab_size), gen_logits),
                axis=-1)
            mean_max_logit = tf.reduce_mean(tf.reduce_max(gen_logits, axis = -1))
            mean_min_logit = tf.reduce_mean(tf.reduce_min(gen_logits, axis = -1))
            median_logit = tf.reduce_mean(
                tf.contrib.distributions.percentile(gen_logits, q=50, axis=-1))
            mean_logit_gen = tf.reduce_mean(gen_logits)
            logit_sd_gen = tf.sqrt(tf.reduce_mean(tf.square(gen_logits)) -\
                                   tf.square(mean_logit_gen))
            mean_length = tf.reduce_mean(gen_lengths)
            max_length = tf.reduce_max(gen_lengths)
            min_length = tf.reduce_min(gen_lengths)

            sample_text = vocab.map_ids_to_tokens(gen_sample_ids)
            sep = '' if config.is_char else ' '
            sample_text = tf.reduce_join(sample_text, axis=-1, separator=sep)
            original_text = vocab.map_ids_to_tokens(x)
            original_text = tf.reduce_join(original_text, axis=-1, separator=sep)

            tf.summary.scalar("mean_max_logit", mean_max_logit)
            tf.summary.scalar("mean_min_logit", mean_min_logit)
            tf.summary.scalar('median_logit', median_logit)
            tf.summary.scalar('logit_sd', logit_sd_gen)
            tf.summary.scalar('mean_length', mean_length)
            tf.summary.scalar('max_length', max_length)
            tf.summary.scalar('min_length', min_length)
            gen_sample_summaries = tf.summary.merge_all(scope='gen_sample')
            gen_sample_summaries = tf.summary.merge([gen_sample_summaries, mle_summaries])
        # capture text
        sample_text_summary = tf.summary.text('sample_text', sample_text)
        original_text_summary = tf.summary.text('original_text', original_text)

        # Train Discriminator Subgraph
        with g.name_scope("disc_train"):
            logger.info("Creating discriminator training subgraph...")
            fake_seq = gen_sample_ids
            fake_seq_emb = embedder(fake_seq)
            f_progress_vector = tf.ones_like(fake_seq_emb)
            
            # Array of  [batch_size, tstep, 1] like 
            #  [[1, 2, 3, 4...]
            #   [1, 2, 3, 4...]]
            b_f = tf.shape(fake_seq_emb)[0]
            t_f = tf.shape(fake_seq_emb)[1]
            b_r = tf.shape(x_emb[:, 1:, :])[0]
            t_r = tf.shape(x_emb[:, 1:, :])[1]
            f_nsteps = tf.reshape(
                tf.tile( 
                    tf.range(start=1, limit=(t_f + 1)),
                    [b_f]),
                [b_f, t_f, 1])
            r_nsteps = tf.reshape(
                tf.tile(
                    tf.range(start=1, limit=(t_r + 1)),
                    [b_r]),
                [b_r, t_r, 1])
            f_nsteps = tf.cast(f_nsteps, tf.float32)
            r_nsteps = tf.cast(r_nsteps, tf.float32)

            gen_lengths_reshape = tf.cast(tf.reshape(
                tf.tile(gen_lengths, [t_f]), 
                [b_f, t_f, 1]), dtype=tf.float32)
            seq_lengths_reshape = tf.cast(tf.reshape(
                tf.tile(seq_lengths, [t_r]),
                [b_r, t_r, 1]), dtype=tf.float32)

            f_progress_vector = tf.ones_like(gen_lengths_reshape) -\
                (tf.multiply(1/gen_lengths_reshape, f_nsteps))
            r_progress_vector = tf.ones_like(seq_lengths_reshape) -\
                tf.multiply(1/seq_lengths_reshape , r_nsteps)
            f_progress_vector = tf.clip_by_value(f_progress_vector, 0, 1e8)
            r_progress_vector = tf.clip_by_value(r_progress_vector, 0, 1e8)
            real_inp = tf.concat([x_emb[:, 1:, :], r_progress_vector], axis = -1)
            fake_inp = tf.concat([fake_seq_emb, f_progress_vector], axis = -1)
            


            r_disc_q_logit, _, r_disc_cell_outputs = discriminator(
                real_inp, sequence_length= seq_lengths, return_cell_output=True)
            f_disc_q_logit, _, f_disc_cell_outputs = discriminator(
                fake_inp, sequence_length = gen_lengths, return_cell_output=True)
            r_disc_qvalues = tf.math.sigmoid(r_disc_q_logit)
            f_disc_qvalues = tf.math.sigmoid(f_disc_q_logit)
            r_disc_score = tf.reduce_mean(r_disc_q_logit, axis=1, keepdims=False)
            f_disc_score = tf.reduce_mean(f_disc_q_logit, axis=1, keepdims=False)

            r_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits = r_disc_score,
                multi_class_labels=tf.ones_like(r_disc_score), 
                label_smoothing = config.disc_label_smoothing_epsilon,
                reduction=tf.losses.Reduction.MEAN)
            f_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_disc_score,
                multi_class_labels=tf.zeros_like(f_disc_score), 
                label_smoothing = config.disc_label_smoothing_epsilon,
                reduction=tf.losses.Reduction.MEAN)
            disc_loss = r_disc_loss + f_disc_loss
            disc_loss.set_shape(())
            
            d_variables = tx.utils.collect_trainable_variables([discriminator])
            disc_train_op = tx.core.get_train_op(disc_loss,
                                                 global_step=global_step,
                                                 variables=d_variables,
                                                 increment_global_step=False,
                                                 hparams=config.d_opt_hparams)
            # Discriminator Critic
            r_disc_crit_inp = r_disc_cell_outputs[:, :-1]
            r_disc_crit_target = r_disc_q_logit[:, 1:]
            f_disc_crit_inp = f_disc_cell_outputs[:, :-1]
            f_disc_crit_target = f_disc_q_logit[:, 1:]
            r_disc_crit_baselines = disc_crit(r_disc_crit_inp)
            f_disc_crit_baselines = disc_crit(f_disc_crit_inp)
            # Somewhat concerned about masking here...
            r_disc_crit_loss = tf.losses.mean_squared_error(labels=r_disc_crit_target,
                                                          predictions=r_disc_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            f_disc_crit_loss = tf.losses.mean_squared_error(labels=f_disc_crit_target,
                                                          predictions=f_disc_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            if config.disc_crit_train_on_fake_only:
                disc_crit_loss = f_disc_crit_loss
            else:
                disc_crit_loss = r_disc_crit_loss + f_disc_crit_loss
            
            disc_crit_train_op = tx.core.get_train_op(
                disc_crit_loss, global_step=global_step,
                variables=[disc_crit.trainable_variables],
                increment_global_step=False,
                hparams=config.d_crit_opt_hparams)
            # Need to get baseline for last step
            last_f_disc_baseline = disc_crit_layer(f_disc_cell_outputs[:, -1])
            full_f_disc_crit_baselines = tf.concat([
                f_disc_crit_baselines, tf.expand_dims(last_f_disc_baseline, axis=1)],
                axis=1)


            r_probs = tf.math.sigmoid(r_disc_score)
            f_probs = tf.math.sigmoid(f_disc_score)
            r_preds = tf.round(r_probs)
            f_preds = tf.round(f_probs)
            mean_r_disc_score = tf.reduce_mean(r_disc_score)
            mean_f_disc_score = tf.reduce_mean(f_disc_score)
            mean_r_prob = tf.reduce_mean(r_probs)
            mean_f_prob = tf.reduce_mean(f_probs)
            disc_acc = tf.reduce_mean(tf.metrics.accuracy(
                tf.concat([r_preds, f_preds], axis=0), 
                tf.concat([tf.ones_like(r_preds), tf.zeros_like(f_preds)], axis=0)))
            tf.summary.scalar("disc_acc", disc_acc)
            tf.summary.scalar("disc_loss", disc_loss)
            tf.summary.scalar('mean_r_disc_score', mean_r_disc_score)
            tf.summary.scalar('mean_f_disc_score', mean_f_disc_score)
            tf.summary.scalar('mean_r_prob', mean_r_prob)
            tf.summary.scalar('mean_f_prob', mean_f_prob)
            tf.summary.scalar('disc_crit_rmse', tf.sqrt(disc_crit_loss))
            tf.summary.scalar('f_disc_crit_rmse', tf.sqrt(f_disc_crit_loss))
            tf.summary.scalar('r_disc_crit_rmse', tf.sqrt(r_disc_crit_loss))
            tf.summary.scalar('mean_r_disc_crit_baselines',tf.reduce_mean(r_disc_crit_baselines))
            tf.summary.scalar('mean_f_disc_crit_baselines', tf.reduce_mean(f_disc_crit_baselines))
            
            disc_summaries = tf.summary.merge_all(scope='disc_train')
            
        # Train Classifier Subgraph
        with g.name_scope("clas_train"):
            logger.info("Creating classifier training subgraph...")
            r_clas_q_logit, _, r_clas_cell_outputs = classifier(
                x_labeled_emb, sequence_length= label_seq_lengths, return_cell_output=True)
            f_clas_q_logit, _, f_clas_cell_outputs = classifier(
                fake_seq_emb, sequence_length = gen_lengths, return_cell_output=True)
            r_clas_score = tf.squeeze(tf.reduce_mean(r_clas_q_logit, axis=1, keepdims=False))
            f_clas_score = tf.squeeze(tf.reduce_mean(f_clas_q_logit, axis=1, keepdims=False))
            r_clas_qvalues = tf.math.sigmoid(r_clas_q_logit)
            f_clas_qvalues = tf.math.sigmoid(f_clas_q_logit)
            r_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=r_clas_score, 
                multi_class_labels = data_labels,
                reduction=tf.losses.Reduction.MEAN)
                # Have not implemented min-ent term here...
            f_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_clas_score, 
                multi_class_labels=tf.squeeze(tf.cast(random_classes, tf.float32)),
                reduction=tf.losses.Reduction.MEAN)
            clas_loss = r_clas_loss + f_clas_loss
            clas_loss.set_shape(())

            c_variables = tx.utils.collect_trainable_variables([ classifier])
            clas_train_op = tx.core.get_train_op(clas_loss,
                                                 variables=c_variables,
                                                 global_step=global_step,
                                                 increment_global_step=False,
                                                 hparams=config.c_opt_hparams)
            # Classifier critic
            r_clas_crit_inp = r_clas_cell_outputs[:, :-1]
            r_clas_crit_target = r_clas_qvalues[:, 1:]
            f_clas_crit_inp = f_clas_cell_outputs[:, :-1]
            f_clas_crit_target = f_clas_qvalues[:, 1:]
            r_clas_crit_baselines = clas_crit(r_clas_crit_inp)
            f_clas_crit_baselines = clas_crit(f_clas_crit_inp)
            r_clas_crit_loss = tf.losses.mean_squared_error(labels=r_clas_crit_target,
                                                          predictions=r_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            f_clas_crit_loss = tf.losses.mean_squared_error(labels=f_clas_crit_target,
                                                          predictions=f_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            clas_crit_loss = r_clas_crit_loss + config.clas_loss_lambda * f_clas_crit_loss
            clas_crit_train_op = tx.core.get_train_op(
                clas_crit_loss, global_step=global_step,
                variables=[clas_crit.trainable_variables],
                increment_global_step=False,
                hparams=config.c_crit_opt_hparams)
            
            # Need to get last baseline
            last_f_clas_baseline = clas_crit_layer(f_clas_cell_outputs[:, -1])
            full_f_clas_crit_baselines = tf.concat([
                f_clas_crit_baselines, tf.expand_dims(last_f_clas_baseline, axis=1)],
                axis=1)


            r_probs = tf.math.sigmoid(r_clas_score)
            f_probs = tf.math.sigmoid(f_clas_score)
            r_preds = tf.round(r_probs)
            f_preds = tf.round(f_probs)
            r_clas_acc = tf.reduce_mean(tf.metrics.accuracy(
                r_preds, data_labels))
            f_clas_acc = tf.reduce_mean(tf.metrics.accuracy(
                f_preds, random_classes))
            tf.summary.scalar('clas_loss', clas_loss)
            tf.summary.scalar('r_clas_loss', r_clas_loss)
            tf.summary.scalar('f_clas_loss', f_clas_loss)
            tf.summary.scalar('r_clas_acc', r_clas_acc)
            tf.summary.scalar('f_clas_acc', f_clas_acc)
            tf.summary.scalar('clas_crit_rmse', tf.sqrt(clas_crit_loss))
            tf.summary.scalar('f_clas_crit_rmse', tf.sqrt(r_clas_crit_loss))
            tf.summary.scalar('r_clas_crit_rmse', tf.sqrt(f_clas_crit_loss))
            tf.summary.scalar('mean_r_clas_crit_baselines',tf.reduce_mean(r_clas_crit_baselines))
            tf.summary.scalar('mean_f_clas_crit_baselines', tf.reduce_mean(f_clas_crit_baselines))
            clas_summaries = tf.summary.merge_all(scope='clas_train')
            
        # Validate clas summaries
        with g.name_scope('clas_val_sum'):
            tf.summary.scalar('val_clas_loss', clas_loss)
            tf.summary.scalar('val_r_clas_loss', r_clas_loss)
            tf.summary.scalar('val_f_clas_loss', f_clas_loss)
            tf.summary.scalar('val_r_clas_acc', tf.reduce_mean(r_clas_acc))
            tf.summary.scalar('val_f_clas_acc', tf.reduce_mean(f_clas_acc))
            val_clas_summaries = tf.summary.merge_all(scope='clas_val_sum')


        # Train diversity-promoting discriminator
        logger.info("Creating diversifier training subgraph...")
        with g.name_scope('div_train'):
            r_div_inp = x_emb_context[:, 1:, :]
            f_div_inp = fake_seq_emb
            random_context_reshape = tf.reshape(
                tf.tile(zero_context, [1, tf.shape(f_div_inp)[1]]),
                [-1, tf.shape(f_div_inp)[1], context_size])
            f_div_inp = tf.concat([f_div_inp, random_context_reshape], axis = -1)

            r_div_logits, _, r_div_cell_outputs = diversifier(
                r_div_inp, sequence_length=seq_lengths , return_cell_output=True)
            f_div_logits, _, f_div_cell_outputs = diversifier(
                f_div_inp, gen_lengths, return_cell_output=True)
            div_variables = tx.utils.collect_trainable_variables([diversifier])
            
            r_div_log_probs = tx.losses.sequence_sparse_softmax_cross_entropy(
                logits=r_div_logits,
                labels=y[:, 1:],
                sequence_length=seq_lengths ,
                average_across_batch=False,
                average_across_timesteps=False,
                sum_over_batch=False,
                sum_over_timesteps=False)
            f_div_log_probs = tx.losses.sequence_sparse_softmax_cross_entropy(
                logits=f_div_logits,
                labels=gen_sample_ids,
                sequence_length=gen_lengths,
                average_across_batch=False,
                average_across_timesteps=False,
                sum_over_batch=False,
                sum_over_timesteps=False)
            
            r_div_mean_lp = tx.losses.mask_and_reduce(r_div_log_probs, 
                                                      sequence_length = seq_lengths,
                                                      average_across_timesteps=True,
                                                      average_across_batch=True,
                                                      sum_over_batch=False,
                                                      sum_over_timesteps=False)
            f_div_mean_lp = tx.losses.mask_and_reduce(f_div_log_probs, 
                                                      sequence_length = gen_lengths,
                                                      average_across_timesteps=True,
                                                      average_across_batch=True,
                                                      sum_over_batch=False,
                                                      sum_over_timesteps=False)
            div_loss = f_div_mean_lp - r_div_mean_lp

            div_train_op = tx.core.get_train_op(div_loss,
                                                global_step=global_step,
                                                variables=div_variables,
                                                increment_global_step=False,
                                                hparams=config.div_opt_hparams)

            tf.summary.scalar('div_loss', div_loss)
            tf.summary.scalar('r_div_mean_lp', r_div_mean_lp)
            tf.summary.scalar('f_div_mean_lp', f_div_mean_lp)
            div_summaries = tf.summary.merge_all(scope='div_train')









        



        # Generator Policy Gradient Training
        with g.name_scope('pg_train'):
            logger.info("Creating policy gradient subgraph...")

            def blend(dscore, cscore):
                return dscore + config.lambda_q_blend * cscore
            rewards = blend(f_disc_q_logit, 
                            tf.where(tf.squeeze(tf.cast(random_classes, tf.bool)), 
                                    f_clas_q_logit,
                                    -f_clas_q_logit))
            # Critic baselines
            disc_baseline = full_f_disc_crit_baselines
            clas_baseline = full_f_clas_crit_baselines
            if config.discriminator_loss_lambda > 0:
                advantages = rewards - disc_baseline - config.lambda_q_blend * clas_baseline
            
            if config.diversity_only:
                rewards = tx.losses.discount_reward(tf.expand_dims(f_div_log_probs, axis = -1),
                                             sequence_length = gen_lengths,
                                             discount=config.diversity_discount,
                                             normalize=True)
                advantages = rewards

            if config.norm_advantages:
                advantages = tx.losses.discount_reward(tf.expand_dims(advantages, axis=-1), 
                                                       sequence_length=gen_lengths,
                                                       discount=1,
                                                       normalize=True)
            advantages = tf.squeeze(advantages)
            # Advantage clipping
            advantages = tf.clip_by_value(advantages, -config.adv_max_clip, config.adv_max_clip)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits,
                                                                       labels=gen_sample_ids)
            # Bound log probs away from zero
            log_probs = tf.clip_by_value(log_probs, config.min_log_prob, config.max_log_prob)

            pg_loss_full = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=False, 
                average_across_timesteps=False,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)
            pg_loss = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=True, 
                average_across_timesteps=True,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)
            #pg_loss_full = tf.clip_by_value(pg_loss_full, config.min_pg_loss, config.max_pg_loss)
            if config.pg_max_ent:
                pg_ent_loss = tx.losses.sequence_entropy_with_logits(
                    logits = gen_logits,
                    rank=3,
                    average_across_batch=True,
                    average_across_remaining=True)
                pg_loss = pg_loss - config.pg_max_ent_lambda * pg_ent_loss

            if config.mle_loss_in_pg:
                pg_loss = pg_loss + config.mle_loss_in_pg_lambda * loss_mle
            pg_variables = tx.utils.collect_trainable_variables([g_decoder])
            pg_train_op = tx.core.get_train_op(pg_loss,
                                               variables=pg_variables,
                                               global_step=global_step,
                                               increment_global_step=True,
                                               hparams=config.g_opt_pg_hparams)

            my_pg_loss_full =  tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=gen_logits, labels=gen_sample_ids) * advantages
            my_pg_loss = tf.reduce_mean(my_pg_loss_full)



            mean_reward = tf.reduce_mean(rewards)
            mean_adv = tf.reduce_mean(advantages)
            adv_sd = tf.reduce_mean(tf.square(advantages)) - tf.square(mean_adv)
            reward_sd = tf.reduce_mean(tf.square(rewards)) - tf.square(mean_reward)
            mean_log_prob= tf.reduce_mean(log_probs)
            max_log_prob = tf.reduce_max(log_probs)
            pg_loss_sd = tf.reduce_mean(tf.square(pg_loss_full)- tf.square(pg_loss))
            tf.summary.scalar('mean_reward', mean_reward)
            tf.summary.scalar('mean_adv', mean_adv)
            tf.summary.scalar('adv_sd', adv_sd)
            tf.summary.scalar('pg_loss', pg_loss)
            tf.summary.scalar('pg_loss_sd', pg_loss_sd)
            tf.summary.scalar('mean_logit_gen', mean_logit_gen)
            tf.summary.scalar('mean_log_prob', mean_log_prob)
            tf.summary.scalar('max_log_prob', max_log_prob)
            if config.pg_max_ent:
                tf.summary.scalar('pg_max_ent', pg_ent_loss)
            pg_summaries = tf.summary.merge_all(scope='pg_train')
            pg_summaries = tf.summary.merge([gen_sample_summaries, pg_summaries])
    # END GRAPH 
    
    # Epoch running
    def _g_run_epoch(sess, epoch, mode_string, writer):
        if mode_string == 'train' or mode_string == 'pretrain':
            iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            size = train_data.dataset_size()
        elif mode_string == 'val':
            iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = val_data.dataset_size()
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
        progbar = tf.keras.utils.Progbar(10000, 30, 1, 0.05)
        nexamples = 0
        total_loss = 0
        gen_step = 0
        if config.log_verbose_mle or config.log_verbose_rl:
            fl = fakelogger('{}/logs.txt'.format(config.log_dir))
        while True:
            try:
                log_mle = False
                log_rl = False
                start_time = time.time()
                if mode_string == 'pretrain':
                    fetches = {
                        'loss' : loss_mle,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'mle_train_op' : mle_train_op,
                        'summaries' : mle_summaries,

                    }
                    if  gen_step % config.steps_per_summary == 0:
                        fetches['summaries'] = mle_summaries
                        if config.log_verbose_mle:
                            log_mle = True
                            fetches['sentence'] = y[0, :]
                            fetches['logits'] = observed_logits[0, :]
                            fetches['full_cross_ent'] = loss_mle_full[0, :]
                    

                elif mode_string == 'train':
                    fetches = {
                        'mean_adv' : mean_adv,
                        'batch_size' : batch_size,
                        'mean_reward' : mean_reward,
                        'loss' : pg_loss,
                        'train_op' : pg_train_op,
                        'global_step' : global_step,
                    }
                    if  gen_step % config.steps_per_summary == 0:
                        fetches['summaries'] = pg_summaries
                        if config.log_verbose_rl:
                            log_rl = True
                            fetches['sentence'] = gen_sample_ids[40, :]
                            fetches['logits'] = observed_gen_logits[40, :]
                            fetches['log_probs'] = log_probs[40, :]
                            fetches['disc_q_logit'] = f_disc_q_logit[40, :]
                            fetches['clas_q_logit'] = f_clas_q_logit[40, :]
                            fetches['disc_crit'] = disc_baseline[40, :]
                            fetches['clas_crit'] = clas_baseline[40, :]
                            fetches['qvalues'] = rewards[40, :, 0]
                            fetches['advantages'] = advantages[40, :, 0]
                            fetches['pg_loss_full'] = pg_loss_full[40, :, 0]
                            fetches['prog'] = f_progress_vector[40, :, 0]
                    
                        
                elif mode_string == 'val':
                    fetches = {
                        'loss' : loss_mle,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'summaries' : val_mle_summaries
                    }
                    if  gen_step % config.steps_per_summary == 0:
                        fetches['summaries'] = val_mle_summaries
                feed_dict = {tx.global_mode(): modekey}
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['loss']

                if gen_step % config.steps_per_summary == 0:
                    writer.add_summary(rtns['summaries'], glob_step)
                if nexamples % 1000 == 0 and (mode_string == 'train' or mode_string == 'pretrain'):
                    writer.add_summary(sess.run(sample_text_summary), glob_step)
                    writer.add_summary(sess.run(original_text_summary), glob_step)
                if log_mle:
                    header = ['tkn', 'logit', 'crossent']
                    values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])), 
                              rtns['logits'].tolist(),
                              rtns['full_cross_ent'].tolist()]
                    print_out_array(header, values, fl)
                if log_rl:
                    header = ['tkn', 'logit', 'log_prob', 'Q_d', 'Q_c', 'V_d',
                              'V_c', 'Q', 'A', 'pgloss', 'prog']
                    values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])),
                              rtns['logits'].squeeze().tolist(),
                              rtns['log_probs'].squeeze().tolist(),
                              rtns['disc_q_logit'].squeeze().tolist(),
                              rtns['clas_q_logit'].squeeze().tolist(), 
                              rtns['disc_crit'].squeeze().tolist(), 
                              rtns['clas_crit'].squeeze().tolist(),
                              rtns['qvalues'].squeeze().tolist(),
                              rtns['advantages'].squeeze().tolist(),
                              rtns['pg_loss_full'].squeeze().tolist(),
                              rtns['prog'].squeeze().tolist(),
                             ]
                    final_line='mean_pg_loss: {:0.02f}'.format(loss)
                    print_out_array(header, values, fl, final_line)
                                   
                total_loss += loss * rtns['batch_size']
                nexamples += rtns['batch_size']
                gen_step += 1
                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 

                
                # FIXME
                if nexamples > 10000:
                    break

            except tf.errors.OutOfRangeError:
                break

        return total_loss/nexamples


    def _div_run_epoch(sess, epoch, mode_string, writer, div_step):
        if mode_string == 'train':
            iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            size = train_data.dataset_size()
        elif mode_string == 'val':
            iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = val_data.dataset_size()
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
        if config.log_verbose_mle or config.log_verbose_rl:
            fl = fakelogger('{}/logs.txt'.format(config.log_dir))
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        nexamples = 0
        total_loss = 0
        gen_step = 0
        while True:
            try:
                start_time = time.time()
                if mode_string == 'train':
                    fetches = {
                        'loss' : div_loss,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'train_op' : div_train_op,
                    }
                    
                    if div_step % config.steps_per_summary == 0:
                        fetches['summaries'] = div_summaries
                        fetches['fake_sentence'] = gen_sample_ids[0, :]
                        fetches['real_sentence'] = x[0, 1: ]
                        fetches['r_div_log_probs'] = r_div_log_probs[0, :]
                        fetches['f_div_log_probs'] = f_div_log_probs[0, :]
                        fetches['r_div_mean_lp'] = r_div_mean_lp
                        fetches['f_div_mean_lp'] = f_div_mean_lp
                
                feed_dict = {tx.global_mode(): modekey}    
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['loss']
                if div_step % config.steps_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'], glob_step  + div_step)
                    header = ['tkn', 'logprob']
                    r_values = [
                        list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                        rtns['r_div_log_probs'].squeeze().tolist()
                        ]
                    f_values = [
                        list(vocab.map_ids_to_tokens_py(rtns['fake_sentence'])),
                        rtns['f_div_log_probs'].squeeze().tolist()
                        ]
                    r_final_line = 'r_div_mean_lp: {:0.02f}'.format(rtns['r_div_mean_lp'])
                    f_final_line = 'f_div_mean_lp: {:0.02f}'.format(rtns['f_div_mean_lp'])
                    fl.debug('DIV REAL')
                    print_out_array(header, r_values, fl, r_final_line)
                    fl.debug('DIV FAKE')
                    print_out_array(header, f_values, fl, f_final_line)

                div_step += 1
                nexamples += rtns['batch_size']
                total_loss += loss * rtns['batch_size']
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 

            except tf.errors.OutOfRangeError:
                break
        
        return total_loss/nexamples, div_step



    def _d_run_epoch(sess, epoch, mode_string, writer, disc_step, nsteps_to_do=-1):
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
        nexamples = 0
        total_loss = 0
        total_acc = 0
        if mode_string == 'train' or mode_string == 'train_critic':
            modekey = tf.estimator.ModeKeys.TRAIN
            iterator.switch_to_train_data(sess)
            size = train_data.dataset_size()
        elif mode_string == 'val':
            modekey = tf.estimator.ModeKeys.EVAL
            iterator.switch_to_val_data(sess)
            size = val_data.dataset_size()
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        while True:
            try:
                start_time = time.time()
                if mode_string == 'train':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'train_op' : disc_train_op,
                        'crit_train_op' : disc_crit_train_op,
                        'disc_acc' : disc_acc,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'global_step' : global_step,
                        'batch_size' : batch_size,
                    }
                    if disc_step % config.steps_per_summary == 0:
                        fetches['summaries'] = disc_summaries
                if mode_string == 'val':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'disc_acc' : disc_acc,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if disc_step % config.steps_per_summary == 0:
                        fetches['summaries'] = disc_summaries
                if mode_string == 'train_critic':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'crit_train_op' : disc_crit_train_op,
                        'disc_acc' : disc_acc,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'global_step' : global_step,
                        'batch_size' : batch_size
                    }
                    if disc_step % config.steps_per_summary == 0:
                        fetches['summaries'] = disc_summaries
                feed_dict = {tx.global_mode(): modekey}    
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['disc_loss']
                r_loss = rtns['real_loss']
                f_loss = rtns['fake_loss']
                if disc_step % config.steps_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'], glob_step  + disc_step)
                disc_step += 1
                nexamples += rtns['batch_size']
                total_loss += loss * rtns['batch_size']
                total_acc += rtns['disc_acc'] * rtns['batch_size']

                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 
                nsteps_to_do = nsteps_to_do - 1
                if nsteps_to_do == 0:
                    break
            except tf.errors.OutOfRangeError:
                break
        
        return total_loss/nexamples, total_acc/nexamples, disc_step
    
    def _c_run_epoch(sess, epoch, mode_string, writer, clas_step):
        total_loss = 0
        total_acc = 0
        nexamples = 0
        if mode_string == 'train':
            modekey = tf.estimator.ModeKeys.TRAIN
            iterator.switch_to_train_data(sess)
            if config.use_unsup:
                unsup_iterator.switch_to_dataset(sess)
            size = train_data.dataset_size()
        elif mode_string == 'val':
            modekey = tf.estimator.ModeKeys.EVAL
            iterator.switch_to_val_data(sess)
            if config.use_unsup:
                unsup_iterator.switch_to_dataset(sess)
            size = val_data.dataset_size()
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        while True:
            try:
                start_time = time.time()
                if mode_string == 'train':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'train_op' : clas_train_op,
                        'crit_train_op' : clas_crit_train_op,
                        'clas_acc' : r_clas_acc,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config.steps_per_summary == 0:
                        fetches['summaries'] = clas_summaries
                if mode_string == 'val':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'clas_acc' : r_clas_acc,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config.steps_per_summary == 0:
                        fetches['summaries'] = val_clas_summaries
                    
                feed_dict = {tx.global_mode(): modekey}    
                rtns = sess.run(fetches, feed_dict = feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['clas_loss']
                r_loss = rtns['real_loss']
                f_loss = rtns['fake_loss']
                if  clas_step % config.steps_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'], glob_step + clas_step *epoch + clas_step)
                clas_step += 1
                nexamples += rtns['batch_size']
                total_loss += loss * rtns['batch_size']
                total_acc  += rtns['clas_acc'] * rtns['batch_size']

                # Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 
            
            except tf.errors.OutOfRangeError:
                break
        
        return total_loss/nexamples, total_acc/nexamples, clas_step


    
    # Begin training loop
    sess = tf.Session(graph=g)
    with sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        logger.info("Beginning data flow...")

        # Checkpoints
        checkpoint_dir = os.path.abspath(config.checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt-all-adv')
        checkpoint = tf.train.Saver()
        if config.restore_model:
            checkpoint.restore(sess, checkpoint_prefix)
            logger.info("Checkpoint restored from {}".format(checkpoint_dir))
        if config.clear_run_logs:
            logfiles = os.listdir(config.log_dir)
            [os.unlink(os.path.join(config.log_dir, f)) for f in logfiles]

        if config.save_trained_gen:
            gen_checkpoint_dir = os.path.abspath(config.gen_ckpt_dir)
            gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, 'ckpt-gen')
            gen_saver = tf.train.Saver()
        if config.load_trained_gen:
            gen_saver = tf.train.Saver()
            gen_saver.restore(sess, config.gen_ckpt_file)
            logger.info('Generator-only checkpoint restored from {}'.format(config.gen_ckpt_file))

        if config.bleu_test:
            target_list, hyp_list = [], []
            iterator.switch_to_val_data(sess)
            while True:
                try:
                    fetches = {'target_ids' : inp,
                               'hyp_ids' : gen_sample_ids
                              }
                    rtns = sess.run(fetches)
                    tar = vocab.map_ids_to_tokens_py(rtns['target_ids'][:, 1:-1].tolist()) # remove BOS
                    hyp = vocab.map_ids_to_tokens_py(rtns['hyp_ids'].tolist())
                    for t in tar:
                        ct = [w for w in t if not w in ['<PAD>', 'UNK', '<EOS>', '<BOS>']]
                        target_list.append(ct)
                    for h in hyp:
                        ch = [w for w in h if not w in ['<PAD>', 'UNK', '<EOS>', '<BOS>']]
                        hyp_list.append(ch)
                except tf.errors.OutOfRangeError:
                    break
            bleu = tx.evals.corpus_bleu_moses(
                [target_list],
                hyp_list,
                return_all = False)
            print(bleu)

            return

        # Summaries
        sum_writer = tf.summary.FileWriter(config.log_dir,graph=g, session=sess, flush_secs=30)
        with sum_writer:
            # Gen Pre-training
            logger.info("Starting generator pretraining...")
            min_gen_val_loss = 1e8
            patience = 0
            for e in range(config.g_pretrain_epochs):
                train_loss = _g_run_epoch(sess, e, 'pretrain', sum_writer)
                print('VALIDATE')
                val_loss = _g_run_epoch(sess, e, 'val', sum_writer)
                if val_loss < (min_gen_val_loss - config.es_tolerance):
                    min_gen_val_loss = val_loss
                    patience = 0
                    checkpoint.save(sess, checkpoint_prefix)
                    if config.save_trained_gen:
                        gen_saver.save(sess, gen_checkpoint_prefix)
                else:
                    patience += 1
                
                if patience > config.gen_patience:
                    break
            logger.info('Min Gen MLE val loss: {}'.format(min_gen_val_loss))

            #if config.train_lm_only:
            #    return




            # Disc Pretraining
            disc_step = 0
            de = 0
            logger.info("Starting discriminator pretraining...")
            for e in range(config.d_pretrain_epochs):
                train_loss, train_acc, disc_step = _d_run_epoch(
                    sess,de, 'train', sum_writer, disc_step)
                de += 1
                checkpoint.save(sess, checkpoint_prefix)
            logger.info('Discriminator critic pretraining...')
            for e in range(config.d_pretrain_critic_epochs):
                train_loss, train_acc, disc_step = _d_run_epoch(
                    sess, de, 'train_critic', sum_writer, disc_step)
                de += 1
                checkpoint.save(sess, checkpoint_prefix)

            # Div pretraining
            div_step = 0
            div_e = 0
            g_decoder_cell_weights = g_decoder.cell.get_weights()
            g_decoder_output_weights = g_decoder.output_layer.get_weights()
            diversifier.cell.set_weights(g_decoder_cell_weights)
            diversifier.output_layer.set_weights(g_decoder_output_weights)
            logger.info("Starting diversifier pretraining...")
            for e in range(config.div_pretrain_epochs):
                train_loss, div_step = _div_run_epoch(
                    sess, div_e, 'train', sum_writer, disc_step)
                div_e += 1
                checkpoint.save(sess, checkpoint_prefix)
            

            logger.info("Starting classifier pretraining...")
            min_clas_val_loss = 0
            val_acc = 0
            patience = 0
            clas_step = 0
            ce = 0
            for i in range(config.c_pretrain_epochs):
                train_loss, train_acc, clas_step = _c_run_epoch(sess, e, 'train', sum_writer, clas_step)
                print('VALIDATE')
                val_loss, val_acc, clas_step = _c_run_epoch(
                    sess, ce, 'val', sum_writer, clas_step)
                ce += 1
                if val_loss < (min_clas_val_loss - config.es_tolerance):
                    min_clas_val_loss = val_loss
                    checkpoint.save(sess, checkpoint_prefix)
                    patience = 0
                else:
                    patience += 1
                i += 1
                if patience > config.gen_patience:
                    break
            
            logger.info('Min Clas  val loss: {}, acc: {}'.format(min_clas_val_loss, val_acc))
            logger.info("Starting adversarial training...")
            for e in range(config.adversarial_epochs):
                cur_epoch = e + config.g_pretrain_epochs
                # Generator Train
                g_loss = _g_run_epoch(sess, cur_epoch, 'train', sum_writer) 
                val_loss = _g_run_epoch(sess, e, 'val', sum_writer)
                # Check discriminator loss
                _, check_disc_acc, disc_step = _d_run_epoch(
                    sess, de, 'val', sum_writer, disc_step, nsteps_to_do=2)
                checkpoint.save(sess, checkpoint_prefix + '-adv')
                while check_disc_acc < config.min_disc_pg_acc and not config.diversity_only:
                    d_loss, check_disc_acc, disc_step = _d_run_epoch(sess, de, 'train', sum_writer, disc_step)
                    de += 1
                    _, check_disc_acc, disc_step = _d_run_epoch(
                        sess, de, 'val', sum_writer, disc_step, nsteps_to_do=2)
                    checkpoint.save(sess, checkpoint_prefix + '-adv-disc')





                if config.diversity_only:
                    for e in range(5):
                        _, div_step=_div_run_epoch(sess, de, 'train', sum_writer, div_step)

                #c_loss,c_acc, _ = _c_run_epoch(sess, ce, 'train', sum_writer, clas_step)
                ce += 1
                #print('VALIDATE')
                #c_loss, c_acc, clas_step = _c_run_epoch(sess, e, 'val', sum_writer, clas_step)
                #checkpoint.save(sess, checkpoint_prefix)
            return


if __name__ == "__main__":
    tf.app.run(main=main)



    










            











    







             
