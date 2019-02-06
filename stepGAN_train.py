import tensorflow as tf
import texar as tx
import importlib
import numpy as np
import logging
import os
import sys
import time
from tensorflow.python import debug as tf_debug
import custom_helpers
from my_beam_search_decode import beam_search_decode

config = importlib.import_module('stepGAN_config_opspam')

class Generator(tf.keras.Model):
    """Generator wrapper for checkpointing"""
    def __init__(self, vocab_size, decoder_config):
        super(Generator, self).__init__()
        self.decoder = tx.modules.BasicRNNDecoder(vocab_size=vocab_size,
                                                  hparams=decoder_config)


class RNNDiscriminator(tf.keras.Model):
    """Discriminator wrapper"""
    def __init__(self, disc_config, dropout):
        super(RNNDiscriminator, self).__init__()
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams = disc_config['encoder'], cell_dropout_mode=dropout)
            

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




def get_size(sup_dataset, unsup_dataset=None):
    nbatches = sup_dataset.dataset_size() // sup_dataset.batch_size
    if unsup_dataset is not None:
        return sup_dataset.dataset_size() + (nbatches - 1) * unsup_dataset.batch_size
    else:
        return sup_dataset.dataset_size()




def bleu_test():
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
        global_mode = tx.global_mode()
        
        # Get data
        logger.info("Constructing graph...")
        train_data = get_dataset(config.train_lm_only, config.train_data)
        val_data = get_dataset(config.train_lm_only, config.val_data)
        test_data = get_dataset(config.train_lm_only, config.test_data)
        unsup_data = None
        if config.use_unsup :
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
        label_seq_lengths = data_batch['x_length']
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
        if config.disc_has_own_embedder:
            disc_embedder_model = Embedder(vocab_size, config.emb_hparams)
            disc_embedder = disc_embedder_model.embedder
            copy_embedder_weights = disc_embedder.embedding.assign(embedder.embedding)
        else:
            disc_embedder = embedder
        discriminator_dropout = tf.placeholder(dtype=tf.string)
        disc_model = RNNDiscriminator(config.disc_hparams, discriminator_dropout)
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
            inp_emb = embedder(inp)
            label_inp_emb = embedder(label_inp)

            x = inp[:, 0:(tf.shape(inp)[1] -2)]
            x_emb = embedder(x)
            y = inp[:, 1:(tf.shape(inp)[1])-1]
            y_onehot = tf.one_hot(y, vocab_size)

            x_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(x)[1]) # Trim non-ending sentences. 

            context_size = config.noise_size + 1
            zero_context = tf.zeros((batch_size, context_size))
            tiled_context = tf.reshape(
                tf.tile(zero_context, [1, tf.shape(x)[1]]), [-1, tf.shape(x)[1], context_size])
            x_emb_context = tf.concat([x_emb, tiled_context], axis = -1)
            
            outputs_mle, _, _ = g_decoder(
                initial_state=initial_state, 
                decoding_strategy='train_greedy',
                embedding=None,
                inputs=x_emb_context,
                sequence_length=x_lengths)
            
            logits_mle = outputs_mle.logits

            observed_logits = tf.zeros_like(y_onehot)
                #tf.reduce_sum(
                #tf.multiply(logits_mle, y_onehot), axis = -1) #elementwise

            loss_mle_full = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=y,
                logits=logits_mle,
                sequence_length=x_lengths,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False
                )
            loss_mle = tf.reduce_mean(loss_mle_full)
            g_variables = tx.utils.collect_trainable_variables([embedder, g_decoder])
            mle_optimizer = tx.core.get_optimizer(global_step=global_step,
                                                  hparams=config.g_opt_mle_hparams)
            mle_train_op = mle_optimizer.minimize(loss_mle,
                                                  global_step=global_step,
                                                  var_list= g_variables)
            mean_max_logit_mle = tf.reduce_mean(tf.reduce_max(logits_mle, axis = -1))
            mean_min_logit_mle = tf.reduce_mean(tf.reduce_min(logits_mle, axis = -1))
            mean_logit_mle = tf.reduce_mean(logits_mle)
            #median_logit_mle = tf.reduce_mean(
            #    tf.contrib.distributions.percentile(logits_mle, q=50, axis=-1))
            logit_sd_mle = tf.sqrt(tf.reduce_mean(tf.square(logits_mle)) - tf.square(mean_logit_mle))
            tf.summary.scalar('mean_logit_mle', mean_logit_mle)
            tf.summary.scalar("mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("mean_min_logit_mle", mean_min_logit_mle)
            #tf.summary.scalar('median_logit_mle', median_logit_mle)
            tf.summary.scalar('loss_mle', loss_mle)
            tf.summary.scalar('logit_sd_mle', logit_sd_mle)
            perplexity = tf.exp(loss_mle)
            tf.summary.scalar('perplexity', perplexity)
            if config.compute_grad_norms:
                tf.summary.scalar('mle_grad_norm',
                                  tf.linalg.global_norm(
                                      mle_optimizer.compute_gradients(loss_mle)[0]))

            mle_summaries = tf.summary.merge_all(scope='gen_mle')
        # MLE Validate summaries
        with g.name_scope('val_mle_summaries'):
            tf.summary.scalar('val_logit_sd_mle', logit_sd_mle)
            tf.summary.scalar("val_mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("val_mean_min_logit_mle", mean_min_logit_mle)
            tf.summary.scalar('val_mean_logit_mle', mean_logit_mle)
            tf.summary.scalar('val_loss_mle', loss_mle)
            tf.summary.scalar('val_perplexity', perplexity)
            val_mle_summaries = tf.summary.merge_all(scope='val_mle_summaries')



        # Generate subgraph
        with g.name_scope('gen_sample'):
            if config.annealing_length > 0:
                max_length = tf.Variable(config.annealing_length, dtype=tf.int32)
            else:
                max_length = tf.Variable(config.max_decoding_length_infer, dtype=tf.int32)
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
            context_helper = custom_helpers.ContextSampleEmbeddingHelper(
                embedder, random_vector, start_tokens, end_token, softmax_temperature)

            if config.use_beam_search:
                beam_width = config.beam_width
                def context_embedder(x):
                    raw_y = embedder(x)
                    context_size = config.noise_size + 1
                    rv = tf.reshape(tf.tile(random_vector, [1, beam_width]), 
                                    [-1, beam_width, context_size])

                    y = tf.concat([raw_y, rv], axis=-1)
                    return y
                gen_outputs, gen_state, gen_lengths = beam_search_decode(
                    g_decoder,
                    context_embedder,
                    initial_state=initial_state,
                    beam_width=beam_width,
                    vocab_size=vocab_size,
                    start_tokens = tf.tile([vocab.bos_token_id], [batch_size]),
                    end_token=end_token,
                    max_decoding_length = max_length)
                gen_sample_ids = gen_outputs.predicted_ids[:, :, 0] # take only best beam
                gen_lengths = gen_lengths[:, 0] # only take best beam
                gen_logits = gen_outputs.logits[:, :, 0, :]# only take best beam
            else:
                gen_outputs, _, gen_lengths = g_decoder(
                    helper = context_helper,
                    initial_state = initial_state,
                    max_decoding_length = max_length)
                gen_logits = gen_outputs.logits
                gen_sample_ids = gen_outputs.sample_id
            
            # Inefficient, use tf.gather
            observed_gen_logits = tf.zeros_like(gen_sample_ids)
            
            #tf.reduce_sum(
            #    tf.math.multiply(
            #        tf.one_hot(gen_sample_ids, vocab_size), gen_logits),
            #    axis=-1)
            
            mean_max_logit = tf.reduce_mean(tf.reduce_max(gen_logits, axis = -1))
            mean_min_logit = tf.reduce_mean(tf.reduce_min(gen_logits, axis = -1))
            median_logit = tf.reduce_mean(
                tf.contrib.distributions.percentile(gen_logits, q=50, axis=-1))
            mean_logit_gen = tf.reduce_mean(gen_logits)
            logit_sd_gen = tf.sqrt(tf.reduce_mean(tf.square(gen_logits)) -\
                                   tf.square(mean_logit_gen))
            mean_length = tf.reduce_mean(gen_lengths)
            max_gen_length = tf.reduce_max(gen_lengths)
            min_length = tf.reduce_min(gen_lengths)

            sample_text = vocab.map_ids_to_tokens(gen_sample_ids)
            sep = '' if config.use_char_sep else ' '
            sample_text = tf.reduce_join(sample_text, axis=-1, separator=sep)
            original_text = vocab.map_ids_to_tokens(inp)
            original_text = tf.reduce_join(original_text, axis=-1, separator=sep)

            tf.summary.scalar("mean_max_logit", mean_max_logit)
            tf.summary.scalar("mean_min_logit", mean_min_logit)
            tf.summary.scalar('median_logit', median_logit)
            tf.summary.scalar('logit_sd', logit_sd_gen)
            tf.summary.scalar('mean_length', mean_length)
            tf.summary.scalar('max_length', max_gen_length)
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
            real_seq = inp[:, 1:-1] # remove BOS EOS token as fake does not have.
            real_seq_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(real_seq)[1])
            real_inp = disc_embedder(real_seq)
            fake_inp = disc_embedder(fake_seq)

            fake_seq = fake_seq[:, :max_length]
            real_seq = real_seq[:, :max_length]
            real_inp = real_inp[:, :max_length, :]
            real_seq_lengths = tf.clip_by_value(real_seq_lengths, 0, tf.shape(real_inp)[1])
            fake_inp = fake_inp[:, :max_length, :]
            gen_lengths = tf.clip_by_value(gen_lengths, 0, tf.shape(fake_inp)[1])

            if config.add_sentence_progress:
                f_progress_vector = tf.ones_like(fake_inp)
                # Array of  [batch_size, tstep, 1] like 
                #  [[1, 2, 3, 4...]
                #   [1, 2, 3, 4...]]
                b_f = tf.shape(fake_inp)[0]
                t_f = tf.shape(fake_inp)[1]
                b_r = tf.shape(real_inp)[0]
                t_r = tf.shape(real_inp)[1]
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
                real_seq_lengths_reshape = tf.cast(tf.reshape(
                    tf.tile(real_seq_lengths, [t_r]),
                    [b_r, t_r, 1]), dtype=tf.float32)

                f_progress_vector = tf.ones_like(gen_lengths_reshape) -\
                    (tf.multiply(1/gen_lengths_reshape, f_nsteps))
                r_progress_vector = tf.ones_like(real_seq_lengths_reshape) -\
                    tf.multiply(1/real_seq_lengths_reshape , r_nsteps)
                f_progress_vector = tf.clip_by_value(f_progress_vector, 0, 1e8)
                r_progress_vector = tf.clip_by_value(r_progress_vector, 0, 1e8)
                real_inp = tf.concat([real_inp, r_progress_vector], axis = -1)
                fake_inp = tf.concat([fake_inp, f_progress_vector], axis = -1)
            
            #p = tf.print(tf.shape(real_inp), tf.shape(fake_inp), tf.shape(real_seq), 
            #             tf.shape(fake_seq), real_seq, fake_seq)

            r_disc_q_logit, _, r_disc_cell_outputs = discriminator(
                real_inp, sequence_length= real_seq_lengths, return_cell_output=True)
            f_disc_q_logit, _, f_disc_cell_outputs = discriminator(
                fake_inp, sequence_length = gen_lengths, 
                return_cell_output=True)
            r_disc_qvalues = tf.math.sigmoid(r_disc_q_logit)
            f_disc_qvalues = tf.math.sigmoid(f_disc_q_logit)
            if config.discriminator_random_stopping:
                r_u = tf.distributions.Uniform(low=1.0, high=tf.cast(real_seq_lengths, tf.float32))
                f_u = tf.distributions.Uniform(low=1.0, high=tf.cast(gen_lengths, tf.float32))
                r_stopping_indices = tf.squeeze(tf.round(r_u.sample(1)))
                f_stopping_indices = tf.squeeze(tf.round(f_u.sample(1)))
                r_disc_q_logit_sq = tf.squeeze(r_disc_q_logit)
                f_disc_q_logit_sq = tf.squeeze(f_disc_q_logit)
                r_disc_score = tx.losses.mask_and_reduce(r_disc_q_logit_sq,
                                                  r_stopping_indices,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
                f_disc_score = tx.losses.mask_and_reduce(f_disc_q_logit_sq,
                                                  f_stopping_indices,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
            else:
                r_disc_score = tx.losses.mask_and_reduce(r_disc_q_logit_sq,
                                                  real_seq_lengths,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
                f_disc_score = tx.losses.mask_and_reduce(f_disc_q_logit_sq,
                                                  gen_lengths,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)

            r_disc_score = tf.expand_dims(r_disc_score, 1)
            f_disc_score = tf.expand_dims(f_disc_score, 1)
            true_labs = tf.ones_like(r_disc_score)
            fake_labs = tf.zeros_like(f_disc_score)
            r_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits = r_disc_score,
                multi_class_labels=true_labs, 
                label_smoothing = config.disc_label_smoothing_epsilon,
                reduction=tf.losses.Reduction.MEAN)
            f_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_disc_score,
                multi_class_labels=fake_labs, 
                label_smoothing = config.disc_label_smoothing_epsilon,
                reduction=tf.losses.Reduction.MEAN)
            disc_loss = r_disc_loss + f_disc_loss
            disc_loss2 = -tf.reduce_mean(tf.log(tf.nn.sigmoid(r_disc_score)) + 
                                           tf.log(1 - tf.nn.sigmoid(f_disc_score)))
            if config.use_alt_disc_loss:
                disc_loss = disc_loss2
            disc_loss.set_shape(())
            if config.let_discriminator_train_embedder:
                d_variables = tx.utils.collect_trainable_variables([discriminator, disc_embedder])
            else:
                d_variables = tx.utils.collect_trainable_variables([discriminator])
            disc_optimizer = tx.core.get_optimizer(hparams=config.d_opt_hparams)

            disc_train_op = disc_optimizer.minimize(disc_loss,
                                                    var_list=d_variables)

            # Discriminator Critic
            r_disc_crit_inp = r_disc_cell_outputs[:, :-1]
            r_disc_crit_target = r_disc_q_logit[:, :]
            f_disc_crit_inp = f_disc_cell_outputs[:, :-1]
            f_disc_crit_target = f_disc_q_logit[:, :]
            r_disc_crit_baselines = disc_crit(r_disc_crit_inp)
            f_disc_crit_baselines = disc_crit(f_disc_crit_inp)
            # Initially have to predict value based on no input
            init_pred = tf.Variable(0, dtype=tf.float32)
            init_pred_tile = tf.reshape(
                tf.tile(tf.expand_dims(init_pred, 0), [batch_size]), [-1, 1, 1])
            r_disc_crit_baselines = tf.concat([init_pred_tile, r_disc_crit_baselines], axis = 1)
            f_disc_crit_baselines = tf.concat([init_pred_tile, f_disc_crit_baselines], axis = 1)
        
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
            disc_crit_optimizer = tx.core.get_optimizer(hparams=config.d_crit_opt_hparams)
            disc_crit_train_op = disc_crit_optimizer.minimize(disc_crit_loss,
                                                     var_list=[disc_crit.trainable_variables,
                                                               init_pred])


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
            tf.summary.scalar('mean_f_disc_crit_baselines', tf.reduce_mean(f_disc_crit_baselines))
            if not config.disc_crit_train_on_fake_only:
                tf.summary.scalar('r_disc_crit_rmse', tf.sqrt(r_disc_crit_loss))
                tf.summary.scalar('mean_r_disc_crit_baselines',tf.reduce_mean(r_disc_crit_baselines))
            
            disc_summaries = tf.summary.merge_all(scope='disc_train')
            
        # Train Classifier Subgraph
        with g.name_scope("clas_train"):
            logger.info("Creating classifier training subgraph...")
            if config.clas_has_own_embedder: 
                clas_emb_model = Embedder(vocab_size, config.emb_hparams)
                clas_embedder = clas_emb_model.embedder
                clas_copy_embedder_weights = clas_embedder.embedding.assign(embedder.embedding)

            # Designate clas input
            real_label_inp = label_inp[:, 1:-1]
            real_label_inp_emb = label_inp_emb[:, 1:-1, :]
            fake_label_inp_emb = clas_embedder(fake_seq)
            
            real_label_inp = real_label_inp[:, :max_length]
            real_label_inp_emb = real_label_inp_emb[:, :max_length, :]
            fake_label_inp_emb = fake_label_inp_emb[:, :max_length, :]
            label_seq_lengths = tf.clip_by_value(label_seq_lengths, 0, max_length)

            # Pass through classifier
            r_clas_q_logit, _, r_clas_cell_outputs = classifier(
                real_label_inp_emb, sequence_length= label_seq_lengths, return_cell_output=True)
            f_clas_q_logit, _, f_clas_cell_outputs = classifier(
                fake_label_inp_emb, sequence_length = gen_lengths, return_cell_output=True)

            # Random stopping
            if config.classifier_random_stopping:
                r_u = tf.distributions.Uniform(low=1.0, high=tf.cast(label_seq_lengths, tf.float32))
                f_u = tf.distributions.Uniform(low=1.0, high=tf.cast(gen_lengths, tf.float32))
                r_stopping_indices = tf.squeeze(tf.round(r_u.sample(1)))
                f_stopping_indices = tf.squeeze(tf.round(f_u.sample(1)))
                r_clas_score = tx.losses.mask_and_reduce(tf.squeeze(r_clas_q_logit),
                                                  r_stopping_indices,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
                f_clas_score = tx.losses.mask_and_reduce(tf.squeeze(f_clas_q_logit),
                                                  f_stopping_indices,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
            else:
                r_clas_score = tx.losses.mask_and_reduce(tf.squeeze(r_clas_q_logit),
                                                  real_seq_lengths,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)
                f_clas_score = tx.losses.mask_and_reduce(tf.squeeze(f_clas_q_logit),
                                                  gen_lengths,
                                                  average_across_batch=False,
                                                  average_across_timesteps=True,
                                                  sum_over_batch=False,
                                                  sum_over_timesteps=False)

            
            
            r_clas_qvalues = tf.math.sigmoid(r_clas_q_logit)
            f_clas_qvalues = tf.math.sigmoid(f_clas_q_logit)
            r_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=r_clas_score, 
                multi_class_labels = data_labels,
                reduction=tf.losses.Reduction.MEAN)
            f_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_clas_score, 
                multi_class_labels=tf.squeeze(tf.cast(random_classes, tf.float32)),
                reduction=tf.losses.Reduction.MEAN)
            if  config.clas_loss_on_fake_lambda > 0 :
                clas_loss =  r_clas_loss + config.clas_loss_on_fake_lambda * f_clas_loss
            else:
                clas_loss = r_clas_loss
            
            if config.clas_min_ent_lambda > 0 :
                f_clas_ent = tx.losses.sequence_entropy_with_logits(f_clas_q_logit,
                                                                    sequence_length = label_seq_lengths,
                                                                    average_across_batch=True,
                                                                    average_across_timesteps=True,
                                                                    sum_over_batch=False,
                                                                    sum_over_timesteps=False)
                clas_loss = clas_loss + config.clas_min_ent_lambda * f_clas_ent


                    

            clas_loss.set_shape(())

            if config.clas_has_own_embedder:
                c_variables = tx.utils.collect_trainable_variables([classifier, clas_embedder])
            else:
                c_variables = tx.utils.collect_trainable_variables([classifier])
            clas_optimizer = tx.core.get_optimizer(hparams=config.c_opt_hparams)
            clas_train_op = clas_optimizer.minimize(clas_loss,
                                                    var_list=c_variables)
            # Classifier critic
            r_clas_crit_inp = r_clas_cell_outputs[:, :-1]
            r_clas_crit_target = r_clas_q_logit[:, :]
            f_clas_crit_inp = f_clas_cell_outputs[:, :-1]
            f_clas_crit_target = f_clas_q_logit[:, :]
            r_clas_crit_baselines = clas_crit(r_clas_crit_inp)
            f_clas_crit_baselines = clas_crit(f_clas_crit_inp)
            init_clas_pred = tf.Variable(0, dtype=tf.float32)
            init_clas_pred_tile1 = tf.reshape(
                tf.tile(tf.expand_dims(init_clas_pred, 0), [tf.shape(r_clas_crit_inp)[0]]),
                       [-1, 1, 1])
            init_clas_pred_tile2 = tf.reshape(
                tf.tile(tf.expand_dims(init_clas_pred, 0), [tf.shape(f_clas_crit_inp)[0]]),
                        [-1, 1, 1])
            r_clas_crit_baselines = tf.concat([init_clas_pred_tile1, r_clas_crit_baselines], axis=1)
            f_clas_crit_baselines = tf.concat([init_clas_pred_tile2, f_clas_crit_baselines], axis=1)


            r_clas_crit_loss = tf.losses.mean_squared_error(labels=r_clas_crit_target,
                                                          predictions=r_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            f_clas_crit_loss = tf.losses.mean_squared_error(labels=f_clas_crit_target,
                                                          predictions=f_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            if config.clas_crit_train_on_fake_only:
                clas_crit_loss = f_clas_crit_loss
            else:
                clas_crit_loss = r_clas_crit_loss + f_clas_crit_loss

            clas_crit_optimizer = tx.core.get_optimizer(hparams=config.c_crit_opt_hparams)
            clas_crit_train_op = clas_crit_optimizer.minimize(clas_crit_loss,
                                                              var_list=[clas_crit.trainable_variables,
                                                                       init_clas_pred])

            

            r_probs = tf.math.sigmoid(r_clas_score)
            f_probs = tf.math.sigmoid(f_clas_score)
            r_preds = tf.round(r_probs)
            f_preds = tf.round(f_probs)
            r_clas_acc = tf.reduce_mean(tf.metrics.accuracy(
                r_preds, data_labels))
            f_clas_acc = tf.reduce_mean(tf.metrics.accuracy(
                f_preds, random_classes))
            r_clas_prec = tf.reduce_mean(tf.metrics.precision(
                r_preds, data_labels))
            f_clas_prec = tf.reduce_mean(tf.metrics.precision(
                f_preds, random_classes))
            r_clas_recl = tf.reduce_mean(tf.metrics.recall(
                r_preds, data_labels))
            f_clas_recl = tf.reduce_mean(tf.metrics.recall(
                f_preds, data_labels))
            r_f1 = 2 * (r_clas_prec * r_clas_recl)/(r_clas_prec + r_clas_recl)
            f_f1 = 2 * (f_clas_prec * f_clas_recl)/(f_clas_prec + f_clas_recl)
            tf.summary.scalar('clas_loss', clas_loss)
            tf.summary.scalar('r_clas_loss', r_clas_loss)
            tf.summary.scalar('f_clas_loss', f_clas_loss)
            tf.summary.scalar('r_clas_acc', r_clas_acc)
            tf.summary.scalar('f_clas_acc', f_clas_acc)
            tf.summary.scalar('r_clas_prec', r_clas_prec)
            tf.summary.scalar('f_clas_prec', f_clas_prec)
            tf.summary.scalar('r_clas_recl', r_clas_recl)
            tf.summary.scalar('f_clas_recl', f_clas_recl)
            tf.summary.scalar('r_f1', r_f1)
            tf.summary.scalar('f_f1', f_f1)
            tf.summary.scalar('clas_crit_rmse', tf.sqrt(clas_crit_loss))
            tf.summary.scalar('f_clas_crit_rmse', tf.sqrt(r_clas_crit_loss))
            tf.summary.scalar('mean_f_clas_crit_baselines', tf.reduce_mean(f_clas_crit_baselines))
            if not config.clas_crit_train_on_fake_only:
                tf.summary.scalar('r_clas_crit_rmse', tf.sqrt(f_clas_crit_loss))
                tf.summary.scalar('mean_r_clas_crit_baselines',tf.reduce_mean(r_clas_crit_baselines))
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
        if config.diversifier_loss_lambda:
            with g.name_scope('div_train'):
                tiled_context2 = tf.reshape(
                    tf.tile(
                        zero_context,
                        [1, tf.shape(real_inp)[1]]),
                    [-1, tf.shape(real_inp)[1], context_size])
                r_div_inp = tf.concat([real_inp, tiled_context2], axis = -1)

                f_div_inp = fake_inp
                random_context_reshape = tf.reshape(
                    tf.tile(zero_context, [1, tf.shape(f_div_inp)[1]]),
                    [-1, tf.shape(f_div_inp)[1], context_size])
                f_div_inp = tf.concat([f_div_inp, random_context_reshape], axis = -1)

                r_div_logits, _, r_div_cell_outputs = diversifier(
                    r_div_inp, sequence_length=real_seq_lengths , return_cell_output=True)
                f_div_logits, _, f_div_cell_outputs = diversifier(
                    f_div_inp, gen_lengths, return_cell_output=True)
                div_variables = tx.utils.collect_trainable_variables([diversifier])
                
                r_div_log_probs = tx.losses.sequence_sparse_softmax_cross_entropy(
                    logits=r_div_logits,
                    labels=y[:, 1:],
                    sequence_length=real_seq_lengths ,
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
                                                          sequence_length = real_seq_lengths,
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

                div_optimizer = tx.core.get_optimizer(hparams=config.div_opt_hparams)

                div_train_op = div_optimizer.minimize(div_loss,
                                                      var_list=div_variables)

                tf.summary.scalar('div_loss', div_loss)
                tf.summary.scalar('r_div_mean_lp', r_div_mean_lp)
                tf.summary.scalar('f_div_mean_lp', f_div_mean_lp)
                div_summaries = tf.summary.merge_all(scope='div_train')









        



        # Generator Policy Gradient Training
        with g.name_scope('pg_train'):
            logger.info("Creating policy gradient subgraph...")

            def blend(dscore, cscore, blend_factor):
                return dscore +  blend_factor * cscore
            # Convert gen logits to selected action log probs 
            p = tf.print(tf.shape(gen_logits), 
                         tf.shape(gen_sample_ids),
                         gen_logits, gen_sample_ids)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits,
                                                                           labels=gen_sample_ids)
            log_probs = -tf.clip_by_value(log_probs, config.min_log_prob, config.max_log_prob)

            # Critic baselines
            disc_baseline = f_disc_crit_baselines
            clas_baseline = f_clas_crit_baselines

            disc_rewards = f_disc_q_logit
            disc_rewards2 = -tf.squeeze(tf.log(tf.sigmoid(f_disc_q_logit)))
            if config.use_alt_disc_reward:
                disc_rewards = disc_rewards2


            if config.diversity_discount != 1:
                div_rewards  = tf.squeeze(f_div_log_probs)
                div_rewards = tx.losses.discount_reward(div_rewards, 
                                                       sequence_length=tf.squeeze(gen_lengths),
                                                       discount=config.diversity_discount,
                                                       normalize=True,
                                                       tensor_rank=2
                                                       )

            clas_rewards = tf.where(tf.squeeze(tf.cast(random_classes, tf.bool)),
                                    f_clas_q_logit, # Random class is 1
                                    -f_clas_q_logit) # Random class is 0
                                     

            if config.use_sigmoided_rewards:
                disc_rewards = tf.nn.sigmoid(disc_rewards)
                disc_baseline = tf.nn.sigmoid(disc_baseline)
                clas_rewards = tf.nn.sigmoid(clas_rewards)
                clas_baseline = tf.nn.sigmoid(clas_baseline)
            #advantages = tf.squeeze(tf.zeros_like(log_probs))
            #rewards = tf.squeeze(tf.zeros_like(log_probs))
            if config.classifier_loss_lambda >0:
                rewards = tf.nn.sigmoid(disc_rewards) + config.classifier_loss_lambda * tf.nn.sigmoid(clas_rewards)
                advantages = tf.nn.sigmoid(disc_rewards) - tf.nn.sigmoid(disc_baseline)  +\
                    config.classifier_loss_lambda * (tf.nn.sigmoid(clas_rewards) - tf.nn.sigmoid(clas_baseline))
            else:
                rewards = tf.nn.sigmoid(disc_rewards)
                advantages = rewards - tf.nn.sigmoid(disc_baseline)
            if config.reward_blending == 'additive' and False:
                if config.discriminator_loss_lambda > 0:
                    rewards = rewards + config.discriminator_loss_lambda * disc_rewards
                    advantages = advantages +  config.discriminator_loss_lambda *(disc_rewards - disc_baseline)
                
                if config.classifier_loss_lambda > 0 and False:
                    rewards = rewards + config.classifier_loss_lambda * clas_rewards
                    advantages = advantages + config.classifier_loss_lambda *  (clas_rewards - clas_baseline)

                if config.diversifier_loss_lambda > 0:
                    rewards = rewards + config.diversifier_loss_lambda * div_rewards
                    advantages = advantages + config.diversifier_loss_lambda * (div_rewards)
                
            if config.reward_blending == 'f1':
                rewards = tf.multiply(config.discriminator_loss_lambda * disc_rewards,
                                      config.classifier_loss_lambda * clas_rewards)
                rewards = tf.divide(rewards, (config.discriminator_loss_lambda * disc_rewards + 
                                              config.classifier_loss_lambda * clas_rewards))
                baseline = tf.multiply(config.discriminator_loss_lambda * disc_baseline,
                                       config.classifier_loss_lambda * clas_rewards)
                baseline = tf.divide(baseline, (config.discriminator_loss_lambda * disc_baseline + 
                                                config.classifier_loss_lambda * clas_baseline))
                advantages = rewards - baseline
            advantages = tf.squeeze(advantages)
            if config.norm_advantages:
                advantages = tx.losses.discount_reward(advantages, 
                                                       sequence_length=gen_lengths,
                                                       discount=0,
                                                       normalize=True,
                                                       tensor_rank=2)
            if config.linear_decay_pg_weights:
                steps = tf.reshape(
                    tf.tile(
                        tf.range(0.0, tf.cast(tf.shape(advantages)[1], dtype=tf.float32), dtype=tf.float32),
                        [tf.shape(advantages)[0]]),
                    [-1, tf.shape(advantages)[1]])
                alpha = tf.cast(tf.expand_dims(gen_lengths, 1), dtype=tf.float32) - steps
                alpha = tx.utils.mask_sequences(alpha, gen_lengths)
                advantages = tf.multiply(advantages, alpha)

            
            #p = tf.print(tf.shape(advantages))#, advantages[0, :], advantages[1, :])
            #advantages = advantages / config.advantage_var_reduc
            # Advantage clipping
            advantages = tf.clip_by_value(advantages, -config.adv_max_clip, config.adv_max_clip)

            pg_loss_full = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=False, 
                average_across_timesteps=False,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)


            #pg_loss = tx.losses.mask_and_reduce(pg_loss_full, 
            #                                   sequence_length = gen_lengths,
            #                                   average_across_batch = True,
            #                                   average_across_timesteps = True,
            #                                   sum_over_batch = False,
            #                                   sum_over_timesteps = False)
            pg_loss = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=True, 
                average_across_timesteps=True,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)
            if config.pg_max_ent_lambda > 0:
                pg_ent_loss = tx.losses.sequence_entropy_with_logits(
                    logits = gen_logits,
                    rank=3,
                    average_across_batch=True,
                    average_across_remaining=True)
                pg_loss = pg_loss - config.pg_max_ent_lambda * pg_ent_loss



            pg_loss.set_shape(())
            pg_variables = tx.utils.collect_trainable_variables([g_decoder])
            pg_optimizer = tx.core.get_optimizer(global_step=global_step,
                                                 hparams=config.g_opt_pg_hparams)

            pg_train_op = pg_optimizer.minimize(pg_loss,
                                                global_step=global_step,
                                                var_list=pg_variables)

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
            if config.pg_max_ent_lambda > 0:
                tf.summary.scalar('pg_max_ent', pg_ent_loss)
            pg_summaries = tf.summary.merge_all(scope='pg_train')
            pg_summaries = tf.summary.merge([gen_sample_summaries, pg_summaries])
    # END GRAPH 
    
    # Epoch running
    def gen_run_epoch(sess, mode_string, writer):
        if mode_string == 'train' or mode_string == 'pretrain':
            iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            size = get_size(train_data, unsup_data)
            bs = train_data.batch_size
        elif mode_string == 'val':
            iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = get_size(val_data, unsup_data)
            bs = val_data.batch_size
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
            bs += unsup_data.batch_size
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
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
                    if  gen_step % config.batches_per_summary == 0:
                        fetches['summaries'] = mle_summaries
                    if gen_step % config.batches_per_text_summary == 0 and config.log_verbose_mle:
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
                    if  gen_step % config.batches_per_summary == 0:
                        fetches['summaries'] = pg_summaries
                    if gen_step % config.batches_per_text_summary == 0 and config.log_verbose_rl:
                        log_rl = True
                        fetches['sentence'] = gen_sample_ids[40, :]
                        fetches['logits'] = observed_gen_logits[40, :]
                        fetches['log_probs'] = log_probs[40, :]
                        fetches['disc_reward'] = disc_rewards[40, :]
                        fetches['clas_reward'] = clas_rewards[40, :]
                        fetches['disc_crit'] = disc_baseline[40, :]
                        fetches['clas_crit'] = clas_baseline[40, :]
                        fetches['qvalues'] = rewards[40, :]
                        fetches['advantages'] = advantages[40, :]
                        fetches['pg_loss_full'] = pg_loss_full[40, :]
                    
                elif mode_string == 'val':
                    fetches = {
                        'loss' : loss_mle,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'summaries' : val_mle_summaries
                    }
                    if  gen_step % config.batches_per_summary == 0:
                        fetches['summaries'] = val_mle_summaries

                feed_dict = {global_mode : modekey,
                             discriminator_dropout : tf.estimator.ModeKeys.PREDICT}
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['loss']

                # Summaries
                if gen_step % config.batches_per_summary == 0:
                    writer.add_summary(rtns['summaries'], glob_step)
                if gen_step % config.batches_per_text_summary == 0:
                    writer.add_summary(sess.run(sample_text_summary), glob_step)
                    writer.add_summary(sess.run(original_text_summary), glob_step)
                    # Write verbose Summaries
                    if mode_string == 'pretrain' and config.log_verbose_mle:
                        header = ['tkn', 'logit', 'crossent']
                        values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])), 
                                  rtns['logits'].tolist(),
                                  rtns['full_cross_ent'].tolist()]
                        print_out_array(header, values, fl)
                    if mode_string == 'train' and config.log_verbose_rl:
                        header = ['tkn', 'logit', 'log_prob', 'Q_d', 'Q_c', 'V_d',
                                  'V_c', 'Q', 'A', 'pgloss']
                        values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])),
                                  rtns['logits'].squeeze().tolist(),
                                  rtns['log_probs'].squeeze().tolist(),
                                  rtns['disc_reward'].squeeze().tolist(),
                                  rtns['clas_reward'].squeeze().tolist(), 
                                  rtns['disc_crit'].squeeze().tolist(), 
                                  rtns['clas_crit'].squeeze().tolist(),
                                  rtns['qvalues'].squeeze().tolist(),
                                  rtns['advantages'].squeeze().tolist(),
                                  rtns['pg_loss_full'].squeeze().tolist(),
                                 ]
                        final_line='mean_pg_loss: {:0.02f}'.format(loss)
                        print_out_array(header, values, fl, final_line)
                                   
                total_loss += loss * bs
                nexamples += bs
                gen_step += 1
                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 

                if nexamples > size:
                    # Ward off this weird error
                    raise(tf.errors.OutOfRangeError)

            except tf.errors.OutOfRangeError:
                break

        return {'loss' : total_loss/nexamples}


    def div_run_epoch(sess,  mode_string, writer, div_step):
        if mode_string == 'train':
            iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            if config.use_unsup:
                size = get_size(train_data, unsup_data)
            else: 
                size = get_size(train_data)
        elif mode_string == 'val':
            iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            if config.use_unsup:
                size = get_size(val_data, unsup_data)
            else:
                size = get_size(val_data)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            if config.use_unsup:
                size = get_size(test_data, unsup_data)
            else:
                size = get_size(test_data)
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
        if config.log_verbose_mle or config.log_verbose_rl:
            fl = fakelogger('{}/logs.txt'.format(config.log_dir))
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        nexamples = 0
        total_loss = 0
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
                    if div_step % config.batches_per_summary == 0:
                        fetches['summaries'] = div_summaries
                    if div_step % config.batches_per_text_summary == 0:
                        fetches['fake_sentence'] = gen_sample_ids[0, :]
                        fetches['real_sentence'] = x[0, 1: ]
                        fetches['r_div_log_probs'] = r_div_log_probs[0, :]
                        fetches['f_div_log_probs'] = f_div_log_probs[0, :]
                        fetches['r_div_mean_lp'] = r_div_mean_lp
                        fetches['f_div_mean_lp'] = f_div_mean_lp
                if mode_string == 'val' or 'test':
                    fetches = {
                        'loss' : div_loss,
                        'batch_size' : batch_size,
                        'global_step' : global_step
                    }
                    if div_step % config.batches_per_summary == 0:
                        fetches['summaries'] = div_summaries
                    if div_step % config.batches_per_text_summary == 0:
                        fetches['fake_sentence'] = gen_sample_ids[0, :]
                        fetches['real_sentence'] = x[0, 1: ]
                        fetches['r_div_log_probs'] = r_div_log_probs[0, :]
                        fetches['f_div_log_probs'] = f_div_log_probs[0, :]
                        fetches['r_div_mean_lp'] = r_div_mean_lp
                        fetches['f_div_mean_lp'] = f_div_mean_lp
                
                feed_dict = {global_mode: modekey}    
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['loss']
                if div_step % config.batches_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'],  div_step)
                if div_step % config.batches_per_text_summary == 0:
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
                nexamples += bs
                total_loss += loss * bs
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 

            except tf.errors.OutOfRangeError:
                break
        
        return {'loss' : total_loss/nexamples, 'step' : div_step}



    def disc_run_epoch(sess, mode_string, writer, disc_step):
        if mode_string == 'train' or mode_string == 'train_critic':
            iterator.switch_to_train_data(sess)
            bs = train_data.batch_size
            modekey = tf.estimator.ModeKeys.TRAIN
            if config.use_unsup:
                size = get_size(train_data, unsup_data)
                bs += unsup_data.batch_size
            else: 
                size = get_size(train_data)
        elif mode_string == 'val':
            iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            bs = val_data.batch_size
            if config.use_unsup:
                bs += unsup_data.batch_size
                size = get_size(val_data, unsup_data)
            else:
                size = get_size(val_data)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)
            bs = test_data.batch_size
            modekey = tf.estimator.ModeKeys.EVAL
            if config.use_unsup:
                bs += unsup_data.batch_size
                size = get_size(test_data, unsup_data)
            else:
                size = get_size(test_data)
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess)
        if config.log_verbose_mle or config.log_verbose_rl:
            fl = fakelogger('{}/logs.txt'.format(config.log_dir))

        nexamples = 0
        total_loss = 0
        total_acc = 0
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
                    if disc_step % config.batches_per_summary == 0:
                        fetches['summaries'] = disc_summaries
                    if disc_step % config.batches_per_text_summary == 0:
                        fetches['fake_sentence'] = fake_seq[0, :]
                        fetches['real_sentence'] = real_seq[0, : ]
                        fetches['r_disc_q_logit'] = r_disc_q_logit[0, :]
                        fetches['f_disc_q_logit'] = f_disc_q_logit[0, :]
                        fetches['r_disc_score'] = r_disc_score[0, 0]
                        fetches['f_disc_score'] = f_disc_score[0, 0]
                        fetches['r_disc_loss']  = r_disc_loss
                        fetches['f_disc_loss']  = f_disc_loss
                        fetches['disc_baseline'] =  disc_baseline[0, :]
                        
                if mode_string == 'val':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'disc_acc' : disc_acc,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if disc_step % config.batches_per_summary == 0:
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
                    if disc_step % config.batches_per_summary == 0:
                        fetches['summaries'] = disc_summaries
                feed_dict = {global_mode : modekey,
                             discriminator_dropout : tf.estimator.ModeKeys.TRAIN}
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['disc_loss']
                r_loss = rtns['real_loss']
                f_loss = rtns['fake_loss']

                if disc_step % config.batches_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'], disc_step)
                if disc_step % config.batches_per_text_summary == 0 and mode_string=='train':
                    r_header = ['tkn', 'logit']
                    f_header = ['tkn', 'logit', 'v_d']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_disc_q_logit'].squeeze().tolist()
                               ]
                    f_values = [list(vocab.map_ids_to_tokens_py(rtns['fake_sentence'])),
                                rtns['f_disc_q_logit'].squeeze().tolist(),
                                rtns['disc_baseline'].squeeze().tolist()
                               ]
                    r_final_line = 'r_disc_loss: {:0.02f} r_disc_score: {:0.02f}'.format(
                        rtns['r_disc_loss'], rtns['r_disc_score'])
                    f_final_line = 'f_disc_loss: {:0.02f} f_disc_score: {:0.02f}'.format(
                        rtns['f_disc_loss'], rtns['f_disc_score'])
                    fl.debug('REAL SENTENTCE')
                    print_out_array(r_header, r_values, fl, r_final_line)
                    fl.debug('FAKE SENTENCE')
                    print_out_array(f_header, f_values, fl, f_final_line)



                disc_step += 1
                nexamples += bs
                total_loss += loss * bs
                total_acc += rtns['disc_acc'] * bs

                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 
            except tf.errors.OutOfRangeError:
                break
        
        return {'loss' : total_loss/nexamples, 'acc' : total_acc/nexamples, 'step' : disc_step}
    
    def clas_run_epoch(sess, mode_string, writer, clas_step):
        total_loss = 0
        total_real_acc = 0
        total_fake_acc = 0
        nexamples = 0
        if mode_string == 'train':
            bs = train_data.batch_size
            modekey = tf.estimator.ModeKeys.TRAIN
            iterator.switch_to_train_data(sess)
            size = get_size(train_data)
        elif mode_string == 'val':
            bs = val_data.batch_size
            modekey = tf.estimator.ModeKeys.EVAL
            iterator.switch_to_val_data(sess)
            size = get_size(val_data)
        elif mode_string == 'test':
            s = test_data.batch_size
            modekey = tf.estimator.ModeKeys.EVAL
            iterator.switch_to_test_data(sees)
            size = get_size(test_data)
        if config.use_unsup:
            unsup_iterator.switch_to_dataset(sess) # although not used, must init.
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        if config.log_verbose_mle or config.log_verbose_rl:
            fl = fakelogger('{}/logs.txt'.format(config.log_dir))
        while True:
            try:
                start_time = time.time()
                if mode_string == 'train':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'train_op' : clas_train_op,
                        'crit_train_op' : clas_crit_train_op,
                        'r_clas_acc' : r_clas_acc,
                        'f_clas_acc' : f_clas_acc,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config.batches_per_summary == 0:
                        fetches['summaries'] = clas_summaries
                    if clas_step % config.batches_per_text_summary == 0:
                        fetches['fake_sentence'] = fake_seq[0, :]
                        fetches['real_sentence'] = real_label_inp[0, : ]
                        fetches['r_clas_q_logit'] = r_clas_q_logit[0, :]
                        fetches['f_clas_q_logit'] = f_clas_q_logit[0, :]
                        fetches['r_clas_score'] = r_clas_score[0]
                        fetches['f_clas_score'] = f_clas_score[0]
                        fetches['r_clas_loss']  = r_clas_loss
                        fetches['f_clas_loss']  = f_clas_loss
                        fetches['clas_baseline'] =  clas_baseline[0, :]
                        fetches['fake_class'] = random_classes[0]
                        fetches['real_class'] = data_labels[0]

                if mode_string == 'val' or mode_string == 'test':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'r_clas_acc' : r_clas_acc,
                        'f_clas_acc' : f_clas_acc,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config.batches_per_summary == 0:
                        fetches['summaries'] = val_clas_summaries
                
                feed_dict = {global_mode: modekey}
                rtns = sess.run(fetches, feed_dict = feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['clas_loss']
                r_loss = rtns['real_loss']
                f_loss = rtns['fake_loss']
                if  clas_step % config.batches_per_summary == 0:
                    writer.add_summary(
                        rtns['summaries'], clas_step)
                if clas_step % config.batches_per_text_summary == 0 and mode_string == 'train':
                    r_header = ['tkn', 'logit']
                    f_header = ['tkn', 'logit', 'V_c']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_clas_q_logit'].squeeze().tolist()
                               ]
                    f_values = [list(vocab.map_ids_to_tokens_py(rtns['fake_sentence'])),
                                rtns['f_clas_q_logit'].squeeze().tolist(),
                                rtns['clas_baseline'].squeeze().tolist()
                               ]
                    fl.debug('REAL SENTENTCE')
                    r_final_line = 'class: {} r_disc_loss: {:0.02f} r_disc_score: {:0.02f}'.format(
                        rtns['real_class'], rtns['r_clas_loss'], rtns['r_clas_score'])
                    f_final_line = 'class: {} f_disc_loss: {:0.02f} f_disc_score: {:0.02f}'.format(
                        rtns['fake_class'], rtns['f_clas_loss'], rtns['f_clas_score'])
                    print_out_array(r_header, r_values, fl, r_final_line)
                    fl.debug('FAKE SENTENCE')
                    print_out_array(f_header, f_values, fl, f_final_line)

                clas_step += 1
                nexamples += rtns['batch_size']
                total_loss += loss * rtns['batch_size']
                total_real_acc  += rtns['r_clas_acc'] * rtns['batch_size']
                total_fake_acc += rtns['f_clas_acc'] * rtns['batch_size']

                # Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                progbar.update(nexamples,
                               [('loss', loss), ('batch_time', per_step_time)]) 
            
            except tf.errors.OutOfRangeError:
                break
        
        return {'loss' : total_loss/nexamples, 'real_acc' : total_real_acc/nexamples,
                'fake_acc' : total_fake_acc/nexamples, 'step' : clas_step}

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True) 
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
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt-all')
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

        if config.load_trained_gen and not config.restore_model:
            gen_saver = tf.train.Saver()
            gen_saver.restore(sess, config.gen_ckpt_file)
            logger.info('Generator-only checkpoint restored from {}'.format(config.gen_ckpt_file))


        # Summaries
        sum_writer = tf.summary.FileWriter(config.log_dir,graph=g, session=sess, flush_secs=30)
        with sum_writer:
            # Check if testing
            if config.clas_test:
                outs = clas_run_epoch(sess, 'test', sum_writer, 0)
                logger.info(outs)
                return


            # Gen Pre-training
            logger.info("Starting generator pretraining...")
            min_gen_val_loss = 1e8
            patience = 0
            for e in range(config.g_pretrain_epochs):
                logger.info('\n Gen Pretrain Epoch {}'.format(e))
                gen_rtns = gen_run_epoch(sess, 'pretrain', sum_writer)
                logger.info('\n Gen Validate Pretrain Epoch {}'.format(e))
                gen_rtns = gen_run_epoch(sess, 'val', sum_writer)

                # Early Stopping
                if gen_rtns['loss'] < (min_gen_val_loss - config.gen_es_tolerance):
                    min_gen_val_loss = gen_rtns['loss']
                    patience = 0
                    checkpoint.save(sess, checkpoint_prefix)
                    if config.save_trained_gen:
                        gen_saver.save(sess, gen_checkpoint_prefix)
                else:
                    patience += 1
                
                if patience > config.gen_patience:
                    logger.info("\n Gen Early Stopping Reached at val loss {:0.02f}".format(min_gen_val_loss))
                    break
            logger.info('Min Gen MLE val loss: {}'.format(min_gen_val_loss))

            #if config.train_lm_only:
            #    return

            # Keep track of these separate from global_step
            # Disc Pretraining
            logger.info("Starting discriminator pretraining...")
            if config.disc_has_own_embedder:
                sess.run(copy_embedder_weights)
            disc_rtns = {'step' : 0}
            for e in range(config.d_pretrain_epochs):
                logger.info('\n Disc Pretrain Epoch {} '.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'train', sum_writer, disc_rtns['step'])
                logger.info('\n Disc Val Epoch {} '.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'val', sum_writer, disc_rtns['step'])
                checkpoint.save(sess, checkpoint_prefix)
            logger.info('\n Discriminator critic pretraining...')
            for e in range(config.d_pretrain_critic_epochs):
                logger.info('\n Disc-Crit Pretrain Epoch {}'.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'train_critic', sum_writer, disc_rtns['step'])
                checkpoint.save(sess, checkpoint_prefix)

            # Div pretraining
            logger.info('Copying cell weights into diversifier...')
            g_decoder_cell_weights = g_decoder.cell.get_weights()
            g_decoder_output_weights = g_decoder.output_layer.get_weights()
            #diversifier.cell.set_weights(g_decoder_cell_weights)
            #diversifier.output_layer.set_weights(g_decoder_output_weights)
            logger.info("Starting diversifier pretraining...")
            div_rtns = {'step' : 0}
            for e in range(config.div_pretrain_epochs):
                logger.info('Div Pretrain Epoch {}'.format(e))
                div_rtns = div_run_epoch(
                    sess, 'train', sum_writer, div_rtns['step'])
                checkpoint.save(sess, checkpoint_prefix)
            

            logger.info("Starting classifier pretraining...")
            min_clas_val_loss = 1e8
            patience = 0
            clas_rtns = {'step' : 0, 'real_acc': 0}
            if config.clas_has_own_embedder:
                sess.run(clas_copy_embedder_weights)
            for e in range(config.c_pretrain_epochs):
                logger.info('\nClas Pretrain Epoch {}'.format(e))
                clas_rtns = clas_run_epoch(sess, 'train', sum_writer, clas_rtns['step'])
                print('\n Clas Validate Pretrain Epoch {}'.format(e))
                clas_rtns = clas_run_epoch(
                    sess, 'val', sum_writer, clas_rtns['step'])
                checkpoint.save(sess, checkpoint_prefix)
                if clas_rtns['loss'] < (min_clas_val_loss - config.clas_es_tolerance):
                    min_clas_val_loss = clas_rtns['loss']
                    patience = 0
                else:
                    patience += 1
                if patience > config.clas_patience:
                    logger.info('Clas Early Stopping reached at {:0.02f}'.format(clas_rtns['loss']))
                    break
            
            logger.info('Min Clas  val loss: {}, acc: {}'.format(min_clas_val_loss, clas_rtns['real_acc']))
            logger.info("Starting adversarial training...")
            prev_gen_val = 1e8
            extra_disc = False
            if config.adversarial_length > 0:
                max_length.load(config.adversarial_length, sess)
                for e in range(1):
                    disc_rtns = disc_run_epoch(
                        sess, 'train', sum_writer, disc_rtns['step'])
                    clas_rtns = clas_run_epoch(
                        sess, 'train', sum_writer, clas_rtns['step'])
                    checkpoint.save(sess, checkpoint_prefix)
                    
            for e in range(config.adversarial_epochs):
                cur_epoch = e + config.g_pretrain_epochs
                # Generator Train
                logger.info('\nGen Adv-Train Epoch {}'.format(cur_epoch))
                gen_rtns = gen_run_epoch(sess, 'train', sum_writer) 
                if config.mle_loss_in_adv:
                    logger.info('\nGen Adv-MLE Train Epoch{}'.format(cur_epoch))
                    gen_rtns = gen_run_epoch(sess, 'pretrain', sum_writer)
                # Save
                checkpoint.save(sess, checkpoint_prefix +( '-adv'))
                
                # Check discriminator loss
                if config.discriminator_loss_lambda > 0:
                    logger.info('\nDisc Adv-Valid Epoch {}'.format(cur_epoch))
                    disc_rtns = disc_run_epoch(
                        sess, 'val', sum_writer, disc_rtns['step'])
                
                # Train Disc
                disc_e = 0
                while config.discriminator_loss_lambda > 0 and disc_e < config.disc_adv:
                    logger.info('\nDisc Adv-Train Epoch: {}+{}'.format(cur_epoch, disc_e))
                    disc_rtns = disc_run_epoch(sess, 'train', sum_writer, disc_rtns['step'])
                    disc_rtns = disc_run_epoch(sess, 'val', sum_writer, disc_rtns['step'])
                    checkpoint.save(sess, checkpoint_prefix + '-adv')
                    disc_e += 1
                
                # Generator validate
                logger.info('\nGen Adv-Valid Epoch {}'.format(cur_epoch))
                gen_rtns = gen_run_epoch(sess, 'val', sum_writer)
                if gen_rtns['loss'] < prev_gen_val:
                    prev_gen_val = gen_rtns['loss']
                    extra_disc = False
                else:
                    extra_disc = True

                


                # Check Div Loss
                if config.diversifier_loss_lambda > 0:
                    logger.info('\nDiv Adv-Val Epoch {}'.format(cur_epoch))
                    div_rtns = div_run_epoch(sess, 'val', sum_writer, div_rtns['step'])
                # Train Div
                div_e = 0
                while config.diversifier_loss_lambda >0  and div_rtns['loss'] > config.max_div_pg_loss:
                    logger.info('\nDiv Adv-Train Epoch {}+{}'.format(cur_epoch, div_e))
                    div_rtns = div_run_epoch(sess, 'train', sum_writer, div_rtns['step'])
                    div_e += 1
                    checkpoint.save(sess, checkpoint_prefix + '-adv')
                    if div_e > config.max_extra_div_adv_epochs:
                        logger.info('Reached extra div epoch limit at loss: {}'.format(div_rtns['loss']))
                        break

                # Check Clas Acc
                if config.classifier_loss_lambda > 0:
                    logger.info('\nClas Adv-Val Epoch {}'.format(cur_epoch))
                    clas_rtns = clas_run_epoch(sess, 'val', sum_writer, clas_rtns['step'])
                # Train Clas
                clas_e = 0
                while config.classifier_loss_lambda > 0 and clas_e < config.clas_adv: 
                    logger.info('\nClas Adv-Train Epoch {}+{}'.format(cur_epoch, clas_e))
                    clas_rnts = clas_run_epoch(sess, 'train', sum_writer, clas_rtns['step'])
                    checkpoint.save(sess, checkpoint_prefix + '-adv')
                    clas_e += 1



if __name__ == "__main__":
    tf.app.run(main=main)



    










            











    







             
