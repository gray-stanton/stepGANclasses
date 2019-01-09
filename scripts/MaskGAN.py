def rec_layer(use_cudnn):
    if use_cudnn:
        return layers.CuDNNGRU
    else:
        return layers.GRU

class MaskGAN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_rnn_size,
                 use_cudnn = False):
        self.vocab_size = vocab_size
        mask_zero = not use_cudnn # CUDNN doesn't support masking
        # Construct seq2seq generator model
        # Same embedding used for both encoder and decoder on both gen and disc
        self.embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          mask_zero=mask_zero)
        #TODO Both need Variational dropout, Decoder needs attention
        self.gen_enc_rnn = rec_layer(use_cudnn)(gen_enc_rnn_size,
                                                  return_sequences=True,
                                                  return_state = True,
                                                  stateful = False)
        self.gen_dec_rnn = rec_layer(use_cudnn)(gen_dec_rnn_size,
                                                  return_sequences = True,
                                                  return_state = True,
                                                  stateful=False)
        #TODO Use weight tying between embedding and dense output (transpose)
        # Also ensure time distributed
        self.token_logits = layers.Dense(self.vocab_size, activation='linear')

        # Construct discriminator, also seq2seq with VD
        self.dis_enc_rnn = rec_layer(use_cudnn)(dis_enc_rnn_size,
                                                      return_sequences=True,
                                                      return_state=True,
                                                      stateful=False)
        self.dis_dec_rnn = rec_layer(use_cudnn)(dis_dec_rnn_size,
                                                      return_sequences=False,
                                                      return_state=False,
                                                      stateful=False)
        self.dis_predictions = layers.Dense(1, activation='linear')
    
    def transform_input_with_is_missing_token(inputs,
                                              targets_present,
                                              is_missing_id):
        input_missing = tf.constant(
            is_missing_id,
            dtype=tf.int16,
            shape=inputs.shape)
        # First input is never missing
        initial_input_present = tf.constant(True, tf.bool,
                                            shape=(inputs.shape[0], 1))
        inputs_present = tf.concat(
            [initial_input_present, targets_present[:, :-1]], axis=1)
        transformed_input = tf.where(inputs_present, inputs, input_missing)
        return transformed_input

    def generate(inputs, targets, targets_present, is_training, teacher_forcing):
        real_inputs = inputs
        #TODO fix use of masking token
        masked_inputs = transform_input_with_is_missing_token(
            inputs, target_present, 2)
        # Pass through gen_encoder
        real_rnn_inputs = self.embedding(real_inputs)
        masked_rnn_inputs = self.embedding(masked_inputs)    
        #TODO add Variational Dropout (the make_mask func)
        real_enc_outs, real_enc_final_state = self.gen_enc_rnn(real_rnn_inputs)
        masked_enc_outs, masked_enc_final_state = self.gen_enc_rnn(masked_rnn_inputs)
        # Pass masked through decoder
        gen_dec_state = masked_enc_final_state
        gen_dec_rnn_outs = []
        gen_dec_rnn_logits = []
        seq_len = inputs.shape[1]
        for t in range(seq_len):
            if t == 0:
                rnn_inp = real_rnn_inputs[:, t]
            else:
                real_rnn_inp = real_rnn_inputs[:, t]
                fake_rnn_inp = gen_dec_rnn_out:, t-1]
                if teacher_forcing:
                    # if TF, always give true previous token.
                    rnn_inp = real_rnn_inp
                else:
                    # otherwise, if last token was generated/fake, give that.
                    rnn_inp = tf.where(targets_present[:, t-1], 
                                       real_rnn_inp,
                                       fake_rnn_inp)
            rnn_out, gen_dec_state = self.gen_dec_rnn(rnn_inp, initial_state=gen_dec_state)
            logits = self.token_logits(rnn_out)
            gen
            real = targets_present[:, t]
            if teacher_forcing:
                # Always produce correct output.
                fake = real
            else:
                dist = tf.contrib.distributions.Categorical(logits=logit)
                fake = dist.sample()
            output = tf.where(targets_present[:, t], real, fake)
            gen_dec_rnn_outs.append(output)
            gen_dec_rnn_logits.append(logits)
        #TODO attention_construct_fn here
        logits = tf.stack(logits, axis =1)
        outs = tf.stack(gen_dec_rnn_outs, axis = 1)
        return logits, outs

    def discriminate(sequence, masked_input, is_training):
        # Discriminator encoder reads in "True" masked input sequence
        # to eliminate the "director director" failure mode
        masked_input_embed = self.embedding(masked_input)
        dis_enc_hidden, dis_enc_final_state = self.dis_enc_rnn(masked_input_embed)
        
        # Seed decoder with final state = true masked context
        # Pass in token sequence
        seq_len = sequence.shape[1]
        state = dis_enc_final_state
        input_embed = self.embedding(sequence)
        preds = [] # predicted logit for being fake/real per token
        for t in range(seq_len):
            rnn_out, state = dis_dec_rnn(input_embed[:, t], state)
            pred = self.dis_predictions(rnn_out)
            preds.append(pred)
    
        predictions = tf.stack(preds, axis=1)
        return tf.squeeze(predictions, axis=2)





            
        






    

        





