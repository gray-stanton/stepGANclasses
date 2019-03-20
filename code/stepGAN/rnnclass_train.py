import tensorflow as tf
import texar as tx
import importlib
import numpy as np

config = importlib.import_module('rnn_config')
def main(_):
    train_data = tx.data.MultiAlignedData(config.train_data_hparams)
    val_data = tx.data.MultiAlignedData(config.val_data_hparams)
    #test_data = tx.data.MultiAlignedData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=None)
    data_batch = iterator.get_next()
    vocab = train_data.vocab('x')
    vocab_size = vocab.size
    global_step = tf.train.get_or_create_global_step()

    # Layers
    embedding = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                        hparams=config.emb_hparams)
    rnn_layers = []
    for p in config.rnn_layers_hparams:
        layer = tf.keras.layers.CuDNNGRU(**p)
        rnn_layers.append(layer)

    classifier = tf.layers.Dense(**config.output)

    # Apply
    input_text = data_batch['x_text_ids']
    output_labels = data_batch['label']
    output_label_onehots = tf.one_hot(output_labels, 2, axis=-1)
    x = embedding(input_text)
    for l in rnn_layers:
        x =  l(x)
    logits = classifier(x)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = output_label_onehots, logits=logits)
    # Wrap up
    train_op = tx.core.get_train_op(loss, global_step=global_step, hparams=config.opt)
    preds = tf.argmax(logits, axis=1)
    acc = tx.evals.accuracy(output_labels, preds)
    loss_sum = tf.summary.scalar('cross_ent_loss', loss)
    acc_sum = tf.summary.scalar('accuracy', acc)
    all_sum = tf.summary.merge_all()

    def run_one_epoch(sess, mode, epoch, writer):
        is_train = tx.utils.is_train_mode_py(mode)
        
        fetches = {
            "acc" : acc,
            "loss" : loss,
            "batch_size" : tx.utils.get_batch_size(data_batch['label']),
            "summaries" : all_sum
        }
        if is_train:
            fetches['train_op'] = train_op
        feed_dict = {tx.context.global_mode(): mode}
        total_loss = 0
        total_acc = 0
        nsamples = 0
        while True:
            try:
                rets  = sess.run(fetches, feed_dict)
                batch_loss, batch_acc, size = rets['loss'], rets['acc'], rets['batch_size']
                sums = rets['summaries']
                total_loss += size * batch_loss
                total_acc  += size * batch_acc
                nsamples += size
                step = tf.train.global_step(sess, global_step)
                writer.add_summary(sums, step)
                if step % 100 == 0:
                    print('step: {}, epoch:{}'.format(step, epoch))
                    #print('step: {}, epoch {}, loss {:0.02f}, accuracy {:0.02f}'.format(
                    #    step, epoch, total_loss/nsamples, total_acc/nsamples))
            except tf.errors.OutOfRangeError:
                break
        return total_loss / nsamples , total_acc / nsamples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        writer = tf.summary.FileWriter(config.log_dir, session=sess, flush_secs=30)
        with writer:
            min_loss = -1
            for epoch in range(1, config.num_epochs + 1):
                # Train
                iterator.switch_to_train_data(sess)
                epoch_loss, epoch_acc = run_one_epoch(sess, tf.estimator.ModeKeys.TRAIN,
                                           epoch, writer)
                print("TRAIN -- EPOCH{}   EPOCH LOSS:{:0.02f}  EPOCH ACC:{:0.02f}".format(
                    epoch, epoch_loss, epoch_acc))
                iterator.switch_to_val_data(sess)
                val_loss, val_acc = run_one_epoch(sess, tf.estimator.ModeKeys.EVAL,
                                                 epoch, writer)
                print("VAL -- EPOCH{}   EPOCH LOSS:{:0.02f}  EPOCH ACC:{:0.02f}".format(
                                                  epoch, val_loss, val_acc))
            print("DONE")

if __name__ == '__main__':
    tf.app.run(main=main)
                

    
    










