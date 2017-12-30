import tensorflow as tf
import os, time, datetime
import pandas as pd
import numpy as np
from copy import deepcopy
from CharNet import CharNet
from CharNetConfig import CharNetConfig
from Data import Data

def main(train_file, dev_file, config):
    train_data = Data(train_file, config.alstr, is_dev=False, batch_size=128)
    dev_data = Data(dev_file, config.alstr, is_dev=True, batch_size=128)
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True

    with tf.Session(config=conf) as sess:
        charnet = CharNet(config.conv_layers, config.fc_layers, config.l0, config.alphabet_size, train_data.encoder)
        
        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(charnet.loss)
        train_op = optimizer.apply_gradients(grads)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", charnet.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(num_batch, x_batch, y_batch, step):
            """
            A single training step
            """
            y_batch = np.reshape(y_batch, (-1, 1))
            mm = x_batch.tolist()
            x_batch = np.array([k.tolist() for k in mm])
            
            feed_dict = {
                charnet.input_num: num_batch,
                charnet.input_x: x_batch,
                charnet.input_y: y_batch,
                charnet.dropout_keep_prob: .5
            }

            _, summaries, loss = sess.run(
                [train_op,
                 train_summary_op,
                 charnet.loss],
                feed_dict
            )

            print("step {}, loss {:g}".format(step, loss))
            train_summary_writer.add_summary(summaries, step)

        def predict_on_test(num_batch, x_batch, results):
            mm = x_batch.tolist()
            x_batch = np.array([k.tolist() for k in mm])
            
            feed_dict = {
                charnet.input_num: num_batch,
                charnet.input_x: x_batch,
                charnet.dropout_keep_prob: 1.0
            }
            result = sess.run([charnet.yhat], feed_dict)
            for l in result:
                for elem in l:
                    results.append(elem)

        for epoch in range(5):
            train_data.shuffling()
            for i in range(int(len(train_data.y)/train_data.batch_size) + 1):
                input_x, input_num, y = train_data.next_batch(i)
                train_step(input_num, input_x, y, i)
                
                if i % 1000 == 0:
                    path = saver.save(sess, './model.ckpt')
                    print("Epoch {}, Saved model checkpoint to {}\n".format(epoch, path))
        test_id = [i for i in range(len(dev_data.input_x))]
        submission = pd.DataFrame(test_id, columns=['test_id'])
        preds = []
        for i in range(int(len(dev_data.input_x)/dev_data.batch_size) + 1):
            input_x, input_num, _ = dev_data.next_batch(i)
            predict_on_test(input_num, input_x, preds)
        
        submission['price'] = pd.Series(preds)
        submission.to_csv('./submission.csv', index=False)

if __name__ == '__main__':
    train_file = './train.tsv'
    dev_file = './test.tsv'
    config = CharNetConfig()
    main(train_file, dev_file, config)
