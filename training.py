import tensorflow as tf
import os, time

def main(train_file, dev_file, config):
    train_data = Data(train_file, config.alstr)
    dev_data = Data(dev_file, config.alstr)
    config = CharNetConfig()

    with tf.Session() as sess:
        charnet = CharNet(config.conv_layers,
                          config.fc_layers,
                          config.l0,
                          config.alphabet_size,
                          train_data.encoder)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads = optimizer.compute_gradients(charnet.loss)
        train_op = optimizer.apply_gradients(grads)

        global_step = tf.Variable(0, trainable=False)

        # Summaries for grads
        grad_summaries = []
        for g, v in grads:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.histogram("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", char_cnn.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(num_batch, x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                charnet.input_num: num_batch,
                charnet.input_x: x_batch,
                charnet.input_y: y_batch,
                charnet.dropout_keep_prob: .5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op,
                 global_step,
                 train_summary_op,
                 charnet.loss],
                feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def predict_on_test(num_batch, x_batch, results):
            feed_dict = {
                charnet.input_num: num_batch,
                charnet.input_x: x_batch,
                charnet.dropout_keep_prob: 1.0
            }

            results.append(sess.run([charnet.yhat], feed_dict))

        for epoch in range(self.epochs):
            train_data.shuffling()
            for _ in range(len(train_data.y)/train_data.batch_size):
                input_x, input_num, y = train_data.next_batch()
                train_step(input_num, input_x, y)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

        results = []
        predict_on_test(dev_data.input_num, dev_data.input_x)
        submission = pd.DataFrame([[i for i in range(len(results))], results], columns=['id', 'price'])
        submission.to_csv('./submission.csv')

if __name__ == '__main__':
    train_file = './train.tsv'
    dev_file = './test.tsv'
    config = CharNetConfig()
    main(train_file, dev_file, config)
