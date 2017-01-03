# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import random


class ToySequenceData:
	def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value = 1000):
		self.data = []
		self.labels = []
		self.seqlen = []

		for i in range(n_samples):

			length = random.randint(min_seq_len, max_seq_len)

			self.seqlen.append(length)

			if random.random() < 0.5:
				rand_start = random.randint(0, max_value - length)
				s = [[float(i)/max_value] for i in range(rand_start, rand_start + length)]

				s += [[0.] for i in range(max_seq_len - length)]

				self.data.append(s)
				self.labels.append([1.,0.])

			else:
				s = [[float(random.randint(0, max_value))/max_value] for i in range(length)]

				s += [[0.] for i in range(max_seq_len - length)]

				self.data.append(s)
				self.labels.append([0.,1.])

		self.batch_id = 0


	def next(self, batch_size):
		if self.batch_id == len(self.data):
            		self.batch_id = 0
        	batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        	batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        	batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        	self.batch_id = min(self.batch_id + batch_size, len(self.data))
		return batch_data, batch_labels, batch_seqlen

def weights_initializer(shape):
	value = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(value)

def bias_initializer(shape):
	value = tf.constant(0.1, shape = shape)
	return tf.Variable(value)

class Dynamic_RNN:
	def __init__(self, max_seq_len, hidden_num = 64, n_classes = 2, learning_rate = 0.01, training_iters = 100000):
		self.learning_rate = learning_rate
		self.training_iters = training_iters
		self.n_classes = n_classes
		self.hidden_num = hidden_num
		self.max_seq_len = max_seq_len
		
		self.seqs = tf.placeholder(tf.float32, [None, max_seq_len, 1])
		self.labels = tf.placeholder(tf.float32, [None, n_classes])
		self.seqlen = tf.placeholder(tf.int32, [None])
		self.partitions = tf.placeholder(tf.int32, [None])
		self.preds = None

	def generate_partition(self, batch_size, seqlen):
		partitions = [0] * (batch_size * self.max_seq_len)
		step = 0
		for each in seqlen:
			idx = each + self.max_seq_len * step
			partitions[idx - 1] = 1
			step += 1
		return partitions

	def initialize_rnn_net(self):
		weights = weights_initializer([self.hidden_num, self.n_classes])
		bias = bias_initializer([self.n_classes])

		lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_num)

		# return output shape is [batch_size, time_step, hidden_num]
		output, state = tf.nn.dynamic_rnn(lstm, self.seqs, dtype=tf.float32, sequence_length = self.seqlen)

		outputs = tf.reshape(tf.pack(output), [-1, lstm.output_size])

		num_partitions = 2

		self.res_out = tf.dynamic_partition(outputs, self.partitions, num_partitions)

		'''
		Use tf.gather, however we might receive a warning

		# Get the batch size 
		batch_size = tf.shape(outputs)[0]

		# Use index to find the max time step output for each input sequence
		index = tf.range(0, batch_size) * self.max_seq_len + self.seqlen - 1

		index = tf.cast(index, tf.int32)

		outputs = tf.pack(tf.reshape(outputs, [-1, lstm.output_size]))

		new_outputs = tf.gather(outputs, index)

		'''

		self.preds = tf.matmul(self.res_out[1], weights) + bias

		return self.preds

	def initialize_rnn_optimizer(self):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.preds, self.labels))

		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.labels, 1)), tf.float32))		

		return loss, optimizer, accuracy


if __name__ == '__main__':

	max_seq_len = 20
	train_numbers = 1000
	test_numbers = 500
	train_iters = 100000
	batch_size = 64

	train_set = ToySequenceData(n_samples = train_numbers, max_seq_len = max_seq_len)
	test_set = ToySequenceData(n_samples = test_numbers, max_seq_len = max_seq_len)

	drnn = Dynamic_RNN(max_seq_len = max_seq_len, learning_rate = 0.01, training_iters = train_iters)

	drnn.initialize_rnn_net()
	
	loss, opt, acc = drnn.initialize_rnn_optimizer()

	with tf.Session() as sess:
		print "Start Initializing Glabal Variables"
		sess.run(tf.global_variables_initializer())

		step = 0

		print "Start Training"
		while step * batch_size < drnn.training_iters:

			batch_data, batch_labels, batch_seqlen = train_set.next(batch_size)

			partitions = drnn.generate_partition(len(batch_data), batch_seqlen)

			res = sess.run(opt, feed_dict = {drnn.seqs: batch_data, drnn.labels: batch_labels,drnn.seqlen: batch_seqlen, drnn.partitions: partitions})

			if step % 20 == 0:
				l = sess.run(loss, feed_dict = {drnn.seqs: batch_data, drnn.labels: batch_labels, drnn.seqlen: batch_seqlen, drnn.partitions: partitions})
				a = sess.run(acc, feed_dict = {drnn.seqs: batch_data, drnn.labels: batch_labels, drnn.seqlen: batch_seqlen, drnn.partitions: partitions})
				print "Global Step: %d, loss: %.5f, accuracy: %.5f" % (step, l, a)

			step += 1
		

		# "Start Testing"
		partitions = drnn.generate_partition(test_numbers, test_set.seqlen)

		print "Test accuracy: %.5f" % sess.run(acc, feed_dict = {drnn.seqs: test_set.data, drnn.labels: test_set.labels, drnn.seqlen: test_set.seqlen, drnn.partitions: partitions})
