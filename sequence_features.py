import string
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, layers


def error_demo():
	data = np.chararray(100, itemsize=1, unicode=True)
	func = lambda x: [random.choice(string.ascii_letters) for i in x]
	data = func(data)
	data = {'data': np.array(data).reshape((20, 5))}
	labels = np.random.rand(20)

	dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	for x in dataset:
		print(x)

	fc = tf.feature_column.sequence_categorical_column_with_hash_bucket('features', hash_bucket_size=5, dtype=tf.dtypes.string)
	fc = tf.feature_column.embedding_column(fc, dimension=10)
	fc = [fc]

	sequence_feature_layer = tf.keras.experimental.SequenceFeatures(fc)
	sequence_input, sequence_length = sequence_feature_layer({'features': tf.keras.Input(shape=(1,), name='firmenname', dtype=tf.string)})
	sequence_length_mask = tf.sequence_mask(sequence_length)

	lstm = layers.Bidirectional(layers.LSTM(8))(sequence_input, mask=sequence_length_mask)
	dense = layers.Dense(16)(lstm)
	out = layers.Dense(1, activation='sigmoid')(dense)


if __name__ == '__main__':
	error_demo()
