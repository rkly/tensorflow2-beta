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

	dataset = tf.data.Dataset.from_tensor_slices((data,))

	for x in dataset:
		print(x)


if __name__ == '__main__':
	error_demo()
