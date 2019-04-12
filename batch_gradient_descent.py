import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

#for tensorboard visualitzation
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#get california housing data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

#scaling the numberical values 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
#adding bias variable
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

#using placeholder for batch gradient descent
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0 ), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
#Name Scopes to reduce clutter in tensorboard graphs
with tf.name_scope("loss") as scope : 
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)			
training_op = optimizer.minimize(mse)

#initialising the default variables
init = tf.global_variables_initializer()

#for tensorboard
# creates a node in the graph that will evaluate the MSE value and write it to a TensorBoard-compatible binary log string called a summary
mse_summary = tf.summary.scalar('MSE', mse)
# to write summaries to logfiles in the log directory
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#for saving tensorflow model
saver = tf.train.Saver()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

#fetching data in Batches using np.random
def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
	indices = np.random.randint(m, size=batch_size)  # not shown
	X_batch = scaled_housing_data_plus_bias[indices] # not shown
	y_batch = housing.target.reshape(-1, 1)[indices] # not shown
	return X_batch, y_batch

#session starts here
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			if(batch_index % 10 == 0) :
				summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

	best_theta = theta.eval()
	save_path = saver.save(sess, "model/my_model_final.ckpt")

file_writer.close()


"""
#following code can be used to restore the graph structure and the values of theta
saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
	saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
	best_theta_restored = theta.eval()
"""
