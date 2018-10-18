import tensorflow as tf
import cv2
import sys
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

sys.path.append("game/")

class Network(object):

	def __init__(self, img_width, img_height, name = "network"):

		self.name = name
		self.img_width = img_width
		self.img_height = img_height

	def weight_variable(self, shape):

		return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

	def bias_variable(self, shape):

		return tf.Variable(tf.constant(0.01, shape=shape,dtype="float32"))

	def conv2d(self, x, W, stride):

		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool(self,x):

		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def createNetwork(self):

		with tf.variable_scope(self.name):
			W_conv1 = self.weight_variable([8, 8, 4, 32])
			b_conv1 = self.bias_variable([32])

			W_conv2 = self.weight_variable([4, 4, 32, 64])
			b_conv2 = self.bias_variable([64])

			W_conv3 = self.weight_variable([3, 3, 64, 64])
			b_conv3 = self.bias_variable([64])

			W_fc1 = self.weight_variable([1600, 512])
			b_fc1 = self.bias_variable([512])

			W_fc2 = self.weight_variable([512, 2])  #2 actions
			b_fc2 = self.bias_variable([2])  #2 actions

			#layers
			self.inp = tf.placeholder(tf.float32, [None, 80, 80, 4])

			h_conv1 = tf.nn.relu(self.conv2d(self.inp, W_conv1, 4) + b_conv1)
			h_maxpool1 = self.max_pool(h_conv1)

			h_conv2 = tf.nn.relu(self.conv2d(h_maxpool1, W_conv2, 2) + b_conv2)

			h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

			h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

			self.out = tf.matmul(h_fc1, W_fc2) + b_fc2  #Q values

			return self.inp, self.out, h_fc1


class Flappy(object):

	def __init__(self):

		self.img_width = 80
		self.img_height = 80
		self.img_depth = 4
		self.epsilon = 0.001   #might have to decrease this value during later stages of training

		self.num_episodes = 100000
		self.pre_train_steps = 10000
		    #I have kept this to a large value. Decrease it if not necessary
		self.update_freq = 100
		self.batch_size = 32
		self.gamma = 0.99
		self.learning_rate = 1e-6
		self.max_steps = 1e4

	def copy_network(self, net1, net2, sess):

		vars_net1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net1.name)
		vars_net2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net2.name)

		for index in range(len(vars_net1)):
			sess.run(vars_net2[index].assign(vars_net1[index]))

	def model(self):

		self.target_reward = tf.placeholder(tf.float32, [None, 1], name = "target_reward")
		self.action_list = tf.placeholder(tf.uint8, [None, 1], name = "action_list")     #changed from int32 to uint32

		observed_reward = tf.reduce_sum(self.main_net.out * tf.one_hot(tf.reshape(self.action_list,[-1]), 2, dtype = tf.float32), 1, keepdims = True)

		self.loss = tf.reduce_mean(tf.square(observed_reward - self.target_reward))

		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.loss_opt = optimizer.minimize(self.loss)

		self.model_vars = tf.trainable_variables()

	def pre_process(self, img):

		x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

		return x_t

	def policy(self, sess, algo, img_batch):

		if algo == "epsilon_greedy":
			temp = random.random()

			if temp < self.epsilon:
				print("Random Action from Epsilon Greedy")
				temp_action = random.randint(0, 1)
			else:
				temp_q_values = sess.run([self.main_net.out], feed_dict = {self.main_net.inp: np.reshape(np.stack(img_batch, axis = 2), [-1, 80, 80, 4])})
				temp_action = np.argmax(temp_q_values)

			return temp_action

	def train(self):

		#Separate Train and Target Networks

		self.main_net = Network(self.img_width, self.img_height, name = "main_net")
		self.target_net = Network(self.img_width, self.img_height, name = "target_net")

		#Initialize the networks

		self.main_net.createNetwork()
		self.target_net.createNetwork()

		self.model()

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			self.copy_network(self.main_net, self.target_net, sess)

			total_steps = 0
			total_reward_list = []
			hist_buffer = []

			for i in range(self.num_episodes):

				#Adding initial 4 frames to the image buffer array

				game_state = game.GameState()
				img_batch = []
				total_reward = 0.0

				temp_action = random.randint(0, 1)
				action = np.zeros([2])
				action[temp_action] = 1
				new_state, reward, done = game_state.frame_step(action)

				total_steps += 1

				temp_img = self.pre_process(new_state)
				img_batch = [temp_img] * 4

				while(True):

					if total_steps < 3000:
						temp_action = random.randint(0, 1)
						#print("Exploring. Action taken: ", temp_action)
					else:
						temp_action = self.policy(sess, "epsilon_greedy", img_batch)

					#print(temp_action)

					action = np.zeros([2])
					action[temp_action] = 1
					new_state, reward, done = game_state.frame_step(action)

					temp_img = self.pre_process(new_state)
					#temp_img = np.reshape(temp_img, (80, 80))   #added this new

					total_reward += reward

					new_img_batch = img_batch[1:]
					new_img_batch.insert(3, temp_img)

					hist_buffer.append((np.stack(img_batch, axis = 2), temp_action, reward, np.stack(new_img_batch, axis = 2), done))

					if len(hist_buffer) >= 10000:
						hist_buffer.pop(0)

					# Adding image to the batch

					img_batch.insert(len(img_batch), temp_img)
					img_batch.pop()

					################

					if(total_steps > self.pre_train_steps):

						rand_batch = random.sample(hist_buffer, self.batch_size)

						reward_hist = [m[2] for m in rand_batch]
						state_hist = [m[0] for m in rand_batch]
						action_hist = [m[1] for m in rand_batch]
						next_state_hist = [m[3] for m in rand_batch]


						temp_target_q = sess.run(self.target_net.out, feed_dict = {self.target_net.inp: np.stack(next_state_hist)})
						temp_target_reward=[]

						for j in range(self.batch_size):
							terminal = rand_batch[j][4]
							if terminal:
								temp_target_reward.append(reward_hist[j])
							else:
								#temp_target_q = np.amax(temp_target_q, 1)
								temp_target_reward.append(reward_hist[j] + self.gamma * np.max(temp_target_q[j]))

						temp_target_reward = np.reshape(temp_target_reward, [self.batch_size, 1])

						_ = sess.run(self.loss_opt,
								feed_dict = {self.main_net.inp: np.stack(state_hist), self.target_reward: temp_target_reward,
											self.action_list: np.reshape(np.stack(action_hist), [self.batch_size, 1])})

						if total_steps % self.update_freq == 0:
							self.copy_network(self.main_net, self.target_net, sess)

						#print("Training...")

					#added this new
					#img_batch = new_img_batch

					if done:
						break

					total_steps += 1

				print("Total rewards in episode {} is {}, total number of steps are {}".format(i, total_reward, total_steps))


	def play(self, mode = "random"):

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			writer = imageio.get_writer("gif/demo.gif", mode = 'I')

			game_state = game.GameState()
			total_steps = 0
			img_batch = []

			action = np.zeros([2])
			action[0] = 1
			new_state, reward, done = game_state.frame_step(action)

			temp_img = self.pre_process(new_state)

			for j in range(4):
				img_batch.insert(len(img_batch), temp_img)

			for j in range(self.max_steps):

				if mode == "random":
					temp_action = 0 if random.randint(0, 1) <= 0.6 else 1
				else:
					temp_weights = sess.run([self.main_net.out], feed_dict = {self.main_net.inp: np.reshape(np.stack(img_batch, axis = 2), [-1, 80, 80, 4])})
					temp_action = np.argmax(temp_weights)

				action = np.zeros([2])
				action[temp_action] = 1

				new_state, reward, done = game_state.frame_step(action)

				temp_img = self.pre_process(new_state)
				img_batch.insert(0, temp_img)
				img_batch.pop(len(img_batch)-1)

				total_steps += 1

				if done:
					break

			print("Total Steps {}".format(total_steps))

def main():

	model = Flappy()
	model.train()
	#model.play("not random")

if __name__ == "__main__":
	main()
