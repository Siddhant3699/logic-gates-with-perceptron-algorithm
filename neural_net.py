import time
import numpy as np

class Perceptron:

	def __init__(self):
		np.random.seed(1)
		self.weights = np.random.random(3)

	def activation(self,x):
		if x>0.5:
			return 1
		return 0

	def train(self,inputs,outputs):
		self.training_outputs = outputs
		for i in range(100):
			for j in range(len(inputs)):
				w_sum = np.dot(inputs[j],self.weights)
				yin = self.activation(w_sum)
				error = self.training_outputs[j] - yin
				self.weights += (0.05*error*inputs[j])

	def predict(self,inputs):
		temp = []
		for i in inputs:
			y = self.activation(np.dot(i,self.weights))
			temp.append(y)
		return np.array(temp)

if __name__ == "__main__":
	ip = np.array([
		[0,0,0],
		[0,0,1],
		[0,1,0],
		[0,1,1],
		[1,0,0],
		[1,0,1],
		[1,1,0],
		[1,1,1],
		])
	or_op = np.array([0,1,1,1,1,1,1,1])
	and_op = np.array([0,0,0,0,0,0,0,1])
	o = Perceptron()

	print("Neural Network training for AND Gate...\n")
	o.train(ip,and_op)
	time.sleep(2)
	y_pred = o.predict(ip)
	print("AND Gate ->")
	print("| A | B | C | Y |")
	print("-----------------")
	for i in range(len(y_pred)):
		print("| {0} | {1} | {2} | {3} |".format(ip[i][0],ip[i][1],ip[i][2],y_pred[i]))

	print("\n")
	
	print("Neural Network training for OR Gate...\n")
	o.train(ip,or_op)
	time.sleep(2)
	y_pred = o.predict(ip)
	print("OR Gate ->")
	print("| A | B | C | Y |")
	print("-----------------")
	for i in range(len(y_pred)):
		print("| {0} | {1} | {2} | {3} |".format(ip[i][0],ip[i][1],ip[i][2],y_pred[i]))