from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

class PytorchLearn:
	"""Class intended to learn the flow of using Pytorch to train a model."""
	def __init__(self):
		self.test_data = None
		self.training_data = None
		self.loaded_test = None
		self.loaded_train = None

	def get_data(self):
		"""Retrieves MNIST data using torchvisions datasets"""
		self.training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor())
		self.test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())

	def visualize_data(self):
		"""A way to visualize MNIST data"""
		figure = plt.figure(figsize=(8, 8))
		cols, rows = 5, 5

		for i in range(1, cols * rows + 1):
			sample_idx = torch.randint(len(self.training_data), size=(1,)).item()
			img, label = self.training_data[sample_idx]
			figure.add_subplot(rows, cols, i)
			plt.axis("off")
			plt.imshow(img.squeeze(), cmap="gray")
		plt.show()
	
	def data_loader(self):
		"""
		Allows iteration over the dataset in mini-batches instead of all at once, 
		shuffle the data while training the models
		"""
		self.loaded_train = DataLoader(self.training_data, batch_size=64, shuffle=True)
		self.loaded_test = DataLoader(self.test_data, batch_size=64, shuffle=True)
		return self.loaded_train, self.loaded_test
	
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		"""Called when model is executed"""
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
	
	def train_model(self, dataloader, model, loss_fn, optimizer):
		"""Used to train neural network on training data."""
		size = len(dataloader.dataset)
		for batch, (X, y) in enumerate(dataloader):
			pred = model(X)
			loss = loss_fn(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 1000 == 0:
				loss, current = loss.item(), batch * len(X)
				print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	def test(self, dataloader, model, loss_fn):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		test_loss, correct = 0, 0

		with torch.no_grad():
			for X, y in dataloader:
				pred = model(X)
				test_loss += loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()

		test_loss /= num_batches
		correct /= size
		print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	
	def predict(self, data):
		"""Predicts labels for given data using trained model."""
		# Convert data to PyTorch tensor if it's not
		if not isinstance(data, torch.Tensor):
			data = torch.from_numpy(data)

		# Add a dimension for the batch size
		data = data.unsqueeze(0)  # Now the shape is [1, 784]

		# Set the model to evaluation mode
		self.eval()

		# Perform forward pass and get the predictions
		with torch.no_grad():
			output = self(data)
			_, predicted = torch.max(output, 1)

		return predicted.item()

def main():
	pl = PytorchLearn()
	pl.get_data()
	dataloader_train, dataloader_test = pl.data_loader()

	model = NeuralNetwork()
	loss_function = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

	# Training Neural Net with an epoch of 10. 
	epochs = 10
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		model.train_model(dataloader_train, model, loss_function, optimizer)
		model.test(dataloader_test, model, loss_function)
	print("Done!")

	torch.save(model, "model.pth")

	new_data = np.random.rand(784) 

	# Convert it to a float32 tensor
	new_data = torch.from_numpy(new_data).float()

	# Load the trained model
	model = torch.load("model.pth")

	# Make a prediction
	prediction = model.predict(new_data)
	print(f"The predicted label is: {prediction}")

if __name__ == "__main__":
	main()
