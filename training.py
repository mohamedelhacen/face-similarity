import torch
from torch import optim

from network import SiameseNetwork
from utils import ContrastiveLoss

def train(train_dataloader, epochs=100):

	net = SiameseNetwork().cuda()
	criterion = ContrastiveLoss()
	optimizer = optim.Adam(net.parameters(), lr = 0.0005)

	counter = []
	loss_history = []
	iteration_number = 0

	for epoch in range(epochs):

		for i, (img0, img1, label) in enumerate(train_dataloader, 0):
			
			img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

			optimizer.zero_grad()

			output1, output2 = net(img0, img1)

			loss_contrastive = criterion(output1, output2, label)
			loss_contrastive.backward()
			
			optimizer.step()
			
			if i % 10 == 0:
				print(f"Epoch number {epoch}: current loss {loss_contrastive.item()}\n")
				iteration_number +=10

				counter.append(iteration_number)
				loss_history.append(loss_contrastive.item())
    

	torch.save(net, 'network.pth')
	
	return counter, loss_history