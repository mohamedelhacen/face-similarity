import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ContrastiveLoss(nn.Module):
	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive	

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis('off')
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', 
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def convert_img(img, size=[1, 1, 100, 100]):
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()])
    im = Image.open(img)
    im = im.convert('L')
    im = transformation(im)
    im = im.reshape(size)

    return im
