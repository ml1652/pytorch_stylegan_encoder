import torch
import matplotlib.pyplot as plt
import random

bins_num = 60
torch.manual_seed(20)
choice_sigma = 0

class SoftHistogram(torch.nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)
        #self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x
x = torch.rand(224)*255

while choice_sigma <= 10:
    softhist = SoftHistogram(bins_num, min=0, max=255, sigma=choice_sigma).cuda()
    histc_result = torch.histc(x, bins_num, min=0, max=255).cpu().detach().numpy()
    softhist_result = softhist(x.cuda()).cpu().detach().numpy()
    str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/diagram/hist_sigmaTest/'
    str2 = 'simga=%s.jpg' % choice_sigma
    hist_save_path = str1 + str2

    plt.plot(histc_result, color='blue',label='histc')
    plt.plot(softhist_result,color='red',label='softhist')
    plt.legend()
    plt.xlim(0, 50)
    #plt.show()
    plt.savefig(hist_save_path)
    plt.clf()
    choice_sigma += 0.05