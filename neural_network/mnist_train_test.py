import torch
from torchvision import datasets, transforms
from network import NeuralWork
from algo.matrix import Matrix

def to_matrix(batch):
    x = batch.view(batch.size(0), -1).T
    return Matrix(x.tolist())

def mean_abs(m):
    r = len(m)
    c = len(m[0])
    s = 0.0
    for i in range(r):
        row = m[i]
        for j in range(c):
            s += abs(row[j])
    return s / (r * c)

def train():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    net = NeuralWork(hidden=1, size=784)
    train_images = []
    train_labels = []
    for idx, (data, _) in enumerate(train_loader):
        m = to_matrix(data)
        train_images.append(m)
        train_labels.append(m)
        if idx >= 500:
            break
    net.learn(0.1, train_images, train_labels)
    for data, _ in test_loader:
        m = to_matrix(data)
        out = net.predict(m)
        if isinstance(out, Matrix):
            err = out - m
            print('test_mae', mean_abs(err))
        break
    net.save('./data/mnist_autoencoder.nn')

if __name__ == '__main__':
    train()
