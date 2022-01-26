import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.patch_num = (28 // patch_size) ** 2    #patch的个数

        self.mlp1 = nn.Sequential(  #对每个token进行全连接
            nn.Linear(self.patch_num, hidden_dim),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, self.patch_num),
            nn.Dropout(0)
        )
        self.mlp2 = nn.Sequential(  #对每个channel进行全连接
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, 14),
            nn.Dropout(0)
        )

        self.norm = nn.LayerNorm(14)    #layer norm
        ########################################################################

    def forward(self, x):
        ########################################################################
        #输出训练的结果
        x = self.norm(x)
        x2 = self.mlp1(x.transpose(1, 2)).transpose(1, 2)   #转置再转置
        x = x + x2      #skip_connection
        x2 = self.norm(x)
        x2 = self.mlp2(x2)
        return x + x2   #skip_connection
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        self.embed = nn.Conv2d(1, 14, kernel_size=patch_size, stride=patch_size)
        MIX = Mixer_Layer(patch_size, hidden_dim)
        self.mixlayers = nn.Sequential(*[MIX for _ in range(depth)])
        self.norm = nn.LayerNorm(14)
        self.connect = nn.Linear(14, 10)
        ########################################################################

    def forward(self, data):
        ########################################################################
        # 注意维度的变化
        x = self.embed(data).flatten(2).transpose(1, 2) #Pre-patch Fully-connected
        x = self.mixlayers(x)
        x = self.norm(x)
        x = torch.mean(x, dim=1)    #global average pooling
        x = self.connect(x)     #fully connected
        return x
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 计算loss并进行优化
            optimizer.zero_grad()   #每次运行前清空梯度
            pre = model(data)
            loss = criterion(pre, target)
            loss.backward()
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 需要计算测试集的loss和accuracy
            pre = model(data)
            loss = criterion(pre, target)
            test_loss += loss
            a = pre.argmax(dim=1).tolist()
            b = target.tolist()
            for i in range(len(a)):
                if a[i] == b[i]:
                    num_correct += 1
        accuracy = num_correct / 10000
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))



if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    model = MLPMixer(patch_size=2, hidden_dim=384, depth=2).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = torch.optim.Adamax(params, lr=10*learning_rate, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
    ########################################################################
    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
