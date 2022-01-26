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
        self.token_dim = (28 // patch_size) ** 2  # 序列数（patch个数）
        self.channel_dim = 14  # 通道数（嵌入维度）
        self.mlp1 = nn.Sequential(  # Token_mixing
            nn.Linear(self.token_dim, hidden_dim),  # Fully-connected
            nn.GELU(),  # 激活函数GELU
            nn.Dropout(0),  # 防止或减轻过拟合
            nn.Linear(hidden_dim, self.token_dim),  # Fully-connected
            nn.Dropout(0)
        )
        self.mlp2 = nn.Sequential(  # Channel_mixing
            nn.Linear(self.channel_dim, hidden_dim),  # Fully-connected
            nn.GELU(),  # 激活函数GELU
            nn.Dropout(0),
            nn.Linear(hidden_dim, self.channel_dim),  # Fully-connected
            nn.Dropout(0)
        )
        self.norm = nn.LayerNorm(self.channel_dim)
        ########################################################################

    def forward(self, x):
        ########################################################################
        x = self.norm(x).transpose(1, 2)  # Layer Norm
        x_token = self.mlp1(x)  # MLP1-Token_mixing
        x_skip = self.norm((x + x_token).transpose(1, 2))  # Skip-connections & Layer Norm
        x_channel = self.mlp2(x_skip)  # MLP2-Channel_mixing
        x_mixer = x_skip + x_channel  # Skip-connections
        return x_mixer
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        self.image_channel = 1  # MNIST数据集图片维度
        self.channel_dim = 14  # 通道数（嵌入维度）
        self.class_num = 10  # 分类标签数（0~9）
        # Per-patch Fully-connected
        self.Pre_patch = nn.Conv2d(in_channels=self.image_channel, out_channels=self.channel_dim,
                                   kernel_size=patch_size, stride=patch_size)
        # N * Mixer Layer
        self.Nmixer_layer = nn.Sequential(*[Mixer_Layer(patch_size, hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(self.channel_dim)
        self.fully_connected = nn.Linear(self.channel_dim, self.class_num)
        self.global_pool = torch.mean
        ########################################################################

    def forward(self, data):
        ########################################################################
        x = self.Pre_patch(data)    # Per-patch Fully-connected
        x = x.flatten(2).transpose(1, 2)    # 展平
        x = self.Nmixer_layer(x)    # N * Mixer Layer
        x = self.norm(x)    # Pre_head Layer_norm
        x = self.global_pool(x, dim=1)  # Global Average Pooling
        x = self.fully_connected(x)     # Fully_connected
        return x
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            optimizer.zero_grad()   # 梯度清零
            out = model(data)
            loss = criterion(out, target.long())    # 损失函数（交叉熵）
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
            out = model(data)
            test_loss += criterion(out, target.long()) * len(data)    # 损失函数（交叉熵）
            out = out.argmax(dim=1).cpu().numpy().tolist()
            target = target.cpu().numpy().tolist()
            for i in range(len(out)):   # 计算正确个数
                if out[i] == target[i]:
                    num_correct += 1
            if 'test_num' in vars().keys():
                test_num += len(data)
            else:
                test_num = len(data)
        test_loss = test_loss / test_num
        accuracy = num_correct / test_num   # 正确率
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
    model = MLPMixer(patch_size=4, hidden_dim=128, depth=6).to(device)  # 参数自己设定，其中depth必须大于1

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=5*learning_rate, betas=(0.9, 0.999))
    optimizer = torch.optim.Adamax(model.parameters(), lr=10*learning_rate, betas=(0.5, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=5*learning_rate, momentum=0.99)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=1)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
