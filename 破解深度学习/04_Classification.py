import torch
import torchvision

'''
数据预处理
'''
#ToTensor()将数据集中的图像数据转换为Pytorch张量，这使得我们可以直接将图像作为模型输入
transformation = torchvision.transforms.ToTensor()
# 获取数据集
train_dataset = torchvision.datasets.MNIST(root='./mnist/data', train=True, transform=transformation, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/data', train=False, transform=transformation,download=True)

# 指定批量大小
batch_size = 64
# 将数据载入数据加载器中
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
绘图，验证数据的正确性
'''
import matplotlib.pyplot as plt
# 检查输入数据的正确性
for images, labels in train_dataloader:
    # 打印出images 和 labels 的形状
    print('images.shape:', images.shape, 'labels.shape:', labels.shape)
    print(labels[0]) #打印出labels的第一个元素
    plt.imshow(images[0][0], cmap='gray') #显示该图片
    plt.show()
    break

'''
定义模型，这里定义一个简单的前馈神经网络
'''
import torch.nn as nn
# 定义一个名为Model的类，继承nn.Module
class Model(nn.Module):
    # 定义__init__函数，初始化模型参数
    def __init__(self, input_size, output_size):
        super().__init__()
        # 定义一个线性层，其输入大小为input_size，输出大小为output_size
        self.linear = nn.Linear(input_size, output_size)
    # 定义前向传播函数，用于向前传播
    def forward(self, x):
        # 计算线性层的输出，即模型的logits值
        logits = self.linear(x)
        # 返回模型的logits值
        return logits

# 定义超参数
input_size = 28*28
output_size = 10
model = Model(input_size, output_size)

'''
创建交叉熵损失函数和优化器
'''
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 定义随机梯度下降优化器，其中model.parameters()获取模型所有可学习的参数
# lr表示学习率，它控制优化器在更新参数时的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 定义evaluate() 函数，用于评估模型在指定数据集上的准确率
def evaluate(model, data_loader):
    # 将模型设为评估模式，即关闭Dropout和批归一化等随机性操作
    model.eval()
    # 定义变量用于记录分类正确的样本数量和总样本数量
    correct = 0
    total = 0
    # 在上下文管理器中使用 torch_no_grad()，以避免在评估模型时计算梯度和更新模型参数
    with torch.no_grad():
        # 遍历数据加载器中的每个batch
        for x, y in data_loader:
            # 将输入数据x变形为二维张量，第一维是batch_size，第二维是input_size
            x = x.view(-1, input_size)
            # 计算模型的输出，即logits值
            logits = model(x)
            # 从logits中选取最大值作为预测值，并返回预测值的索引
            _, predicted = torch.max(logits.data, 1)
            # 更新总样本数量
            total += y.size(0)
            # 更新分类正确的样本数量
            correct += (predicted == y).sum().item()
        #计算正确率，并返回
        return correct / total
    
'''
模型训练
'''
# 对神经网络进行多次训练迭代
for epoch in range(10):
    # 遍历训练集数据加载器中的每个batch
    for images, labels in train_dataloader:
        # 将图像和标签转为张量
        images = images.view(-1, 28 * 28)
        labels = labels.long()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化，即计算梯度并更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #在测实集上评估模型的准确率，并输出
    accuracy = evaluate(model, test_dataloader)
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, epoch, loss.item(), accuracy))

        
