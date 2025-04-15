'''
导入必要的包
'''
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

'''
加载数据集，管理数据
'''

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
#输出数据集大小
len(train_dataset), len(test_dataset)
# 通过transform=transforms.ToTensor() 已经将图片数据从(28 * 28) 转为 (1 * 28 * 28) 添加了通道数量
# 同时通过此参数，将数值归一化到[0.0, 1.0]


'''
批量获取数据
'''
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#-----------------------------------------------------------------------------------------------------
'''
网络模型
'''
# 定义MLP网络，继承nn.Module
class MLP(nn.Module):
    # 初始化方法
    # input_size输入数据的维度
    # hidden_size 隐藏层的大小
    # num_classes 输出类别的数量
    def __init__(self, input_size, hidden_size, num_classes):
        # 调用父类初始化方法
        super(MLP, self).__init__()
        #定义第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        #定义激活函数
        self.relu = nn.ReLU()
        #定义第二个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #定义第3个全连接层
        self.fc3 = nn.Linear(hidden_size, num_classes)

    # 定义forward函数，向前传播函数
    # x 输入的数据
    def forward(self, x):
        # 第一层运算
        out = self.fc1(x)
        # 将上一步结果送入激活函数
        out = self.relu(out)
        # 将上一步结果送入fc2层
        out = self.fc2(out)
        # 同样将结果送入激活函数
        out = self.relu(out)
        # 将上一步结果传送入fc3层
        out = self.fc3(out)
        # 返回结果
        return out

'''
构造模型实例
'''
# 定义参数
input_size = 28*28 # 输入大小
hidden_size = 512  # 隐藏层大小
num_classes = 10 # 输出大小（类别数量）

#MLP实例
model = MLP(input_size, hidden_size, num_classes)

#-----------------------------------------------------------------------------------------------------
'''
损失函数和优化器
'''
# 分类问题，选择交叉熵损失函数
criterion = nn.CrossEntropyLoss()

#定义优化器，定义学习率,使用Adam优化器
learning_rate = 0.001 # 学习率
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#-----------------------------------------------------------------------------------------------------
'''
模型训练
'''
num_epochs = 10 #训练轮数
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将images转成向量
        images = images.reshape(-1, 28*28)
        # 将数据送入网络中
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)

        #首先将梯度清0
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        
        if(i + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


#-----------------------------------------------------------------------------------------------------
'''
测试效果
'''
#测试网络
with torch.no_grad():
    correct = 0
    total = 0
    # 从 test_loader 中循环读取测试数据
    for images, labels in test_loader:
        # 将images转换成向量
        images = images.reshape(-1, 28*28)
        # 将数据送入网络
        outputs = model(images)
        # 取数最大值对应的索引，即预测值
        _, predicted = torch.max(outputs.data, 1)
        # 累加label数
        total += labels.size(0)
        # 预测值与labels值对比，获取预测正确的数量
        correct += (predicted == labels).sum().item()

    # 打印最终的准确率
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#-----------------------------------------------------------------------------------------------------
'''
保存模型
'''
torch.save(model,"mnist_mlp_model.pkl")
