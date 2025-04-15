'''
数据生成
'''
import numpy as np 
import torch

# 设置随机数种子，使得每次运行代码生成的数据相同
np.random.seed(42)

# 生成随机数据，w为2，b为1
x = np.random.randn(100, 1) # 创建100个随机数，作为x
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

# 将数据转换为PyTorch张量
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 设置超参数
learning_rate = 0.1
num_epochs = 1000

# 初始化参数，可以使用常数、随机数或预训练参数等
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

'''
开始训练
'''
# 开始训练
for epoch in range(num_epochs):
    # 计算预测值
    y_pred = w * x_tensor + b
    
    # 计算损失
    loss = ((y_pred - y_tensor)**2).mean()

    #反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        #更新参数
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        #清空梯度
        w.grad.zero_()
        b.grad.zero_()

#输出训练后的参数，与数据生成时设置的参数一本一致
print('w:',w)
print('b:',b)

'''
进行可视化绘图
'''
import matplotlib.pyplot as plt
#绘制散点图和直线
plt.scatter(x, y, label='o')
plt.plot(x_tensor.numpy(), y_pred.detach().numpy())
plt.show()



