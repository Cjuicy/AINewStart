import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 从CSV加载数据
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

# 分离特征和标签
y_train = train_data["label"].values
x_train = train_data.drop("label", axis=1).values
x_test = test_data.values

# 划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)