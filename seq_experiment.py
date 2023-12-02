#!/usr/bin/env python
# coding: utf-8

# # Part 2: Stock price prediction

# In[206]:


import pandas as pd
# daily_data = pd.read_csv('./data/dataset_FinBERT_sliding_window.csv')
# daily_data = pd.read_csv('./data/dataset_FinBERT.csv')
daily_data = pd.read_csv('./data/dataset_VADER.csv')


# In[223]:


# 选择特征和目标
# 保留原本的index，将'Date'列单独提取出来保存
date = daily_data['Date']
date = pd.to_datetime(date)

# features = daily_data_BERT.drop(['Date','P_news_pos', 'P_news_neg', 'P_op_pos', 'P_op_neg'], axis=1)
features = daily_data.drop(['Date'], axis=1)
# Open作为预测目标
target = daily_data['Open']
features.tail()


# In[208]:


target.tail()


# In[209]:


# normalization
from sklearn.preprocessing import MinMaxScaler

# Apply the MinMaxScaler to the features and target
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# fit_transform根据数据计算缩放参数
scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# 保存缩放参数
import joblib
joblib.dump(scaler_features, './model/scaler_features.pkl')
joblib.dump(scaler_target, './model/scaler_target.pkl')

# Create new DataFrames with the scaled features and target
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_target_df = pd.DataFrame(scaled_target, columns=['Open'])

print(scaled_features.shape)


# In[210]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates


# In[211]:


def create_sequences(features, targets, seq_length):
    """
    Create sequences of specified length from time series data.

    Args:
    features (np.array): The feature data.
    targets (np.array): The target data.
    seq_length (int): The length of the sequence.

    Returns:
    np.array: Sequences of features.
    np.array: Corresponding targets for each sequence.
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = targets[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# In[212]:

# exp
seq_lengths = [10, 15, 20, 25, 30, 35, 40, 45, 50]
rmse_results = []

for seq_length in seq_lengths:
    # 清空model
    import os
    import shutil

    folder = './model'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    print(f"The current sequence length is: {seq_length}")
    # 创建序列
    features_seq, target_seq = create_sequences(scaled_features, scaled_target, seq_length)

    # train_test_split随机划分

    # In[213]:

    train_features, test_features, train_target, test_target = train_test_split(
        features_seq, target_seq, test_size=0.2, random_state=42
    )

    val_features, test_features, val_target, test_target = train_test_split(
        test_features, test_target, test_size=0.5, random_state=42
    )

    # - 准备训练

    # In[214]:


    # Convert sequences to Tensor
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_target = torch.tensor(train_target, dtype=torch.float32)

    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_target = torch.tensor(val_target, dtype=torch.float32)

    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_target = torch.tensor(test_target, dtype=torch.float32)

    # 创建TensorDataset
    train_dataset = TensorDataset(train_features, train_target)
    val_dataset = TensorDataset(val_features, val_target)
    test_dataset = TensorDataset(test_features, test_target)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # In[215]:


    # 使用普通的LSTM模型，不使用注意力机制
    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.2):
            super(SimpleLSTM, self).__init__()
            self.hidden_size = hidden_size

            # LSTM层
            self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, 
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
            
            # 全连接层
            self.fc = nn.Linear(hidden_size, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # 打印lstm_out的形状
            # print(lstm_out.shape)
            # 取最后一个时间步的输出
            output = self.fc(lstm_out[:, -1, :])
            return output


    # In[217]:
    # 超参数
    input_dim = scaled_features_df.shape[1]  # 特征数量
    hidden_size = 100  # 隐藏状态中的特征数量，可以调整
    num_layers = 4    # 堆叠的LSTM层的数量
    output_dim = 1    # 输出维度的数量（预测一个值）

    # 使用SimpleLSTM
    model = SimpleLSTM(input_dim, hidden_size, num_layers, output_dim, dropout=0.2)
    # 使用AttentionLSTM
    # model = AttentionLSTM(input_dim, hidden_size, num_layers, output_dim, dropout=0.2)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # In[218]:
    # 训练模型
    num_epochs = 50
    best_val_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # 如果有多个GPU，你可以通过改变这里的数字来选择特定的GPU
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 将模型移动到指定的设备
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            # 将数据移动到指定的设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 将数据移动到指定的设备
                inputs = inputs.to(device)
                targets = targets.to(device)
            
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './model/best_model.pth')
            print('best_model updated at epoch {}, best_val_loss : {:.4f}'.format(epoch+1, best_val_loss))
            
        # 每5轮打印一次train loss和val loss
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 记录两个loss
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    

    # In[187]:
    # 加载最佳模型
    model.load_state_dict(torch.load('./model/best_model.pth'))

    # 计算新的测试集的大小
    test_size_new = int(len(features_seq) * 0.05)

    # 按时间顺序划分新的测试集
    test_features_new, test_target_new = features_seq[-test_size_new:], target_seq[-test_size_new:]

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        test_features_new = torch.tensor(test_features_new, dtype=torch.float32).to(device)
        test_predictions_new = model(test_features_new).cpu().numpy()

    # 反缩放预测值
    test_predictions_new = scaler_target.inverse_transform(test_predictions_new).flatten()

    # 反缩放真实目标值
    test_target_new = scaler_target.inverse_transform(test_target_new.reshape(-1, 1)).flatten()
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(test_target_new, test_predictions_new))
    rmse_results.append(rmse)

# 打印每个seq_length对应的RMSE
for seq_length, rmse in zip(seq_lengths, rmse_results):
    print(f'Seq_length: {seq_length}, RMSE: {rmse}')