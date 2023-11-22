#!/usr/bin/env python
# coding: utf-8

# # Part 1: Sentiment Analysis

# In[ ]:


import pandas as pd

# Load the CSV file
file_path = './data/NEWS_YAHOO_stock_prediction.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()


# In[ ]:


# Step 1: Remove unnecessary column
data.drop(columns=['Unnamed: 0'], inplace=True)

# Step 2: Remove duplicate texts
data.drop_duplicates(subset=['title', 'content'], inplace=True)

# Step 3: Remove rows with large amount of spaces or empty texts in 'title' and 'content'
data = data[~data['title'].str.isspace()]
data = data[~data['content'].str.isspace()]
data.dropna(subset=['title', 'content'], inplace=True)

# Check the dataframe after these preprocessing steps
data.info()

# Step 5: Check for invalid numeric data
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data[numeric_columns].describe()


# In[ ]:


# (optional) set proxy
import subprocess
import os

result = subprocess.run('bash -c "source ~/clash_dir/set && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
output
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


# In[ ]:


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the FinBERT model and tokenizer
checkpoint = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# Create a pipeline for sentiment analysis
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True, device=0)


# In[ ]:


# Function to apply sentiment analysis to a dataframe
def apply_sentiment_analysis(df, nlp, text_column='content'):
    """
    Apply sentiment analysis to a column in a dataframe.
    
    Args:
    df (pd.DataFrame): Dataframe containing the text data.
    nlp (pipeline): HuggingFace pipeline for sentiment analysis.
    text_column (str): Name of the column containing text data.

    Returns:
    pd.DataFrame: Dataframe with a new column 'sentiment' containing the analysis results.
    """
    # Apply sentiment analysis to each row in the text column
    sentiments = []
    for text in df[text_column]:
        try:
            result = nlp(text)
            sentiments.append(result[0]['label'])
        except Exception as e:
            print(f"Error in processing text: {e}")
            sentiments.append('Error')

    # Add the sentiments as a new column in the dataframe
    df['sentiment'] = sentiments
    return df


# In[ ]:


from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def apply_sentiment_analysis_parallel(df, nlp, text_column='content', batch_size=10):
    """
    Apply sentiment analysis in parallel to a column in a dataframe.

    Args:
    df (pd.DataFrame): Dataframe containing the text data.
    nlp (pipeline): HuggingFace pipeline for sentiment analysis.
    text_column (str): Name of the column containing text data.
    batch_size (int): Number of texts to process in parallel.

    Returns:
    pd.DataFrame: Dataframe with a new column 'sentiment' containing the analysis results.
    """
    # Define a function to process a batch of texts
    def process_batch(texts):
        return [nlp(text)[0]['label'] for text in texts]

    # Break the texts into batches
    batches = [df[text_column][i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Process batches in parallel
    sentiments = []
    with ThreadPoolExecutor() as executor:
        for batch_result in tqdm(executor.map(process_batch, batches), total=len(batches)):
            sentiments.extend(batch_result)

    # Add the sentiments as a new column in the dataframe
    df['sentiment'] = sentiments
    return df


# In[ ]:


# Example usage of the function
# Note: You will run this on your local machine as it requires GPU support
sample_texts = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
sample_df = pd.DataFrame(sample_texts, columns=['content'])
apply_sentiment_analysis(sample_df, nlp)


# In[ ]:


# Assuming the apply_sentiment_analysis function is defined as shown previously

# Step 1: Apply sentiment analysis to the dataset
# This step should be done on your local machine due to the requirement of GPU support
data = apply_sentiment_analysis_parallel(data, nlp)

# Step 2: Prepare data for the prediction model
# Here we'll assume the sentiment analysis has been applied and 'sentiment' column is added to the data

# We might want to convert sentiments to numerical values for model training
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
data['sentiment_numeric'] = data['sentiment'].map(sentiment_mapping)


# In[ ]:


# Example code to save the processed DataFrame to a CSV file
data.to_csv('./data/dataset_with_sentiment.csv', index=False)


# In[ ]:


# Adjust display settings for better visualization of samples
pd.set_option('display.max_colwidth', 200)  # Adjust the width to fit longer texts

# Display some random samples with formatted output
sample_data = data.sample(n=10)[['content', 'sentiment']]

# Print each sample in a more readable format
for index, row in sample_data.iterrows():
    print(f"Sample {index}:")
    print(f"Content: {row['content']}")
    print(f"Sentiment: {row['sentiment']}\n")


# In[ ]:


# Assuming 'data' is your DataFrame with 'sentiment' and 'label' columns
# Calculate the proportion of each sentiment category
sentiment_counts = data['sentiment'].value_counts(normalize=True) * 100

# Calculate the proportion of each label
label_counts = data['label'].value_counts(normalize=True) * 100

# Print the results
print("Sentiment Distribution (%):")
print(sentiment_counts)
print("\nLabel Distribution (%):")
print(label_counts)

# For additional insights, we can also look at the cross-tabulation of sentiment and label
crosstab = pd.crosstab(data['sentiment'], data['label'], normalize='index') * 100
print("\nCross-Tabulation of Sentiment and Label (%):")
print(crosstab)


# In[8]:


# read for existed csv
import pandas as pd
data = pd.read_csv('./data/dataset_with_sentiment.csv')

# Convert the 'Date' column to datetime format and sort the dataframe by 'Date'
data['Date'] = pd.to_datetime(data['Date'])
data_sorted = data.sort_values(by='Date')


# In[9]:


# 按 'Date' 和 'sentiment' 分组，然后计算每个类别的 category 为news和opinion的数量
category_news_per_day_sentiment = data_sorted[data_sorted['category'] == 'news'].groupby(['Date', 'sentiment']).size().unstack().fillna(0)
category_opinion_per_day_sentiment = data_sorted[data_sorted['category'] == 'opinion'].groupby(['Date', 'sentiment']).size().unstack().fillna(0)
# 分别计算news和opinion的total
category_news_total_per_day_sentiment = data_sorted[data_sorted['category'] == 'news'].groupby(['Date']).size()
category_opinion_total_per_day_sentiment = data_sorted[data_sorted['category'] == 'opinion'].groupby(['Date']).size()

data_sorted = data_sorted.set_index('Date')
data_sorted['P_news_pos'] = category_news_per_day_sentiment['Positive'].reindex(data_sorted.index) / category_news_total_per_day_sentiment.reindex(data_sorted.index)
data_sorted['P_news_neg'] = category_news_per_day_sentiment['Negative'].reindex(data_sorted.index) / category_news_total_per_day_sentiment.reindex(data_sorted.index)
data_sorted['P_op_pos'] = category_opinion_per_day_sentiment['Positive'].reindex(data_sorted.index) / category_opinion_total_per_day_sentiment.reindex(data_sorted.index)
data_sorted['P_op_neg'] = category_opinion_per_day_sentiment['Negative'].reindex(data_sorted.index) / category_opinion_total_per_day_sentiment.reindex(data_sorted.index)
data_sorted = data_sorted.reset_index()


# In[12]:


daily_data = data_sorted.groupby('Date').last()

# Shift the 'Open' column to get the next day's opening price
daily_data['Next_Open'] = daily_data['Open'].shift(-1)

# Drop the last row as it will not have a 'Next_Open' value
daily_data = daily_data[:-1]


# In[19]:


date_to_query = pd.to_datetime('2016-10-28')
daily_data.loc[(date_to_query)]


# In[18]:


date_to_query = pd.to_datetime('2016-10-28')
data_sorted.loc[data_sorted['Date'] == date_to_query]


# In[17]:


date_to_query = pd.to_datetime('2020-01-23')
daily_data.loc[(date_to_query)]


# In[23]:


date_to_query = pd.to_datetime('2018-05-06')
category_to_query = 'news'
data_sorted.loc[(data_sorted['Date'] == date_to_query) & (data_sorted['category'] == category_to_query)]


# In[24]:


daily_data['P_news_neg'].fillna(0, inplace=True)
daily_data['P_news_pos'].fillna(0, inplace=True)
daily_data['P_op_neg'].fillna(0, inplace=True)
daily_data['P_op_pos'].fillna(0, inplace=True)


# In[25]:


# 不直接删除，而是存到新的df中
columns_to_keep = [col for col in daily_data.columns if col not in ['ticker', 'Adj Close', 'sentiment', 'sentiment_numeric', 'title', 'category', 'content', 'label']]
daily_data_selected = daily_data[columns_to_keep]


# In[28]:


daily_data_selected.to_csv('./data/dataset_for_model.csv', index=False)


# In[ ]:


# 选择特征和目标
features = daily_data_selected.drop('Next_Open', axis=1)
target = daily_data_selected['Next_Open']

# normalization
from sklearn.preprocessing import MinMaxScaler

# Apply the MinMaxScaler to the features and target
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create new DataFrames with the scaled features and target
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_target_df = pd.DataFrame(scaled_target, columns=['Next_Open'])


# In[32]:


scaled_features_df.tail(), scaled_target_df.tail()


# # Part 2: Stock price prediction

# In[ ]:


import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimeSeriesTransformerForPrediction
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 将数据转换为Tensor
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
target_tensor = torch.tensor(scaled_target, dtype=torch.float32).unsqueeze(-1)

past_observed_mask = torch.ones_like(features_tensor, dtype=torch.bool)

# 划分训练集和测试集
train_features, test_features, train_target, test_target = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=42)

# 创建TensorDataset
train_dataset = TensorDataset(train_features, train_target)
test_dataset = TensorDataset(test_features, test_target)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


# 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# 设置超参数
num_epochs = 10  # 定义训练的迭代次数

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # # 分离特征和标签
        # inputs, labels = batch

        # # 根据inputs准备past_observed_mask
        # # 如果没有缺失值，可以使用全为True的张量
        # past_observed_mask = torch.ones_like(inputs, dtype=torch.bool)

        # # 重置梯度
        # optimizer.zero_grad()
        
        # # 前向传播，确保使用正确的参数
        # outputs = model(inputs, past_observed_mask)

        # # 计算损失
        # loss = criterion(outputs, labels)

        # # 后向传播和优化
        # loss.backward()
        # optimizer.step()

        # 累计损失
        # total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 保存模型
torch.save(model.state_dict(), "model.pth")


# In[ ]:


# 预测和评估
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        predictions.extend(outputs.numpy())
        actuals.extend(labels.numpy())

# 假设 predictions 和 actuals 是模型的预测结果和实际目标值
predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
actuals_tensor = torch.tensor(actuals, dtype=torch.float32)

# 将预测结果和实际值转换回原始尺度
predicted_prices = scaler_target.inverse_transform(predictions_tensor.numpy())
actual_prices = scaler_target.inverse_transform(actuals_tensor.numpy())

# 可视化预测结果
plt.figure(figsize=(10,6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()

