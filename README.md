# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset 
The objective of this experiment is to develop a Recurrent Neural Network (RNN) model using PyTorch to predict stock prices from historical closing-price data. The provided trainset.csv and testset.csv datasets are used for training and testing the model. The closing prices are normalized using MinMaxScaler, and sequences of 60 previous stock prices are created to predict the next stock price. The RNN model is trained using MSE Loss and the Adam optimizer, and its performance is evaluated by comparing the actual and predicted stock prices.



## DESIGN STEPS
### STEP 1: Load the trainset.csv and testset.csv datasets and extract the Close column as the stock-price data.


### STEP 2: Normalize the stock prices using MinMaxScaler, fitting the scaler only on the training data to avoid data leakage.



### STEP 3: Create time-series sequences using 60 previous closing prices as input and the next closing price as the target value.



### STEP 4: Convert the prepared sequences into PyTorch tensors and create a TensorDataset and DataLoader for efficient model training.



### STEP 5: Define an RNN model with an input layer, two RNN layers with 64 hidden units, and a fully connected output layer. Train the model using MSELoss and the Adam optimizer.



### STEP 6: Test the trained RNN model using testset.csv, convert the predictions back to the original price scale, and plot the training loss and actual versus predicted stock prices.





## PROGRAM

### Name: Yashwanth asv 

### Register Number: 212224230309

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset

!pip install torchinfo

df_train = pd.read_csv("C:/Users/admin/Downloads/trainset.csv")
df_test = pd.read_csv("C:/Users/admin/Downloads/testset.csv")

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)


x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel,self).__init__()
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

model=RNNModel()  #
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

summary(model, input_size=(64, 60, 1))

epochs=20
model.train()
train_losses=[]
for epoch in range(epochs):
    epoch_loss=0
    for x_batch,y_batch in train_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    train_losses.append(epoch_loss/len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}, Loss:{train_losses[-1]:.4f}')


print('Name: NETHRA.K  ')
print('Register Number: 212224230184 ')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:  NETHRA.K               ')
print('Register Number: 212224230184    ')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1435" height="607" alt="image" src="https://github.com/user-attachments/assets/c98e6661-4cbf-4caa-bb16-fe0ad59eac28" />



<img width="1557" height="826" alt="image" src="https://github.com/user-attachments/assets/578180b6-a4f8-46b4-a47f-9ab45572e08a" />



## True Stock Price, Predicted Stock Price vs time

<img width="1512" height="797" alt="image" src="https://github.com/user-attachments/assets/4e504808-d542-4d6a-b71f-4de2dd05a2f3" />


### Predictions
<img width="1251" height="135" alt="image" src="https://github.com/user-attachments/assets/c8143dbe-c2da-473c-bd13-87923b7bee62" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been implemented successfully.
