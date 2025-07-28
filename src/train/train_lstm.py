
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Đọc dữ liệu
run_df = pd.read_csv("data/data_train/RUN1.txt")
#boxing_df = pd.read_csv("BOXING.txt")
sit_df = pd.read_csv("data/data_train/SIT.txt")
nguoite_df = pd.read_csv('data/data_train/NGUOITE.txt')
walking_df = pd.read_csv('data/data_train/WALKING.txt')
X = []
y = []
no_of_timesteps = 10




# Xử lý dữ liệu ngồi
dataset = walking_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(0)

"""# Xử lý dữ liệu chạy
dataset = boxing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(1)


dataset = sit_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(2)

    
dataset = run_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(3)
"""
dataset = nguoite_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append(1)


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Chuyển đổi nhãn thành định dạng one-hot
y = to_categorical(y, num_classes=2)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation="sigmoid"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình
model.save("model/model_te_di.h5")
