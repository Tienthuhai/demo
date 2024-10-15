import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Bước 1: Tải dữ liệu và tiền xử lý
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Tiền xử lý dữ liệu
X_train = X_train / 255.0
X_test = X_test / 255.0

# Chuyển đổi nhãn thành dạng one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Bước 2: Xây dựng mô hình
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Chuyển đổi hình ảnh 28x28 thành vector
    Dense(128, activation='relu'),   # Lớp ẩn với 128 nơ-ron và hàm kích hoạt ReLU
    Dense(64, activation='relu'),    # Lớp ẩn với 64 nơ-ron
    Dense(10, activation='softmax')   # Lớp đầu ra với 10 nơ-ron (cho 10 nhãn từ 0 đến 9)
])

# Bước 3: Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Bước 4: Huấn luyện mô hình và lưu lại lịch sử
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Bước 5: Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Bước 6: Dự đoán và tìm các hình ảnh dự đoán
predictions = model.predict(X_test)

# Tìm chỉ số của các hình ảnh dự đoán đúng và sai
correct_indices = np.where(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))[0]
incorrect_indices = np.where(np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1))[0]

# Giới hạn số lượng hình ảnh hiển thị là 5
num_display = 5
correct_indices = correct_indices[:num_display]  # Chỉ lấy 5 hình đúng
incorrect_indices = incorrect_indices[:num_display]  # Chỉ lấy 5 hình sai

# Hiển thị hình ảnh dự đoán đúng
fig, axes = plt.subplots(2, num_display, figsize=(12, 8))

# Hiển thị hình ảnh dự đoán đúng
for i, index in enumerate(correct_indices):
    image = X_test[index]
    axes[0, i].imshow(image, cmap='gray')
    axes[0, i].axis('off')
    
    # Thêm nhãn đúng và nhãn dự đoán
    true_label = np.argmax(y_test[index])  # Nhãn thực tế
    prediction = np.argmax(predictions[index])  # Nhãn dự đoán
    axes[0, i].set_title(f"True: {true_label}\nPredicted: {prediction}")

axes[0, 0].set_ylabel("Correctly Predicted", fontsize=14)

# Hiển thị hình ảnh dự đoán sai
for i, index in enumerate(incorrect_indices):
    image = X_test[index]
    prediction = np.argmax(predictions[index])  # Nhãn dự đoán
    true_label = np.argmax(y_test[index])  # Nhãn thực tế

    axes[1, i].imshow(image, cmap='gray')
    axes[1, i].axis('off')
    
    # Thêm nhãn dự đoán và nhãn thực tế
    axes[1, i].set_title(f"Predicted: {prediction}\nTrue: {true_label}")

axes[1, 0].set_ylabel("Incorrectly Predicted", fontsize=14)

plt.tight_layout()
plt.show()

# Đồ thị liên hệ
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
