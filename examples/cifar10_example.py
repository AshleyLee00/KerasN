import numpy as np
from kerasN.models import Sequential
from kerasN.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input
from kerasN.callbacks import EarlyStopping
from kerasN.datasets import load_data
from kerasN.utils import to_categorical, train_test_split, evaluate
import matplotlib.pyplot as plt

# 데이터 로드
print("Loading CIFAR-10 dataset...")
X, y = load_data('cifar10', normalize=True, reshape_to_image=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f"Data loaded successfully!")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Input shape: {X_train.shape[1:]}")

# 모델 정의
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Conv2D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Conv2D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
]

model.compile(loss='categorical_crossentropy', learning_rate=0.001)
history = model.fit(X_train, y_train, 
                   epochs=100,
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=callbacks)

history.plot()
plt.show()

acc = evaluate(model, X_test, y_test)
print(f"\nTest accuracy: {acc:.2%}") 