import numpy as np
from kerasN.models import Sequential
from kerasN.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input
from kerasN.callbacks import EarlyStopping
from kerasN.datasets import load_data
from kerasN.utils import to_categorical, train_test_split, evaluate
import matplotlib.pyplot as plt

# 데이터 로드
print("Loading MNIST dataset...")
X, y = load_data('mnist', normalize=True, reshape_to_image=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f"Data loaded successfully!")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Input shape: {X_train.shape[1:]}")  # 입력 shape 출력

# 모델 정의
model = Sequential([
    Input(shape=(28, 28, 1)),  # 입력 레이어 추가
    Conv2D(32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Conv2D(32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# 모델 구조 출력
model.summary()

# 콜백 설정
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )
]

# 모델 컴파일 & 학습
model.compile(loss='categorical_crossentropy', learning_rate=0.001)
history = model.fit(X_train, y_train, 
                   epochs=50,
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=callbacks)

# 결과 시각화
history.plot()
plt.show()

# 테스트 세트 평가
acc = evaluate(model, X_test, y_test)
print(f"\nTest accuracy: {acc:.2%}") 