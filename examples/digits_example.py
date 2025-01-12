from kerasN.models import Sequential
from kerasN.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input
from kerasN.callbacks import EarlyStopping
from kerasN.datasets import load_data
from kerasN.utils import to_categorical, train_test_split, evaluate
import matplotlib.pyplot as plt

# 데이터 로드
print("Loading Digits dataset...")
X, y = load_data('digits', normalize=True, reshape_to_image=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 더 깊은 모델로 수정
model = Sequential([
    Input(shape=(8, 8, 1)),
    # 첫 번째 블록
    Conv2D(16, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(16, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    
    # 두 번째 블록
    Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# 모델 구조 출력
model.summary()

# 콜백 설정
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,  # 최소 개선치
        patience=5,        # 5 에폭 동안 개선이 없으면 중단
        verbose=1
    )
]

# 모델 컴파일 & 학습
model.compile(loss='categorical_crossentropy', learning_rate=0.0005)
history = model.fit(X_train, y_train, 
                   epochs=100,
                   batch_size=16,
                   validation_split=0.2, 
                   callbacks=callbacks)

# 결과 시각화
history.plot()
plt.show()

# 테스트 세트 평가
acc = evaluate(model, X_test, y_test)
print(f"\n테스트 정확도: {acc:.2%}") 