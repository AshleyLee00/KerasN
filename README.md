# KerasN

KerasN은 순수 Python/NumPy로 구현된 간단한 딥러닝 프레임워크입니다. Keras와 유사한 API를 제공하여 직관적인 딥러닝 모델 구현이 가능합니다.

## 설치 방법

pip를 사용하여 간단히 설치할 수 있습니다:

```bash
pip install kerasN
```

개발 버전을 설치하려면:

```bash
git clone https://github.com/yourusername/kerasN.git
cd kerasN
pip install -e .
```

## API 문서

### API 상세 문서

#### 모델 (Models)

##### Sequential
순차적 레이어 스택을 생성합니다.
```python
Sequential(layers=None)
```
- layers: 레이어 리스트 (선택사항)

메서드:
- `add(layer)`: 새로운 레이어를 모델에 추가
  - layer: 추가할 레이어 인스턴스
  - 반환값: None

- `compile(loss='mse', learning_rate=0.01)`: 모델 학습을 위한 설정
  - loss: 손실 함수 ('mse', 'categorical_crossentropy', 'binary_crossentropy')
  - learning_rate: 학습률
  - 반환값: None

- `fit(X, y, epochs=10, batch_size=32, validation_split=0.0, callbacks=None)`: 모델 학습 수행
  - X: 입력 데이터
  - y: 타겟 데이터
  - epochs: 전체 데이터셋을 반복 학습할 횟수
  - batch_size: 한 번에 처리할 데이터 샘플 수
  - validation_split: 검증 데이터셋 비율 (0.0 ~ 1.0)
  - callbacks: 콜백 함수 리스트
  - 반환값: History 객체 (학습 과정 기록)

- `predict(X)`: 입력 데이터에 대한 예측 수행
  - X: 예측할 입력 데이터
  - 반환값: 예측 결과 배열

- `summary()`: 모델 구조와 파라미터 정보 출력
  - 출력 정보: 레이어 종류, 출력 shape, 파라미터 수
  - 반환값: None

- `save_weights(filepath)`: 모델 가중치를 파일로 저장
  - filepath: 저장할 파일 경로
  - 반환값: None

- `load_weights(filepath)`: 저장된 가중치를 모델에 로드
  - filepath: 가중치 파일 경로
  - 반환값: None

- `evaluate(X, y)`: 모델 성능 평가
  - X: 테스트 데이터
  - y: 테스트 레이블
  - 반환값: 정확도 (float)

사용 예시:
```python
# 방법 1: 생성자에 레이어 리스트 전달
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=3, activation='relu'),
    Dense(10, activation='softmax')
])

# 방법 2: add() 메서드로 레이어 추가
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', learning_rate=0.001)

# 모델 학습
history = model.fit(X_train, y_train,
                   epochs=50,
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=[EarlyStopping(patience=5)])

# 모델 예측
predictions = model.predict(X_test)

# 모델 평가
accuracy = model.evaluate(X_test, y_test)
```

#### 레이어 (Layers)

##### Input
입력 레이어를 정의합니다.
```python
Input(shape)
```
- shape: 입력 데이터의 형태 (예: (28, 28, 1), (784,))

##### Dense
완전 연결층입니다.
```python
Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
```
- units: 출력 뉴런의 수
- activation: 활성화 함수 ('relu', 'sigmoid', 'tanh', 'softmax')
- use_bias: 편향 사용 여부
- kernel_initializer: 가중치 초기화 방법 ('glorot_uniform', 'random_normal', 'zeros')

##### Conv2D
2D 합성곱 층입니다.
```python
Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
       activation=None, use_bias=True)
```
- filters: 출력 필터 수
- kernel_size: 커널 크기 (정수 또는 튜플)
- strides: 스트라이드 크기
- padding: 패딩 방식 ('valid', 'same')
- activation: 활성화 함수
- use_bias: 편향 사용 여부

##### MaxPool2D
최대 풀링 층입니다.
```python
MaxPool2D(pool_size=2, strides=None, padding='valid')
```
- pool_size: 풀링 윈도우 크기
- strides: 스트라이드 (기본값: pool_size)
- padding: 패딩 방식 ('valid', 'same')

##### BatchNormalization
배치 정규화 층입니다.
```python
BatchNormalization(epsilon=1e-5, momentum=0.99)
```
- epsilon: 수치 안정성을 위한 작은 상수
- momentum: 이동 평균을 위한 모멘텀

#### 콜백 (Callbacks)

##### EarlyStopping
```python
EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```
- monitor: 모니터링할 지표 ('val_loss', 'val_accuracy', 'loss', 'accuracy')
- min_delta: 개선으로 인정할 최소 변화량
- patience: 개선이 없을 때 기다릴 에폭 수
- verbose: 출력 상세도 (0: 출력 없음, 1: 진행률 표시)
- mode: 모니터링 모드 ('auto', 'min', 'max')

#### 유틸리티 함수 (Utils)

##### to_categorical
레이블을 원-핫 인코딩으로 변환합니다.
```python
to_categorical(y, num_classes=None)
```
- y: 변환할 레이블 배열
- num_classes: 클래스 수 (None이면 자동 감지)

##### train_test_split
데이터를 학습셋과 테스트셋으로 분할합니다.
```python
train_test_split(X, y, test_size=0.2, shuffle=True)
```
- X: 특성 데이터
- y: 레이블 데이터
- test_size: 테스트셋 비율 (0.0 ~ 1.0)
- shuffle: 데이터 섞기 여부

##### evaluate
모델의 성능을 평가합니다.
```python
evaluate(model, X, y)
```
- model: 평가할 모델
- X: 테스트 데이터
- y: 테스트 레이블

#### 데이터셋 로더 (Datasets)

##### load_data
```python
load_data(name, normalize=True, reshape_to_image=False)
```
- name: 데이터셋 이름 ('mnist', 'fashion_mnist', 'cifar10', 'digits', 'iris', 'breast_cancer', 'wine')
- normalize: 데이터 정규화 여부
- reshape_to_image: 이미지 형태로 변환 여부 (이미지 데이터셋만 해당)

반환값:
- X: 특성 데이터
- y: 레이블 데이터

### 모델 컴파일 & 학습

```python
# 모델 컴파일
model.compile(loss='categorical_crossentropy', learning_rate=0.001)

# 모델 학습
history = model.fit(X_train, y_train,
                   epochs=50,
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=[EarlyStopping(patience=5)])
```

### 콜백

#### EarlyStopping
```python
from kerasN.callbacks import EarlyStopping

EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0)
```
- monitor: 모니터링할 지표
- min_delta: 최소 개선치
- patience: 개선이 없을 때 기다릴 에폭 수
- verbose: 출력 레벨
### 데이터셋 로드

```python
from kerasN.datasets import load_data
from kerasN.utils import to_categorical

# 데이터셋 로드
X, y = load_data('mnist', normalize=True, reshape_to_image=True)
y = to_categorical(y)
```

지원하는 데이터셋:

#### 이미지 데이터셋
- 'mnist': MNIST 손글씨 숫자 (28x28)
- 'fashion_mnist': Fashion-MNIST 의류 이미지 (28x28)
- 'cifar10': CIFAR-10 컬러 이미지 (32x32x3)
- 'digits': scikit-learn 손글씨 숫자 (8x8)

#### scikit-learn 데이터셋
- 'iris': 붓꽃 분류 데이터 (4 특성)
- 'breast_cancer': 유방암 진단 데이터 (30 특성)
- 'wine': 와인 분류 데이터 (13 특성)

매개변수:
- normalize: 데이터 정규화 여부 (기본값: True)
- reshape_to_image: 이미지 데이터를 이미지 형태로 변환 (기본값: False)

예시:
```python
# 이미지 데이터셋
X, y = load_data('mnist', normalize=True, reshape_to_image=True)
X, y = load_data('fashion_mnist', normalize=True, reshape_to_image=True)
X, y = load_data('digits', normalize=True, reshape_to_image=True)

# scikit-learn 데이터셋
X, y = load_data('iris', normalize=True)
X, y = load_data('breast_cancer', normalize=True)
X, y = load_data('wine', normalize=True)
```

### 유틸리티 함수

```python
from kerasN.utils import to_categorical, train_test_split, evaluate

# 원-핫 인코딩
y = to_categorical(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 평가
acc = evaluate(model, X_test, y_test)
```

## 주요 기능

- Sequential 모델 API
- 다양한 레이어 지원 (Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization 등)
- 콜백 시스템 (EarlyStopping)
- 데이터셋 로더 (MNIST, Fashion-MNIST, CIFAR-10, Digits 등)
- 학습 과정 시각화

## 예제

### MNIST 손글씨 숫자 분류

```python
from kerasN.models import Sequential
from kerasN.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input
from kerasN.callbacks import EarlyStopping
from kerasN.datasets import load_data
from kerasN.utils import to_categorical, train_test_split, evaluate

# 데이터 로드
X, y = load_data('mnist', normalize=True, reshape_to_image=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 모델 정의
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일 & 학습
model.compile(loss='categorical_crossentropy', learning_rate=0.001)
history = model.fit(X_train, y_train, 
                   epochs=50,
                   batch_size=32,
                   validation_split=0.2)
```

## 지원하는 데이터셋

- MNIST (손글씨 숫자)
- Fashion-MNIST (의류 이미지)
- CIFAR-10 (컬러 이미지)
- Digits (8x8 손글씨 숫자)

## 요구사항

- Python >= 3.7
- NumPy >= 1.19.2
- Matplotlib >= 3.3.2
- scikit-learn >= 0.23.2
- pandas >= 1.2.0
- tqdm >= 4.50.2

## 예제 실행

examples 디렉토리에 다양한 예제가 포함되어 있습니다:

```bash
python examples/mnist_example.py
python examples/fashion_mnist_example.py
python examples/cifar10_example.py
python examples/digits_example.py
```

## 프로젝트 구조

```
KerasN/
├── src/
│   └── kerasN/
│       ├── layers/       # 신경망 레이어 구현
│       ├── models/       # Sequential 모델 구현
│       ├── callbacks/    # 콜백 시스템
│       ├── datasets/     # 데이터셋 로더
│       └── utils/        # 유틸리티 함수
├── examples/             # 예제 코드
├── tests/               # 테스트 코드
├── setup.py            # 설치 설정
└── README.md           # 문서
```

## 라이선스

MIT License

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.