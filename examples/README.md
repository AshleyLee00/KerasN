# KerasN 예제

이 디렉토리는 KerasN을 사용한 다양한 예제 코드를 포함하고 있습니다.

## 예제 목록

### 1. MNIST 손글씨 숫자 분류 (mnist_example.py)
MNIST 데이터셋을 사용한 CNN 모델 구현 예제입니다.

```bash
python mnist_example.py
```

주요 특징:
- CNN 아키텍처 사용
- BatchNormalization과 Dropout을 통한 정규화
- EarlyStopping과 ModelCheckpoint 콜백 사용
- 모델 가중치 저장 및 로드 기능 시연

### 2. Fashion-MNIST 의류 분류 (fashion_mnist_example.py)
Fashion-MNIST 데이터셋을 사용한 의류 이미지 분류 예제입니다.

```bash
python fashion_mnist_example.py
```

주요 특징:
- 더 복잡한 CNN 아키텍처
- 데이터 전처리 및 정규화
- 학습 과정 시각화

## 실행 방법

1. 프로젝트 루트 디렉토리에서 실행:
```bash
cd examples
python mnist_example.py  # 또는 다른 예제 파일
```

2. 다른 디렉토리에서 실행:
```bash
python path/to/examples/mnist_example.py
```

## 데이터셋

예제 실행 시 필요한 데이터셋은 자동으로 다운로드됩니다. 데이터는 `kerasN/datasets/data` 디렉토리에 저장됩니다.

## 의존성
- numpy
- matplotlib (시각화용)
- scikit-learn (데이터셋 로드)

## 주의사항
- 첫 실행 시 데이터셋 다운로드로 인해 시간이 걸릴 수 있습니다
- 학습된 모델의 가중치는 `weights` 디렉토리에 저장됩니다
- GPU가 없어도 실행 가능하지만, 학습 시간이 오래 걸릴 수 있습니다

## 커스터마이징

예제 코드를 수정하여 다음과 같은 실험을 해볼 수 있습니다:
- 모델 아키텍처 변경
- 하이퍼파라미터 조정
- 다른 활성화 함수 사용
- 콜백 설정 변경
- 데이터 전처리 방식 수정

## 문제 해결

1. ImportError 발생 시:
```bash
# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH=/path/to/kerasN:$PYTHONPATH
```

2. 메모리 부족 시:
- 배치 크기를 줄여보세요
- 더 작은 모델을 사용해보세요

3. 학습이 너무 느린 경우:
- 에폭 수를 줄여보세요
- 더 작은 데이터셋으로 테스트해보세요 