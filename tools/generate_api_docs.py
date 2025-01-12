import pandas as pd
import inspect
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerasN.layers import *
from kerasN.models import *
from kerasN.activations import *
from kerasN.callbacks import *
from kerasN.datasets import *

def get_function_info(func):
    """함수 정보 추출"""
    doc = inspect.getdoc(func)
    params = inspect.signature(func)
    return {
        'name': func.__name__,
        'docstring': doc,
        'parameters': str(params)
    }

def get_class_info(cls):
    """클래스 정보 추출"""
    doc = inspect.getdoc(cls)
    methods = []
    attributes = []
    
    # 메서드 정보 수집
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_'):  # private 메서드 제외
            method_doc = inspect.getdoc(method)
            params = inspect.signature(method)
            methods.append({
                'method_name': name,
                'parameters': str(params),
                'docstring': method_doc
            })
    
    # 속성 정보 수집
    try:
        instance = cls()
        for name, value in inspect.getmembers(instance):
            if not name.startswith('_') and not callable(value):
                attributes.append(name)
    except:
        pass
    
    return {
        'class_name': cls.__name__,
        'docstring': doc,
        'methods': methods,
        'attributes': attributes
    }

def generate_api_docs():
    """API 문서 생성"""
    modules = {
        'Layers': {
            'classes': [Input, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout],
            'description': '신경망의 기본 구성 요소가 되는 레이어들'
        },
        'Models': {
            'classes': [Sequential],
            'description': '신경망 모델을 구성하는 클래스들'
        },
        'Activations': {
            'functions': ['relu', 'sigmoid', 'tanh', 'softmax'],
            'description': '활성화 함수들'
        },
        'Callbacks': {
            'classes': [EarlyStopping, ModelCheckpoint],
            'description': '모델 학습 과정을 제어하는 콜백들'
        },
        'Datasets': {
            'functions': [load_data],
            'description': '내장 데이터셋 로드 함수들'
        }
    }
    
    # DataFrame 생성을 위한 데이터 수집
    rows = []
    
    for category, info in modules.items():
        # 카테고리 설명 추가
        rows.append({
            'Category': category,
            'Name': '설명',
            'Type': 'Description',
            'Description': info['description'],
            'Parameters': '',
            'Methods/Returns': '',
            'Attributes': ''
        })
        
        # 클래스 정보 추가
        if 'classes' in info:
            for cls in info['classes']:
                cls_info = get_class_info(cls)
                rows.append({
                    'Category': category,
                    'Name': cls_info['class_name'],
                    'Type': 'Class',
                    'Description': cls_info['docstring'],
                    'Parameters': '',
                    'Methods/Returns': '\n'.join([f"{m['method_name']}{m['parameters']}\n{m['docstring']}" 
                                                for m in cls_info['methods']]),
                    'Attributes': ', '.join(cls_info['attributes'])
                })
        
        # 함수 정보 추가
        if 'functions' in info:
            for func_name in info['functions']:
                if isinstance(func_name, str):
                    rows.append({
                        'Category': category,
                        'Name': func_name,
                        'Type': 'Function',
                        'Description': '활성화 함수',
                        'Parameters': '-',
                        'Methods/Returns': '-',
                        'Attributes': '-'
                    })
                else:
                    func_info = get_function_info(func_name)
                    rows.append({
                        'Category': category,
                        'Name': func_info['name'],
                        'Type': 'Function',
                        'Description': func_info['docstring'],
                        'Parameters': func_info['parameters'],
                        'Methods/Returns': '-',
                        'Attributes': '-'
                    })
    
    # DataFrame 생성
    df = pd.DataFrame(rows)
    
    # Excel 파일로 저장
    output_dir = 'docs'
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, 'kerasN_api.xlsx')
    
    # 열 너비 자동 조정을 위한 writer 옵션 설정
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='API Documentation')
        worksheet = writer.sheets['API Documentation']
        
        # 각 열의 최대 길이에 따라 열 너비 조정
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.set_column(idx, idx, max_length + 2)
    
    print(f"API 문서가 '{excel_path}'에 저장되었습니다.")

if __name__ == '__main__':
    generate_api_docs() 