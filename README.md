# SKN09 - EDA 3Team

> **SK Networks AI CAMP 9기**  
> **개발기간:** 2025.01.22 ~ 2025.02.02  
> **팀명:** 경제 콜럼버스  

---

## 📢 Team Introduction (**경제 콜럼버스 (SKN09-eda-3Team)**)
| 이름      | GitHub ID                          |
|-----------|------------------------------------|
| 🧑‍💻 김정훈  | [@Zayden0815](https://github.com/Zayden0815) |
| 👩‍💻 김하늘  | [@nini12091](https://github.com/nini12091)        |
| 👩‍💻 이광운  | [@Leegwangwoon](https://github.com/Leegwangwoon)          |

---

## 🎯 Project Introduction (프로젝트 개요)

### 프로젝트명
**글로벌 자산 시장 동향 분석**

### 프로젝트 소개
📈 **주식(S&P 500, KOSPI)**, ₿ **비트코인(BTC)**, 🪙 **금(Gold)**, 💵 **환율(USD)**의 시계열 데이터를 활용하여 **경제적 변수와 자산 가격 간의 관계를 탐구**하고, 각 자산 간 연관성을 **시각화 자료**를 통해 확인합니다.

### 프로젝트 주제 선정 배경
금융 시장은 글로벌 경제와 다양한 자산군의 상호작용에 의해 크게 영향을 받습니다. 주식, 비트코인, 금, 환율과 같은 주요 자산들은 서로 밀접하게 연관되어 있으며, 전통 자산(금, 주식)과 디지털 자산(비트코인)의 상호작용 및 역할 변화를 이해하는 것이 중요합니다.

### ✅ 프로젝트 목표
1. **자산 간 상관관계 분석**  
   주요 자산들의 가격 데이터를 분석하여 **각 자산 간의 상관관계를 관찰**하고, 특정 경제적 사건(예: 코로나) 기간의 반응을 **시각적으로 표현**합니다.

2. **경제적 사건에 대한 관찰**  
   과거 경제적 사건이 자산 가격에 미친 영향을 분석하여 **특정 사건이 자산군에 미치는 패턴과 상관관계를 도출**하고, 이를 통해 **경제적 사건과 자산 간의 관계를 심층적으로 이해**합니다.

---

## 🚀 Getting Started (시작 가이드)

### 주요 사용 라이브러리
```python
# 예시 코드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## 실험 결과
# Bitcoin Price Prediction - Model Comparison
## 모델 비교
| **모델**                | **사용된 알고리즘**         | **하이퍼파라미터 튜닝** | **RMSE**   | **주요 특징** |
|-----------------------|-------------------------|----------------------|-----------|-------------|
| **1. XGBoost (기본)**   | XGBoost                  | 없음                   | 𝑅𝑀𝑆𝐸: 839.1638782705273 | - 기본 XGBoost 회귀 모델<br> - 기본 학습률과 트리 개수 |
| **2. XGBoost (GridSearch)** | XGBoost                  | `n_estimators`, `learning_rate` (GridSearchCV 사용) | 𝑅𝑀𝑆𝐸: 50.33194281465232 | - `GridSearchCV`로 최적 파라미터 튜닝<br> - 교차 검증을 통한 최적화 |
| **3. Random Forest**     | 랜덤 포레스트 (RandomForestRegressor) | 없음                   | 𝑅𝑀𝑆𝐸: 35.96528975058352 | - 앙상블 모델<br> - 여러 트리를 결합하여 예측<br> - 비선형 관계에 강함 |

## 🖥️ Server Specifications (서버 사양)

| 사양               | 서버 1                      | 서버 2(랩탑)   수정하기                |
|--------------------|-----------------------------|-----------------------------|
| **CPU**            | Intel Core i7 11th Gen       | Intel Core i5 13th Gen           |
| **RAM**            | 32GB                       | 32GB                    |
| **Storage**        | 512GB SSD                   |512GB SSD                       |
| **Operating System**| window                  | window                 |
| **GPU**            | NVIDIA RTX 3050           | iris(R) Xe Graphics            |
