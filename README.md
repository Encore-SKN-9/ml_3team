# SKN09 - ML 3Team

> **SK Networks AI CAMP 9기**  
> **개발기간:** 2025.01.22 ~ 2025.02.02  
> **팀명:** 경제 콜럼버스  

---

## 📢 Team Introduction (**경제 콜럼버스 (SKN09-ML-3Team)**)
| 이름      | GitHub ID                          |
|-----------|------------------------------------|
| 🧑‍💻 김정훈  | [@Zayden0815](https://github.com/Zayden0815) |
| 👩‍💻 김하늘  | [@nini12091](https://github.com/nini12091)        |
| 👩‍💻 이광운  | [@Leegwangwoon](https://github.com/Leegwangwoon)          |

---

## 🎯 Project Introduction (프로젝트 개요)

### 프로젝트명
**머신러닝을 적용한 주식(S&P500), 가상화폐(비트코인) 가격 예측**

### 프로젝트 소개
📈 **주식(S&P 500)**, ₿ **비트코인(BTC)** 의 시계열 데이터를 기반으로 각 자산의 가격을 머신러닝 기법을 활용해 **예측**하고 **시각화 자료**를 통해 확인

### 프로젝트 주제 선정 배경
![image](https://github.com/user-attachments/assets/b2b0e1ef-9a52-4682-af74-554996a503db)
직전 프로젝트인 EDA 프로젝트의 결과로 분석한 경제 요소지표중 두 개의 상관관계가 상대적으로 높다는 것을 파악했고, 이를 기반으로 두 데이터를 활용하여 가격을 예측하는 방향으로 후속 프로젝트 주제를 선정

### ✅ 프로젝트 목표
1. **자산의 가격 분석**  
   주요 자산들의 가격 데이터를 기반하여 **각 자산 간의 가격을 예측**하고, 결과를 **시각적으로 표현**하여 예측 결과 분석

2. *주요 자산 가격의 추이**를 상세 분석
- **자산 간 가격 관계**를 탐색하고 미래 가격을 예측
- 다양한 기법을 적용하여 모델 성능을 최적화
---

## 🚀 Getting Started 

### 사용 모델 
```python
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
```
---
### 코드 이슈: `GradientBoostingRegressor` 버전 이슈

#### 문제 발생
`from sklearn.ensemble import GradientBoostingRegressor`에서 버전 호환성 문제로 오류가 발생. 이 문제는 **최신버전**의 `scikit-learn`에서 발생할 수 있습니다.

#### 해결 방법

1. **특정 버전 설치**  
   `scikit-learn` 1.3.2 버전에서 동작이 확인되었습니다. 해당 버전을 설치하려면 아래 명령어를 사용하세요:
   
   ```bash
   pip install scikit-learn==1.3.2
   
2. **파이썬 가상 환경 생성 후 설치**  
   파이썬 가상 환경을 새로 만들어서 해당 버전을 설치하면 충돌을 피할 수 있습니다. 아래 절차를 따라 새로운 가상 환경을 생성하고 필요한 라이브러리를 설치하세요:

   ```bash
   # 가상 환경 생성
   python -m venv myenv

   # 가상 환경 활성화 (Windows)
   myenv\Scripts\activate

   # 가상 환경 활성화 (Mac/Linux)
   source myenv/bin/activate

   # 필요한 라이브러리 설치
   pip install scikit-learn==1.3.2
---
## 모델 학습 및 평가 결과
### XGBoost
![XGBoost](image/xgboost.png)
### XGBoost - 최적화
![XGBoost - 최적화](image/xgboost_opt.png)
### Random Forest
![Random Forest](image/randomforest_gw.png)

---
### Light Gradient Boosting
![1](https://github.com/user-attachments/assets/c4410215-236a-4512-9687-00238220e4cb)
![2](https://github.com/user-attachments/assets/568675da-462a-4cad-a174-44807719fa2e)

📉 독립 변수로 현재 시점의 데이터만 학습한 결과
- 과거 데이터를 고려하지 않아 예측 성능이 매우 낮음
- R² 값이 음수로, 모델이 데이터 패턴을 학습하지 못함

#### 📌 **초기 성능:**
| 자산  | MAE       | RMSE      | R²    |
|------|----------|----------|------|
| BTC  | 13,570.67 | 16,148.75 | -0.3708 |
| S&P 500 | 448.28   | 501.18   | -2.2996 |

❌ 과거 데이터를 고려하지 않은 모델은 패턴을 제대로 학습하지 못함

### Light Gradient Boosting - Lag 적용 (5)
![5](https://github.com/user-attachments/assets/59f1644d-522c-4bf9-aebe-5f29130ab79e)
![6](https://github.com/user-attachments/assets/60d07778-bb78-4b6a-8d67-cf7505a2d02a)

📈 과거 5개의 데이터를 입력 변수로 추가하여 학습
- 모델의 R² 값이 크게 증가하며, 패턴 학습 능력이 향상됨

#### 📌 **Lag=5 성능:**
| 자산  | MAE   | RMSE   | R²   |
|------|------|------|------|
| BTC  | 2,336.33 | 3,358.64 | 0.8576 |
| S&P 500 | 57.80  | 73.63  | 0.9134 |


### Light Gradient Boosting - Lag 적용 (10)
![9](https://github.com/user-attachments/assets/4a2e59d8-adc3-4f83-a487-3c23fec375ae)
![10](https://github.com/user-attachments/assets/0aefcf81-a72a-41e4-b898-94bba05e69a2)

📊 Lag 변수를 10개로 확장하여 학습
- BTC의 경우 RMSE 증가, SNP는 소폭 개선
- 일정 수준 이상으로 Lag을 추가하면 성능 개선이 제한적일 수 있음

#### 📌 **Lag=10 성능:**
| 자산  | MAE   | RMSE   | R²   |
|------|------|------|------|
| BTC  | 3,098.08 | 3,670.99 | 0.8298 |
| S&P 500 | 58.74  | 74.52  | 0.9113 |

📢 적절한 Lag 개수를 선택하는 것이 중요함

### Light Gradient Boosting - 하이퍼파라미터 튜닝
![13](https://github.com/user-attachments/assets/1e9057e9-6abc-49bd-95a8-77d4691c5c57)
![15](https://github.com/user-attachments/assets/33d251c1-f4be-4092-9eb1-fc019d28f943)

✨ 최적의 하이퍼파라미터를 찾기 위해 Grid Search를 활용
- R² 값이 BTC: 0.8372, SNP: 0.9448까지 증가하여 높은 정확도를 달성
= 모델 최적화 과정에서 일부 RMSE 값이 비정상적으로 증가하는 현상이 발생하여 추가적인 검토 필요

#### 📌 **최적 모델 성능:**
| 자산  | MAE   | RMSE   | R²   |
|------|------|------|------|
| BTC  | 3,006.41 | 12,891.27 | 0.8372 |
| S&P 500 | 47.46  | 3,449.39 | 0.9448 |

---
### Linear Regression, Gradient Boosting, Random Forest 3가지 모델 학습 및 예측 

## 모델 학습 및 평가 결과 (단일 변수(Price_snp사용))
### Linear Regression
![BTC_Linear_Regression](image/BTC_Linear%20Regression.png)
### Gradient Boosting
![BTC_Gradient_Boostiing](image/BTC_Gradient%20Boosting.png)
### Random Forest
![BTC_Random_Forest](image/BTC_Random%20Forest.png)

📈 결과 
-R² Score 매우 낮음 (음수)
|MODEL             | Mean Squared Error (MSE)  | R² Score |
|------------------|---------------------------|----------|
|Linear Regression | 47,398,364.45             | -23.29   |
|Gradient Boosting | 9,119,963.96              | -3.67    |
|Random Forest     | 6,616,306.57              | -2.39    |

📌 문제점:
- 단일 변수를 사용하여 복잡한 금융 데이터를 예측하려고 했기 때문에 예측력이 부족함.
- R² Score가 음수로, 모델이 패턴을 제대로 학습하지 못함.

### 2단계: 특성 확장 및 데이터 정규화
 - 설명: 변동률(Change %), 이동 평균(MA7, MA14), 과거 3일치 가격(Lag_1, Lag_2, Lag_3) 추가.
 - 데이터 정규화 (StandardScaler, MinMaxScaler) 적용하여 학습 안정성 증가
### Linear Regression
![BTC_Linear_Regression](image/2.BTC_Linear%20Regression_fix.png)
### Gradient Boosting
![BTC_Gradient_Boostiing](image/2.BTC_Gradient%20Boosting_fix.png)
### Random Forest
![BTC_Random_Forest](image/2.BTC_Random%20Forest_fix.png)

📈 결과
|MODEL             | Mean Squared Error (MSE)  | R² Score |
|------------------|---------------------------|----------|
|Linear Regression | 9,348,282.23              | 0.65     |
|Gradient Boosting | 113,742,884.52            | 0.75     |
|Random Forest     | 90,489,079.20             | 0.89     |

📌 개선 사항:

- 추가된 특성 덕분에 모델이 데이터 패턴을 학습하기 시작함.
- Gradient Boosting과 Random Forest의 성능이 일부 개선됨.

### 3단계: 하이퍼파라미터 튜닝
- 설명:
   - Gradient Boosting: n_estimators=300, learning_rate=0.01, max_depth=6, subsample=0.8 조정.
   - Random Forest: n_estimators=300, max_depth=15, min_samples_split=5
### Linear Regression BTC
![BTC_Linear_Regression](image/3.BTC_Linear%20Regression_fix.png)
### Gradient Boosting BTC
![BTC_Gradient_Boostiing](image/3.BTC_Gradient%20Boosting_fix.png)
### Random Forest BTC
![BTC_Random_Forest](image/3.BTC_Random%20Forest_fix.png)
### Linear Regression SNPE
![SNPE_Linear_Regression](image/3.SNPE_Linear%20Regression_fix.png)
### Gradient Boosting SNPE
![SNPE_Gradient_Boostiing](image/3.SNPE_Gradient%20Boosting_fix.png)
### Random Forest SNPE
![SNPE_Random_Forest](image/3.SNPE_Random%20Forest_fix.png)

📈 결과
|MODEL             | Mean Squared Error (MSE)  | R² Score |
|------------------|---------------------------|----------|
|Linear Regression | 934,828.23                | 1.00     |
|Gradient Boosting | 125,172,899.53            | 0.77     |
|Random Forest     | 92,435,873.56             | 0.83     |

📌 개선 사항:
- 하이퍼파라미터 튜닝을 통해 모델이 더욱 정교하게 학습.
- Gradient Boosting의 성능이 소폭 개선되었으나 여전히 낮은 편.

### 4단계 : 데이터 추가 및 시계열 데이터 강화
- 설명:
   - 과거 5일치 가격 (Lag_1 ~ Lag_5) 추가.
   - 변동성(Volatility), 모멘텀(Momentum) 등 추가 특성 생성.
   - 학습 데이터 개수 증가.
   - 기존 80:20 Train/Test Split 대신 전체 데이터를 학습.
### Linear Regression BTC
![BTC_Linear_Regression](image/4.BTC_Linear%20Regression_fix_FINAL.png)
### Gradient Boosting BTC
![BTC_Gradient_Boostiing](image/4.BTC_Gradient%20Boosting_fix_FINAL.png)
### Random Forest BTC
![BTC_Random_Forest](image/4.BTC_Random%20Forest_fix_FINAL.png)
### Linear Regression SNPE
![SNPE_Linear_Regression](image/4.SNPE_Linear%20Regression_fix_FINAL.png)
### Gradient Boosting SNPE
![SNPE_Gradient_Boostiing](image/4.SNPE_Gradient%20Boosting_fix_FINAL.png)
### Random Forest SNPE
![SNPE_Random_Forest](image/4.SNPE_Random%20Forest_fix_FINAL.png)

📈 결과(BTC)
|MODEL             | Mean Squared Error (MSE)  | R² Score |
|------------------|---------------------------|----------|
|Linear Regression | 832,564.86                | 1.00     |
|Gradient Boosting | 7,537,585.30              | 0.98     |
|Random Forest     | 274,189.22                | 1.00     |

📌 최종 개선 사항:
   - 데이터 확장과 적절한 하이퍼파라미터 튜닝을 통해 모델 성능 대폭 개선.
   - 특히, Lag Features, Volatility, Momentum 등의 시계열 기반 특성이 예측 성능 향상에 크게 기여함.

# Bitcoin Price Prediction - Model Comparison
### 모델 비교
| **모델**                    | **사용된 알고리즘**                                | **하이퍼파라미터 튜닝**     | **RMSE**              | **주요 특징**                                                                 |
|---------------------------|-------------------------------------------------|------------------------|----------------------|---------------------------------------------------------------------------|
| **1. XGBoost (기본)**       | XGBoost                                          | x                      | 𝑅𝑀𝑆𝐸:  580.0029918279143    | - 기본 XGBoost 회귀 모델<br> - 기본 학습률과 트리 개수                      |
| **2. XGBoost (GridSearch)** | XGBoost                                          | GridSearchCV            | 𝑅𝑀𝑆𝐸: 167.50863485151848    | - `GridSearchCV`로 최적 파라미터 튜닝<br> - 교차 검증을 통한 최적화          |
| **3. Random Forest**       | 랜덤 포레스트 (RandomForestRegressor)            | GridSearchCV            | 𝑅𝑀𝑆𝐸: 697.7323878267981     | - 앙상블 모델<br> - 여러 트리를 결합하여 예측<br> - 비선형 관계에 강함         |
| **4. LightGBM (기본)**       | Light Gradient Boosting Machine (LightGBM)      | x                      | R²: -0.3708, -2.2996           | - 독립 변수로 현재 시점 데이터만 학습<br> - 과거 데이터를 고려하지 않아 성능 저하 |
| **5. LightGBM (Lag=5)**      | Light Gradient Boosting Machine (LightGBM)      | Lag 적용 (5)           | R²:0.8576, 0.9134             | - 과거 5개 데이터 입력 변수 추가<br> - 패턴 학습 능력 향상                   |
| **6. LightGBM (Lag=10)**     | Light Gradient Boosting Machine (LightGBM)      | Lag 적용 (10)          | R²:0.8298, 0.9113            | - Lag 변수 10개 확장<br> - BTC 성능 저하, SNP 성능 개선                     |
| **7. LightGBM (GridSearch)** | Light Gradient Boosting Machine (LightGBM)      | GridSearchCV           | R²:0.8372, 0.9448              | - 최적의 하이퍼파라미터 탐색<br> - BTC 성능 불안정, SNP 성능 개선               |


