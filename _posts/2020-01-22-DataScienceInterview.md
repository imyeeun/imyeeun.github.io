---
layout: post
title: 200122 DataScienceInterview
tags: [DataScience,Data,StatisticalLearning,Outlier,feature,DataScienceInterview]

comments: true
---

 # 01. DataScienceInterview
데이터 사이언스 분야의 인터뷰(https://zzsza.github.io/data/2018/02/17/datascience-interivew-questions/) 에 대한 답변을 매주 2개씩 정리하고자 한다. 금주의 질문은 다음과 같다.


 -  아웃라이어의 판단하는 기준은 무엇인가요?


 - 좋은 feature란 무엇인가요. 이 feature의 성능을 판단하기 위한 방법에는 어떤 것이 있나요?



---
### 아웃라이어의 판단하는 기준은 무엇인가요?
---
**아웃라이어란?**
아웃라이어는 대부분 관측치보다 극단적으로 크거나 작은 값입니다. 이는 드물거나 뚜렷하지 않거나 어떤식으로든 적합하지 않습니다. 아웃라이어는 다음과 같은 원인들을 가지고 있습니다.

- 측정 혹은 입력 오류
- 데이터 손상
- 진정한 아웃라이어 (ex; 농구의 마이클 조던)

정규화뿐 아니라 예측 모델을 구축할 때도 아웃라이어가 악영향을 끼칠 때가 많아 전처리 단계에서 제거해야 합니다. 하지만 데이터를 제거한다는 것 자체는 극단적인 값이 발생하는 상황의 데이터를 고려하지 않는다는 것을 의미하기에 특수한 상황을 분석하는 경우에는 아웃라이어를 살려두기도 합니다.

**아웃라이어를 어떻게 판단하나요?**
가장 일반적이고 자주 사용되는 방법은 정규분포를 전제로 한 검출방법입니다. 표준편차의 일정 배수 이상 떨어진 값을 평균값에서 제거하는 방법으로 일정 배수를 작게 하면 많은 값을 아웃라이어로 검출하고, 크게하면 좀 더 극단적인 값만을 아웃라이어로 검출합니다. 일반적으로 표준편차의 3배 이상으로 큰 값을 설정합니다. 정규분포에 따른 값은 평균값에서 표준편차의 3배 이내 범위에 99.73%의 값이 있으므로 발생할 확률이 0.27%이하인 값을 아웃라이어로 보고 해당 값들이 검출된 행을 제거하여 아웃라이어를 제거할 수 있습니다.

데이터셋은 R MASS패키지에 내장되어 있는 cats를 사용합니다. 고양이 심장 무게의 평균값에서 표준편차의 3배 이내의 값만 있도록 수정하는 코드입니다.  

![removeOutlierCode](./images/removeOutlierCode.PNG)

다음은 아웃라이어 값입니다. 극단적으로 큰 값임을 확인할 수 있습니다.

![outlierCat](./images/outlierCat.PNG)



---
### 좋은 feature란 무엇인가요. 이 feature의 성능을 판단하기 위한 방법에는 어떤 것이 있나요?
---

머신러닝의 시작은 좋은 feature를 찾는 일부터 시작합니다. 이에 쓰이는 기법이 feature selection과 feature extraction입니다.

**feature selection**은 모델에 사용할 좋은 독립변수들의 부분집합을 선택하는 것입니다. 모든 독립변수가 종속변수에 영향을 끼치는것은 아니기때문입니다.(키는 시력과 크게 관련이 없을것) 이를 feature selection이라 하며 대표적인 방법으론 fisher score 등이 있습니다. 해석하는 측면에서는 feature extraction보다 좋지만 늘 최선의 결과를 얻을 수 있는건 아닙니다.   

**feature extraction**은 기존 독립변수들의 조합을 통해 새로운 변수를 만드는 것입니다. 수식을 통해 만들 수 있으며 데이터를 더욱 더 잘 설명할 수 있는 방향으로(==likelihood를 증가시키는 방향으로) 학습해서 얻을수도 있습니다. PCA가 가장 대표적인데 이는 데이터에서 가장 변화량이 큰 방향을 주축으로 삼고 이와 관련된 변수들을 뽑아낼 수 있습니다. feature extraction은 새로운 feature를 만든다는 점에서 새로운 정보를 얻을 수 있지만 그 feature를 해석하는 측면에서 활용도가 떨어집니다.


**참고**

[1] https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/

[2] 모토하시 도모미쓰, 데이터 전처리 대전, 2019.

[3] https://www.facebook.com/PapersTerryRead/posts/1723695611208631/
