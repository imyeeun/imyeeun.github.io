---
layout: post
title: 191019 Statistical Learning
tags: [DataScience,Data,StatisticalLearning,Regression,Classification,Parametric,Nonparametric,Supervised,Unsupervised,Prediction,Inference ]

comments: true
---

 # 01. Statistical Learning
Chapter 02 - Part 1. What is statistical Model?

---


 ### What is statistical learning?




 ---

### Notation

- Response(반응변수, 응답변수), Output Variable(출력변수), Dependent Variable(종속변수)
  - Sales(판매 수치)는 우리가 예측하고자 하는 target, Response.
  - 우리는 이를 Y라고 언급할 것.

- Predictors(설명변수, 예측변수), Input Variable(입력변수), Independent Variable(독립변수)
  - TV는 feature(특징, 변수) or input or predictor이고  X~1~라고 한다.
  - Radio, X~2~
  - Newspaper, X~3~

- 양적 반응변수 Y와 p개의 다른 변수 X~1~, X~2~,...,X~p~가 관찰되었다고 가정하자.
- Y와 X = ( X~1~, X~2~,...,X~p~) 사이에 어떤 상관관계가 있다고 가정하자.
- 우리는 이 관계를 다음과 같은 모델로 표현할 수 있다.
$$ Y = f(x) + \epsilon   $$

- 여기서 f는 X~1~, X~2~,...,X~p~에 대한 알려지지 않은 어떤 함수이고 epsilon은 X와 독립적이고 평균이 0인 랜덤오차항이다.  


### Statistical Learning
- 일반적으로 입력 변수를 출력 변수에 연결하는 함수 f는 알려져 있지 않다.
- 이러한 경우, 함수 f는 관찰된 점들을 기반으로 추정해야 한다.
- **통계학습**은 f를 추정하는 일련의 기법들을 말한다.

### Why do we estimate f?
- 이 강의에서 통계학습은 f를 추정하는 방법에 대한 것이다.
- 통계학습이라는 용어는 데이터를 사용하여 f를 학습한다는 것을 의미합니다.
- 우리는 왜 f를 추정하는데 관심이 있을까?
- f를 추정하고자 하는 두 가지 주요한 이유
  1) Prediction 예측 (과거의 데이터를 기반으로 미래에 대한 설명, 계획, 예정)
  2) Inference 추론 (논리적으로 결과 도출)

---
### Prediction

- f에 대한 좋은 추정치를 생성 할 수 있다면 (ε의 분산이 너무 크지 않은 경우) 새로운 **X** 값을 기반으로 응답 Y에 대한 정확한 예측을 할 수 있습니다.
$$\hat{Y} = \hat{f}(x)$$
 - F는 보통 블랙박스로 취급되는데 그것은 f가 Y에 대해 정확한 예측을 제공한다면 그것의 정확한 형태에 대해서는 통상 신경쓰지 않기 때문이다.

 ### Prediction Errors
 - Y에 대한 예측인 ŷ의 정확도는 축소가능 오차(Reducible error)와 축소불가능 오차(irreducible error)라고 불리는 두 가지에 달려 있다.

 - **Reducible error** :: f̂과 ƒ의 차이
    - f̂는 ƒ를 완벽하게 추정하지 못하며, 이러한 부정확성으로 인해 오차가 발생될 것이다. 이러한 오차는 축소가능하다. 왜냐하면 가장 적절한 통계학습기법을 사용하여 ƒ를 추정함으로써 f̂의 정확성을 개선할 수 있기 때문이다.
  - **Irreducible error** :: f(x)와 Y의 차이
    - E는 정의상 X를 사용하여 예측할 수 없습니다.
    - f를 아무리 잘 추정해도 e에 의한 오차를 줄일 수는 없다

$$ E(Y-\hat{Y})^2 =  E[f(X)+\epsilon - \hat{f}(X)]^2$$$$= [f(X)-\hat{f}(X)]^2 + Var(\epsilon)$$

$$축소가능한 오차 => [f(X)-\hat{f}(X)]^2$$ $$축소불가능한 오차 =>Var(\epsilon)$$
- 우리는 축소가능 오차를 최소로 하며 f를 추정하는 기법들에 대해 중점적으로 다룬다.

---
### Inference
- 또는 Y와 X들의 관계 유형에 관심이 있을 수도 있다.
- 예를들어,
  - 어떤 설명변수들이 반응변수와 관련되어 있는가?
  - 관계가 양의 상관관계를 가지는가? 음의 상관관계를 가지는가?
  - Y의 각 설명변수의 상관관계는 단순한 선형 관계인가? 아니면 더 복잡한 것인가?
- 이제, f̂은 블랙박스로 취급될 수 없다. 왜냐하면 그것의 정확한 형태를 알아야 할 필요가 있기 때문이다.

---

##### Example : Direct Mailing Prediction
- 우리는 400 가지가 넘는 다른 특성을 기록한 90,000 명의 관찰 결과에 근거하여 개인이 얼마나 많은 돈을 기부 할 것인지에 관심이 있다.
- 각 개인의 특성에 너무 신경 않아도 된다.

##### Example : Housing Inference
- 14개의 변수들을 기반으로 중간 주택 가격을 예측하고자 한다.
- 가장 큰 영향을 미치는 요소와 그 영향이 얼마나 큰지를 이해하고자 한다.
- 예를 들어 강이 보이는 전망이 주택 가격에 얼마나 많은 영향을 미치는지.

---
### How do we estimate f?
- 언제나 n개의 다른 데이터 포인트를 관측한다고 가정할 것이다.
- 이러한 관측치들은 훈련 데이터(Training Data)라고 한다.
{(X<sub>1</sub>,Y<sub>1</sub>),(X<sub>2</sub>,Y<sub>2</sub>),...,(X<sub>n</sub>,Y<sub>n</sub>)}
  - X<sub>ij</sub>는 관측 i에 대한 j번째 설명변수 또는 입력의 값을 의미하고
  - X<sub>i</sub> = (X<sub>i1</sub>,X<sub>i2</sub>,...,X<sub>ip</sub>)<sup>T</sub>이다.
- 우리는 f를 추정하기위해 훈련 데이터와 통계학습방법을 사용할 것이다.
- 통계학습방법들은 모수적(parametric) 또는 비모수적(non-parametric)으로 특징지을 수 있다.


### Parametric Method
- f를 추정하는 문제가 파라미터 집합을 추정하는 문제로 된다.
- 모수적 방법은 2단계로 된 모델 기반의 기법이다.

**Step 1:** 먼저, f의 함수 형태 또는 모양에 대해 가정한다. 예를 들어, 아주 단순하게 f는 X에 대해 선형적이라고 가정한다.
$$f(X) = \beta_{0}+\beta_{1}X_{i1}+\beta_{2}X_{i2}+...+\beta_{p}X_{ip} ~$$

하지만, 이 과정에서는 f에 대한 훨씬 더 복잡하고 유연한 모델을 검토한다. 어떤 의미에서는 모델이 더 유연할수록 더 현실적이다.

**step 2:** 모델이 선택된 후 학습 데이터를 사용하여, 모델을 적합(fit)하거나 훈련한다. 예를 들어 추정값 f 또는 $$\beta_{0},\beta_{1},\beta_{2},...,\beta_{p}.$$와 같은 파라미터들을 추정한다.
- 선형 모형에서 파라미터를 추정하는 가장 일반적인 방법은 (보통의)최소제곱이다.
- 하지만, 최소제곱은 선형모델을 적합하는 많은 가능한 방법들 중의 하나이다.
- 우리는 이 강의에서 종종 더 우월한 접근법을 볼 것이다.

$$ income \approx \beta_{0}+\beta_{1} * education + \beta_{2} * seniority $$

반응변수와 2개의 설명변수 사이에 선형 상관관계가 있다고 가정하므로, 전체 적합 문제는 $$\beta_{0},\beta_{1},\beta_{2}$$를 추정하는 문제로 바뀌고, 이것은 최소제곱 선형회귀를 사용하여 추정한다. 주어진 선형적합은 실제와 잘 맞지 않다는 것을 볼 수 있는데, 실제 f는 선형적합으로는 포착되지 않는 곡선 부분이 존재한다. 하지만, 선형적합은 **덜 긍정적인 상관관계를 합리적으로 포착**하는것처럼 보인다.

---
### Non parametric Method
- 비모수적 방법은 f의 함수 형태에 대해 명시적인 가정을 하지 않는다.
- **장점** :
  - f의 형태에 대한 가정이 없다.
  - 더 넓은 범위의 f 형태에 정확하게 적합될 가능성이 있다.
- **단점**:
  - f에 대한 정확한 추정을 얻기 위해서는 아주 많은 수의 (모수적 기법에서 보통 필요로 하는 것보다 훨씬 더 많은 수의) 관측치가 필요하다.

- Thin-plate-spline : 네모나 세모를 붙여서 최대한 곡선을 만드는 것.
- 관측된 데이터에 가능한 한 가까운 f에 대한 추정치를 생성한다.
- 비모수적 적합은 실제 f 값을 놀라울만큼 정확하게 추정한다.

---
### Tradeoff Between Prediction Accuracy and Model Interpretability
- Linear model vs thin plate spline을 고려해보자.
-  도대체 왜 현실에서는  매우 유연한 기법 대신에 더 제한적인 방법을 선택하여 사용하는가?

**Reason 1 : Interpretability**
- 선형 회귀와 같은 간단한 방법은 해석하기가 훨씬 쉬운 모델을 생성한다. (추론에 능하다.)
- 예를들어, 선형 모델에서 β<sub>j</sub>는 다른 모든 변수를 일정하게 유지하는 X<sub>j</sub>의 한 단위 증가에 대한 Y의 평균 증가이다.
#### Interpretability vs Flexibility
![INTERvsFLEX](https://masterr.org/images/xkcd-interpretability-vs-flexibility.png)
그림 1.1 : 통계학습방법에 따른 유연성과 해석력 사이의 관계. 일반적으로 유연성이 증가함에 따라 해석력은 감소한다.

**Reason 2 : Accuracy(Avoid Overfitting)**
- 해석이 문제가 되지 않더라도,
- 때때로, 덜 유연한 모델이 더 정확한 예측 결과를 가능하게 한다.
- 아주 유연한 방법들의 잠재적인 과적합과 관련이 있다..


![THINPLATESPLINE](http://www.sr-sv.com/wp-content/uploads/2015/09/STAT00.png)
그림 1.2 : Income 자료에 대한 thin-plate-spline. 이 적합은 훈련 데이터에 대해 오차가 없다.
- 너무 유연한 모델은 과적합의 가능성이 있다.
---
### Supervised vs Unsupervised
- 대부분의 통계학습 문제들은 두 가지 부류 중 어느 하나, 즉 지도학습 또는 비지도학습에 속한다.

- **Supervised Learning:**
  - 설명변수를 측정한 각 관측치 x<sub>i</sub>(i=1,...,n)에 대하여 연관된 반응변수 y<sub>i</sub>가 있다.
  - 선형 회귀 클래스를 다루는 상황이다.
  - 이과정에서는 대부분 지도 학습을 다룰 것이다.

- **Unsupervised Learning:**
  - 이에 반해, 비지도학습은  x<sub>i</sub>(i=1,...,n)에 대하여 연관된 반응변수 y<sub>i</sub>가 없는 좀 더 어려운 상황을 설명한다.
  - 우리는 x<sub>i</sub>들을 사용하여 Y가 무엇인지 추측하고 거기에서 모델을 만들어야 한다.
  - 일반적인 예로 고객의 특성에 따라 잠재 고객을 그룹으로 나누려고 하는 시장 세분화(market segmentation)를 들 수 있다.
  - 일반적인 접근방식으로 클러스터링이 있다.
  - 이 과정이 끝나면 비지도 학습을 고려해볼 것이다.



---
### Regression vs Classification
- 변수의 종류
  - 양적 변수
    - 수치의 값을 취하는 것.
    - 사람의 나이, 키 또는 수입, 집 값, 그리고 주식 가격
  - 질적 변수
    - K개의 다른 클래스(classes) 또는 카테고리(category) 중의 하나를 값으로 가짐.
    - 사람의 성별(남성 또는 여성), 구입한 제품의 브랜드(A, B, C), 어떤 사람의 채무 지불 여부(연체 또는 연체 아님), 또는 암 진단(급성 골수성 백혈병, 급성 림프구성 백혈병, 또는 백혈병 아님)

- Regression
  - 양적 반응변수를 가지는 문제를 회귀 문제라고 한다.
- Classification
  - 질적 반응변수가 관련된 문제는 분류 문제라고 한다.

*하지만 이 구분이 항상 명확한 것은 아니다.















<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "center"
});
</script>
