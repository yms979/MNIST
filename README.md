# MNIST

1. Plot train, test acc and ross
![Figure_1](https://github.com/yms979/MNIST/assets/45974948/17ae8926-a9be-4ae7-b439-06aa15bde042)

2. Compare the predictive performances of LeNet-5 and your custom MLP. Also, make sure that the accuracy of LeNet-5 (your implementation) is similar to the known accuracy. 
   이때 사용된 MLP과 LeNet-5의 파라미터 수는 다음과 같다.
-----------------------------------------------------------------------------------------------
  **LeNet-5 모델**:
  Conv1: (5 * 5 * 1 + 1) * 6 = 156
  Conv2: (5 * 5 * 6 + 1) * 16 = 2,416
  Conv3: (5 * 5 * 16 + 1) * 120 = 48,120
  FC1: (120 * 3 * 3 + 1) * 84 = 90,804
  FC2: (84 + 1) * 10 = 850
  총 파라미터 수: 156 + 2,416 + 48,120 + 90,804 + 850 = 142,346

  **CustomMLP 모델**:
  입력 크기: 784 (28 * 28)
  은닉층 1: (784 + 1) * 128 = 100,480
  은닉층 2: (128 + 1) * 64 = 8,256
  출력층: (64 + 1) * 10 = 650
  총 파라미터 수: 100,480 + 8,256 + 650 = 109,386
-----------------------------------------------------------------------------------------------
  Epoch [30/30]
  Training LeNet-5...
  LeNet-5 - Train Loss: 0.0910, Train Acc: 97.30%, Test Loss: 0.0661, Test Acc: 97.84%
  Training CustomMLP...
  CustomMLP - Train Loss: 0.4891, Train Acc: 85.41%, Test Loss: 0.2136, Test Acc: 93.65%
-----------------------------------------------------------------------------------------------

   위 30번째 에포크를 바탕으로 확인 한 결과 LeNet-5와 MLP 모두 좋은 결과를 나타내고 있지만, LeNet-5가 약 4%가량 높은 accuracy를 가지며, Loss또한 0.2 정도 낮은 지표를 보여준다.
   LeNet-5를 처음으로 제시한 논문("Gradient-based learning applied to document recognition.")에서는 30에포크에서 약 99%의 accuracy를 나타낸다고 한다.
   하이퍼 파라미터 튜닝, 모델 구조 변환 등 다양한 실험을 통하여 99%를 달성 할 수 있었겠지만, 개인의 역량 부족으로 97%까지밖에 이끌어 낼 수 있었다.
   수업에서 배운것과 같이 약 4만개의 파라미터 차이에서도 단순 MLP 구조보다 Convolution 레이어를 활용하여 지역적인 정보를 포착하는 방식이 이미지 분류 task에 더 우월한 성능을 가지는 것을 확인 할 수 있었다.
   
3. Employ at least more than two regularization techniques to improve LeNet-5 model.
  본 모델 및 학습과정에서 2가지의 정규화 기법이 들어가 있다.
    3-1. 드롭아웃(Dropout) 정규화:
        nn.Dropout 레이어를 추가하여 드롭아웃을 적용
        드롭아웃은 학습 중에 일부 뉴런을 무작위로 비활성화하여 과적합을 방지하고 모델의 일반화 성능을 향상시키는 역할
   
    3-2. L2 정규화(L2 Regularization) 또는 가중치 감쇠(Weight Decay):
        옵티마이저(optim.SGD)의 weight_decay 매개변수를 사용하여 L2 정규화를 적용
        L2 정규화는 가중치의 크기에 제약을 주어 과적합을 방지하고 모델의 일반화 성능을 향상시키는 역할


   LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
