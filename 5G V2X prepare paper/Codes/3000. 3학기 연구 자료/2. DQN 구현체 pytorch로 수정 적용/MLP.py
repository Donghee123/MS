import torch
import torch.nn as nn

"""
신경망을 사용할때 nn.Mudule을 상속받아서 할것임
추가 설명
1. super(MultiLayerPerceptron, self).__init__()
nn.Mudle을 슈퍼클래스로 상속 받았기 때문에 슈퍼클래스 부터 초기화 해야함

2. forward()
심층신경망의 연산 과정을 정의함
nn.Module은 forward() 함수로 __call__() overriding을 함. 보기에 좋으라고 오버라이딩을 하는건 아니고
nn.Mudle의 내부에서 pre_forward_hook, post_forward_hook등을 활용해서 automatic differentiation에 필요한 기능등을 보조 해줌.
따라서 아주 특별한 경우가 아니라면 forward()를 구현하고 심층신경망 연산을 진행할때도 forward() 함수를 사용하는 것을 권장 함
"""
class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32], #각 히든 레이어의 뉴런수
                 hidden_act: str = 'ReLU', #각 히든 레이어의 활성화 함수
                 out_act: str = 'Identity'): #아웃풋 레이어의 활성화 함수
        super(MultiLayerPerceptron, self).__init__() #baseclass 초기화

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)() #히든레이어의 활성화 함수 적용
        self.out_act = getattr(nn, out_act)() #출력레이어의 활성화 함수

        input_dims = [input_dim] + num_neurons #인풋레이어 + 히든레이어 리스트를 합함 Join 느낌?
        #ex [3] + [64,32] = [3, 64, 32]
        
        output_dims = num_neurons + [output_dim] #아웃풋레이어 + 히든레이어 리스트를 합함 Join 느낌?
        #ex [64, 32] + [2] = [64, 32, 2]
         
        #레이어에 뉴런 설정 루프
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:#마지막이면 출력 레이어 활성화 함수 적용
                self.layers.append(self.out_act)
            else:#마지막이 아니면 히든 레이어의 활성화 함수 적용
                self.layers.append(self.hidden_act)
            
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.DEVICE)


    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


if __name__ == '__main__':
    net = MultiLayerPerceptron(10, 1, [20, 12], 'ReLU', 'Identity')
    print(net)

    xs = torch.randn(size=(12, 10))
    ys = net(xs)
    print(ys)
