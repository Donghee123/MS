"""
타겟의 파라미터, 원본의 파라미터를 zip으로 묶은 후
타겟의 파라미터의 데이터 필드에 inplace 카피를 진행함.
이 기능을 소프트 타겟 업데이트라 부름
"""
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
