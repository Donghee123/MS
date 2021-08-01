import numpy as np
import math
import matplotlib.pylab as plt

def clipped_poisson(lam, max_k):
        """
        lamda, 차량 별 pmf값 저장
        """
        pmf = np.zeros(max_k + 1)
        for k in range(max_k):
            pmf[k] = math.exp(-lam) * lam**k / math.factorial(k)
            
        pmf[max_k] = 1 - np.sum(pmf)

        return pmf
   
 
def build_rent_return_pmf(lambda_request, lambda_return, max_cars):
        """
        요청, 반환 람다 값, 차량값 별로 1~20을 모두 저장 이후에 랜덤값 반환시 계속 확률 받환 예정
        """
        pmf = np.zeros((max_cars+1, max_cars+1, max_cars+1))
        
        for init_cars in range(max_cars + 1):
            new_rentals_pmf = clipped_poisson(lambda_request, init_cars)
            for new_rentals in range(init_cars + 1):
                max_returns = max_cars - init_cars + new_rentals
                returns_pmf = clipped_poisson(lambda_return, max_returns)
                for returns in range(max_returns + 1):
                    p = returns_pmf[returns] * new_rentals_pmf[new_rentals]
                    pmf[init_cars, new_rentals, returns] = p
                    
        return pmf

    
class JacksWorld(object):
    
    
    def __init__(self, lambda_return1, lambda_return2, lambda_request1, lambda_request2, max_cars):
        """
        첫번째 주차장, 두번째 주차장 푸아송 pmf 값 저장 및 최대 소유 차량 저장
        self.rent_return_pmf[0] = 첫번째 차량 푸아송 pmf 컨테이너 값 저장
        self.rent_return_pmf[1] = 두번째 차량 푸아송 pmf 컨테이너 값 저장 
        """
        self.request_return_pmf = []
        self.request_return_pmf.append(build_rent_return_pmf(lambda_request1,lambda_return1, max_cars))
        self.request_return_pmf.append(build_rent_return_pmf(lambda_request2, lambda_return2, max_cars))
        self.max_cars = max_cars
                   
    def get_transition_model(self, s, a):
        """
        Return 2-tuple:
        1. p(s'| s, a) as dictionary:
            keys = s'
            values = p(s' | s, a)
        2. E(r | s, a, s') as dictionary:
            keys = s'
            alues = E(r | s, a, s')
        """
        s = (s[0] - a, s[1] + a)         # move a cars from loc1 to loc2
        move_reward = -math.fabs(a) * 2  # ($2) per car moved
        t_prob, expected_r = ([{}, {}], [{}, {}])
        
        #2개 주차장에 대해서 iteration
        for loc in range(2):
            morning_cars = s[loc]
            request_return_pmf = self.request_return_pmf[loc]
            
            #rent 차량 iteration
            for rents in range(morning_cars + 1):
                max_returns = self.max_cars - morning_cars + rents
                
                #return 차량 iteration
                for returns in range(max_returns + 1):
                    p = request_return_pmf[morning_cars, rents, returns]
                    
                    if p < 1e-5:
                        continue
                    
                    s_prime = morning_cars - rents + returns
                    r = rents * 10
                    t_prob[loc][s_prime] = t_prob[loc].get(s_prime, 0) + p
                    expected_r[loc][s_prime] = expected_r[loc].get(s_prime, 0) + p * r
            
        # join probabilities and expectations from loc1 and loc2
        t_model, r_model = ({}, {})
        for s_prime1 in t_prob[0]:
            for s_prime2 in t_prob[1]:
                p1 = t_prob[0][s_prime1]  # p(s' | s, a) for loc1
                p2 = t_prob[1][s_prime2]  # p(s' | s, a) for loc2
                t_model[(s_prime1, s_prime2)] = p1 * p2
                # expectation of reward calculated using p(s', r | s, a)
                # need to normalize by p(s' | s, a)\n",
                norm_E1 = expected_r[0][s_prime1] / p1
                norm_E2 = expected_r[1][s_prime2] / p2
                r_model[(s_prime1, s_prime2)] = norm_E1 + norm_E2 + move_reward
                    
        return t_model, r_model
  
    
max_cars = 20
jacks = JacksWorld(lambda_return1=3, lambda_return2=2,
                     lambda_request1=3, lambda_request2=4, max_cars=max_cars)
  
    
V = np.zeros((max_cars+1, max_cars+1))
states = [(s0, s1) for s0 in range(max_cars+1) for s1 in range(max_cars+1)]
discountfactor = 0.9


# Value iteration
# 총 100번 반복중에 diff_average 값이 0.01 미만 일때 stop
for k in range(100):
    
    V_old = V.copy()
    V = np.zeros((max_cars+1, max_cars+1))
    
    #Value iteration
    for s in states:
        v_best = -1000
        max_a = min(5, s[0], max_cars-s[1])
        min_a = max(-5, -s[1], -(max_cars-s[0]))
        for a in range(min_a, max_a+1):
            t_model, r_model = jacks.get_transition_model(s, a)            
            v_new = 0
            for s_prime in t_model:
                p = t_model[s_prime]
                r = r_model[s_prime]
                # must use previous iteration's V(s): V_old(s)
                v_new += p * (discountfactor * V_old[s_prime] + r)
            v_best = max(v_best, v_new)
        V[s] = v_best
        #diff = max(diff, abs(V[s] - V_old[s]))
        
    diff_average = abs(np.average(abs(V) - abs(V_old)))  
    print("step : " + str(k) +", diff average : " + str(diff_average)) 
    
    if diff_average < 0.01: break
   
#Polcy impovement  
pi = np.zeros((max_cars+1, max_cars+1), dtype=np.int16)
for s in states:
    best_v = -1000
    max_a = min(5, s[0], max_cars-s[1])
    min_a = max(-5, -s[1], -(max_cars-s[0]))
    for a in range(min_a, max_a+1):
        t_model, r_model = jacks.get_transition_model(s, a)
        v = 0
        for s_prime in t_model:
            p = t_model[s_prime]
            r = r_model[s_prime]
            v += p * (discountfactor * V[s_prime] + r)
        if v > best_v:
            pi[s] = a
            best_v = v
                
print(pi)

plt.subplot(121)
CS = plt.contour(pi, levels=range(-6, 6))
plt.clabel(CS)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(21))
plt.yticks(range(21))
plt.grid('on')              
 
plt.subplot(122)
plt.pcolor(V)
plt.colorbar()
plt.axis('equal')

plt.show() 