from random import sample

"""
s,a,r,ns 를 저장함
"""
class ReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        
        #버퍼의 최대 사이즈 지정
        self.buffer = [None] * max_size
        
        #최대 사이즈 저장
        self.max_size = max_size
        
        #현재 인댁스
        self.index = 0
        
        #현재 사이
        self.size = 0

    def push(self, obj):
        #버퍼에 (s,a,r,ns) self.index 번째에 저장
        self.buffer[self.index] = obj
        
        #기존 사이즈에 1을 더하지만 최대 사이즈 보다 크지 않게 만듬
        self.size = min(self.size + 1, self.max_size)
        
        #지정 인덱스 = 지정인덱스 + 1 % 최대 사이즈 
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        
        #batch_size만큼 랜덤 샘플링
        indices = sample(range(self.size), batch_size)
        
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size