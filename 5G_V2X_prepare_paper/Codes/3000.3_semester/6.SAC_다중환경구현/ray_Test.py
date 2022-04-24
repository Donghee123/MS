import ray
import datetime
import time
	
print(ray.__version__)

def print_current_datetime():
    time.sleep(0.3)
    current_datetime = datetime.datetime.now()
    print(current_datetime)
    return current_datetime

print_current_datetime()

ray.init()