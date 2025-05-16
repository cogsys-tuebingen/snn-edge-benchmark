import time
import test_dynap

nameClasspop = 'ClassPopModel'

ONCHIP = False

for i in range(0,10):
    print(i)
    time.sleep(1)
    
    test_dynap.trainTransformTest(nameClasspop + str(i+1), onChip = ONCHIP, runNum = i+1)
