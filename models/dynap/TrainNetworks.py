import time
import train_dynap


nameRegression = 'regressionModel'
nameClasspop = 'ClassPopModel'
nameClassMap = 'ClassMapModel'

data_folder = 'dataset/csv/'

for i in range(0,10):
    print(i)
    time.sleep(10)

    train_dynap.trainTransformTest(data_folder, nameClasspop + str(i+1), Inputtype = 'Lastframe', ordering = [0,1,5,6], onChip = False, onlyLoad = False,  NetworkType = 'ClassPop', SNNfromScratch = True)
