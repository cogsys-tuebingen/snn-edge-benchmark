import time
import train_akida

nameClasspop = 'ClassPopModel_akida_'

for i in range(0,10):
    train_akida.trainModel(nameClasspop + str(i+1))
