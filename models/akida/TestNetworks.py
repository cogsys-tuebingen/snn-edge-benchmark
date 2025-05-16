from akida import devices
import test_akida
import akida

ONCHIP = True

for i in range(0,10):

    model = akida.Model("ClassPopModel_akida_"+str(i+1)+".fbz")

    if ONCHIP:
        device = devices()[0]
        model.map(device, hw_only =True)
    model.summary()

    test_akida.testModel(model, runNum = i+1)
