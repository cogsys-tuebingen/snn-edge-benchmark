from akida import devices
import test_akida
import akida
ONCHIP = False

model = akida.Model('akida_save.fbz')

if ONCHIP:
    device = devices()[0]
    model.map(device, hw_only=True)

model.summary()

test_akida.testModel(model, runNum = 1)
