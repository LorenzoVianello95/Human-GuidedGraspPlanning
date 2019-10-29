import matplotlib.pyplot as plt
import math


train_loss= "loss_history.txt"

f = open(train_loss, "r")

loss_TRAIN= []
for l in f:
    if len(loss_TRAIN)>0:
        loss_TRAIN.append(0.5*math.log(float(l))+0.5*loss_TRAIN[-1])
    else:
        loss_TRAIN.append(math.log(float(l)))


loss_TRAIN= loss_TRAIN[20:]  
print(len(loss_TRAIN))
plt.plot(loss_TRAIN, label='train')

train_loss= "val_loss_history.txt"

f = open(train_loss, "r")

loss_TRAIN= []
for l in f:
    if len(loss_TRAIN)>0:
        loss_TRAIN.append(0.5*math.log(float(l))+0.5*loss_TRAIN[-1])
    else:
        loss_TRAIN.append(math.log(float(l)))

loss_TRAIN= loss_TRAIN[20:]  
print(len(loss_TRAIN))
plt.plot(loss_TRAIN, label='validation')

plt.gca().legend(('Train Loss','Validation Loss'))
plt.show()