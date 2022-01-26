import loader
import model
import trainer
import torch
import torch.nn as nn

identities = 85742
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

validation_split = .01
batch_size= 480
workers = 8

train_loader, validation_loader, classes = loader.get_train_loader(batch_size,workers,validation_split)


net = model.FocusFace(identities=identities)

net = nn.DataParallel(net, device_ids=[0,1,2,3]).to(device)

trainer.train(net,trainloader=train_loader,validationloader=validation_loader,n_epochs=500,lr=0.01)



