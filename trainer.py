from metrics import calculate_metrics
import torch
from torch import nn
import sklearn
import sklearn.metrics
import numpy as np
from tqdm import tqdm
import wandb
import datetime
import pickle
from PIL import Image
import PIL
from collections import defaultdict
import mxnet as mx
from mxnet import ndarray as nd

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class MetricMonitor:
    def __init__(self, float_precision=5):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )



def train(net,trainloader,validationloader,n_epochs=10,lr=0.1):
    MSE = torch.nn.MSELoss()
    data_set = load_bin("faces_emore/lfw.bin", (112,112))
    wandb.init(project='', entity='')
    wandb.config.lr1 = 0.005  
    wandb.config.lr2 = 0.1  
    net.to("cuda:0")
    net.train()
    criterion = nn.CrossEntropyLoss()
    param2 = list(net.module.model.parameters()) + list(net.module.fc.parameters()) + list(net.module.fc2.parameters())
    optimizer2 = torch.optim.SGD(param2, lr=wandb.config.lr2,weight_decay=5e-4,momentum=0.9)
    iteration = 0

    best_score = 100

    rate_decrease=1
    patience = 1

    for epoch in range(0,n_epochs):
        
        metric_monitor = MetricMonitor()
        stream = tqdm(trainloader)
        for _, sample in enumerate(stream, 0):
            net.train()
            inputs = sample['image']
            inputs_masked = sample['image_masked']
            labels = sample['identity']
            labels2 = sample['mask']
            inputs,inputs_masked, labels,labels2 = inputs.to("cuda:0"),inputs_masked.to("cuda:0"), labels.to("cuda:0"),labels2.to("cuda:0")

            
            optimizer2.zero_grad()
            outputs,e1,e2,mask = net(inputs,label=labels)
            loss = (criterion(outputs, labels)) +  0.1 * criterion(mask*0,labels2) 
            outputs,e1_,e2,mask = net(inputs_masked,label=labels)
            loss += (criterion(outputs, labels)) +  0.1 * criterion(mask,labels2) 
            loss /= 2
            loss += MSE(e1,e1_)/3
            loss.backward()
            optimizer2.step()
            

             
            metric_monitor.update("Loss P", loss.item())
            wandb.log({"Loss P":loss.item()})
            
            iteration +=1
            stream.set_description("Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
        fmr100 = validate(net,data_set,str(epoch))
        
        if fmr100 < best_score:
                best_score = fmr100
                torch.save(net.module.state_dict(), "uai_batch" + str(epoch+1) +".mdl")
                print("SAVED THE MODEL")
                patience = 1
        else:
            if patience == 0:
                patience = 1
                rate_decrease /= 10
                optimizer2 = torch.optim.SGD(param2, lr=wandb.config.lr2 * rate_decrease,weight_decay=5e-4,momentum=0.9)
                print("New Learning Rate")
                print(wandb.config.lr2 * rate_decrease)
            else: patience -= 1
    print('Finished Training')

    
def validate(net,data_set,epoch):
    net.eval()
    with torch.no_grad():
        metrics = test(data_set, net, 128,epoch)
        print("FMR100 = " + str(metrics[1]*100))
        wandb.log({"FMR100":metrics[1]*100})
        print("AUC = " + str(metrics[5]))
        wandb.log({"AUC":metrics[5]})
        wandb.log({"GMean":metrics[3]})
        wandb.log({"IMean":metrics[4]})
        return metrics[1]

masked_labels = []
@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    
    
    #print(len(issame_list))
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        #pdb.set_trace()
        #im = Image.fromarray(img.asnumpy())
        #im.save("new_dataset/"+str(idx)+".jpg")
        if idx % 2 == 0:
            try:
                im = Image.open("new_dataset_masked2/"+str(idx)+".jpg")
                R, G, B = im.split()
                im = PIL.Image.merge("RGB", (B, G, R))
                img = mx.nd.array(np.array(im))
                masked_labels.append(1)
            except:
                im = Image.open("new_dataset/"+str(idx)+".jpg")
                R, G, B = im.split()
                im = PIL.Image.merge("RGB", (B, G, R))
                img = mx.nd.array(np.array(im))
                masked_labels.append(0)
        else:
            #_bin = bins[idx]
            #img = mx.image.imdecode(_bin)
            im = Image.open("new_dataset/"+str(idx)+".jpg")
            R, G, B = im.split()
            im = PIL.Image.merge("RGB", (B, G, R))
            img = mx.nd.array(np.array(im))
            masked_labels.append(0)
        
            #if img.shape[1] != image_size[0]:
            #    img = mx.image.resize_short(img, image_size[0])

        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

@torch.no_grad()
def test(data_set, backbone, batch_size,epoch):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    masked = []
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        print(i)
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            img = img.to(device)
            _,net_out,_,y2 = backbone(img,inference = True)
            masked.append((i,y2.detach().cpu().numpy()))
            del img

            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
        if i % 1 == 0:
            print('loading bin', i)
            print(time_consumed)

    masked2 = []
    i = 0
    with open("mask_prediction.txt","w") as w:
        for mask in masked:
            label = mask[0]
            for mask2 in mask[1]:
                mask2=mask2.item()
                
                w.write(str(label) + "," + str(masked_labels[i]) + "," + str(mask2)  + "\n")
                i+=1


    _xnorm = 0.0
    _xnorm_cnt = 0
    print("Normalizing")
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt


    
    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    positives = []
    negatives = []

    print(len(issame_list))

    for embedding1, embedding2,label in zip(embeddings1,embeddings2,issame_list):
        dist = 1- torch.cdist(torch.from_numpy(embedding1).view(1, -1), torch.from_numpy(embedding2).view(1, -1))/2
        if label == 1:
            positives.append(dist)
        else:
            negatives.append(dist)
    return calculate_metrics(positives,negatives,epoch)

