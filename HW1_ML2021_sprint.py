#import gdown
#https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb
print('hello')
#gdown --id '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF' --output covid.train.csv
#gdown --id '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O' --output covid.test.csv
# Download the training data
#gdown.download('https://drive.google.com/uc?id=19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF', 'covid.train.csv', quiet=False)

# Download the testing data
#gdown.download('https://drive.google.com/uc?id=1CE240jLm2npU-tdz81-oVKEF3T2yfT1O', 'covid.test.csv', quiet=False)

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import csv
import os
import matplotlib
#matplotlib.use('TkAgg',force=True)
#print("Switched to:",matplotlib.get_backend())
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069 #set a random see for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    '''Plot learning curve of your DNN (train&dev loss)'''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning Curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    '''Plot prediction of your DNN'''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x,y in dv_set:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Gounrd Truth v.s. prediction')
    plt.show()

class COVID19Dataset(Dataset):
    '''Dataset for loading and preprocessing the COVID19 dataset'''
    def __init__(self, 
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        #Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            #TODO: using 40 states & 2 tested_positive features (indices = 57 & 75)
            #feats = [57, 75]
            pass 
            
        if mode =='test':
            #testing data
            #data: 893 x 93 (40 states + day 1(18) + day2(18) + day3(17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # training data (trian/dev sets)
            #data: 2700 x 94 (40states + day 1(18) + day2(18) +day3(18))
            target = data[:, -1]
            data = data[:, feats]

            #splitting training data into train and dev sets
            if mode =='train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode =='dev':
                indices = [i for i in range(len(data)) if i %10 == 0]

            #convert data into pytorch tensor

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

            #Normalize features (you may remove this part to see what will happen)

        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim = True))\
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
            .format(mode, len(self.data), self.dim))
            
    def __getitem__(self, index):
        #returns one sample at a time
        if self.mode in ['train', 'dev']:
            # for training
            return self.data[index], self.target[index]
        else:
            #for testing no target
            return self.data[index]
            
    def __len__(self):
        #returns the size of the dataset
        return len(self.data)
    
#data loader
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only = False):
    '''Gnerates a dataset, then put into a dataloader.'''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only) #construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode =='train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader
    
# Create a simple neural network
#Neural is an nn.Module designed for regression. The DNN has 2 fully connected layers with ReLU activation. This module asl 
#includs a funciton cal_loss for calculating loss.
class NeuralNet(nn.Module):
    '''A simple fully connected deep neural network'''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        #TODO How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        '''given input of size(batchsize x input_dim), coompute output of the network'''
        return self.net(x).squeeze(1)
    
    def cal_loss(self, pred, target):
        '''Calculate loss'''
        return self.criterion(pred, target)
#Train/Dev/Test
def train(tr_set, dv_set, model, config, device):
    '''DNN training'''
    n_epochs = config['n_epochs']   #maximum number of epochs

    #setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    
    min_mse = 1000.
    loss_record ={'train': [], 'dev': []} #for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()         #set model to training mode
        for x, y in tr_set:   #iterate through the dataloader
            optimizer.zero_grad() #set gradient to zero
            x, y = x.to(device), y.to(device) #move data to device (cpu/cuda)
            pred = model(x)                    #forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  #compute loss
            mse_loss.backward()          #compute gradient (backpropagation)
            optimizer.step()                 #update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

            #after each epoch, test your model on the validation (devlopement) set

        dev_mse = dev(dv_set, model, device)
        if dev_mse<min_mse:
            #save mode if your model improved
            min_mse=dev_mse
            print('saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch +1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            #stop training if your model stops improving for a few epochs
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

#validation
def dev(dv_set, model, device):
    model.eval()   #set model to evaluation mode
    total_loss = 0
    for x, y in dv_set:                         #iterate through the dataloader
        x, y = x.to(device), y.to(device)       #move date to device
        with torch.no_grad():                #disable gradient calculation
            pred = model(x)                      #forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)    #compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  #accumulate loss

    total_loss = total_loss / len(dv_set.dataset)  #compute averaged loss
    return total_loss

#testing
def test(tt_set, model, device):
    model.eval()     #set modeo to evalution mode
    preds = []
    for x in tt_set:      #iterate through the data loader
        x = x.to(device)    #move data to device
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds
#setup hyper-parameters. config contains hyper-parameters for training and the patth to save your model

device = get_device()          #get the current available device
os.makedirs('models', exist_ok=True)      #the trained model witll be saved to ./models/
target_only = False

#TODO: How to tune these hyper-parameters to improve your model's performance?
#    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
 #    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
 #        'lr': 0.001,                 # learning rate of SGD
 #        'momentum': 0.9              # momentum for SGD
 # Adam, lr 0.001, weight_decay 0.0001
config = {
    'n_epochs': 3000,          #maximum number of epochs
    'batch_size': 270,         #mini-batch size for dataloader
    'optimizer': 'SGD',       #optimizer to be used
    'optim_hparas': {          #hyper-parameters for the optimizer
        'lr': 0.001,           #learning rate
        'momentum': 0.9  #weight decay (L2 regularization)
    },
    'early_stop': 200,         #early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  #your model will be saved here
}
#Load data and model
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only = target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)   #construct model and move to device

#start Training

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')    #load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)   #show prediction on the validation set

#start testing. The predictions of your model on teting set will be saved to 'pred.csv'

def save_pred(preds, file):
    '''Save predictions to specified file'''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        print('id,tested_positive', file=fp)
        for i, p in enumerate(preds):
            writer.writerow([i, p])
            #print('{},{}'.format(i, p), file=fp)
preds = test(tt_set, model, device)   # predict covid-19 cases with your model
save_pred(preds, 'pred.csv')          #save prediction to pred.csv
print('Finish running the code')




        



        
