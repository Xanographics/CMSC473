import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import PReLU
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.autograd import Variable

'''
VERSION 2 NOTES: 
Architecture Changes
1. Changed Compression Loss to be L1/L2 instead of L2/L1
2. Removed Conv(3=3, k=5, d=2) from CNet expansion layer
3. Added Conv(3=3, k=3, d=1) to CNet expansion layer
4. Removed Conv(3=3, k=5, d=1) from RNet finish layer
5. Added Conv(3=3, k=3, d=1) to RNet finish layer

Hyper Param Changes
1. Increased elr = 0.003 to elr = 0.01
2. Increased dlr = 0.003 to dlr = 0.01
3. Increased epochs = 100 to epochs = 500
'''

'''
------------------------------------------------------------------------------------
Hyper Param Initialization
'''
batch_size = 8
elr, dlr = 0.01, 0.01
p = 10.0
epochs = 500

'''
------------------------------------------------------------------------------------
Encoder/Compression Architecture Definition
'''

class CompressionNet(nn.Module):
  def __init__(self):
    super(CompressionNet, self).__init__()

    self.expansion = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=3, 
                  kernel_size=3, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, padding='same'),
        nn.PReLU()
    )
    # block 1
    self.mod11 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, padding='same'),
        nn.PReLU()
    )
    self.mod12 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, padding='same'),
        nn.PReLU()
    )
    self.inter1 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=7, padding='same'),
        nn.PReLU()
    )
    # block 2
    self.mod21 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=3, dilation=2, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, dilation=2, padding='same'),
        nn.PReLU()
    )
    self.mod22 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=2, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=2, padding='same'),
        nn.PReLU()
    )
    self.inter2 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=2, padding='same'),
        nn.PReLU()
    )
    # block3
    self.mod31 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, dilation=4, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=3, dilation=4, padding='same'),
        nn.PReLU()
    )
    self.mod32 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=4, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=4, padding='same'),
        nn.PReLU()
    )
    self.inter3 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=4, padding='same'),
        nn.PReLU()
    )
    # block 4
    self.mod41 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, dilation=8, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, dilation=8, padding='same'),
        nn.PReLU()
    )
    self.mod42 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=8, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=8, padding='same'),
        nn.PReLU()
    )
    self.inter4 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=7, dilation=8, padding='same'),
        nn.PReLU()
    )
    # block 5
    self.mod51 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=3, dilation=16, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=3, dilation=16, padding='same'),
        nn.PReLU()
    )
    self.mod52 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=5, dilation=16, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=16, padding='same'),
        nn.PReLU()
    )
    self.inter5 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=16, padding='same'),
        nn.PReLU()
    )
    self.fin = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=5, padding='same'),
        nn.PReLU()
    )

  def forward(self,x):
    # expansion block
    x1 = self.expansion(x)
    # block 1
    out11 = self.mod11(x1)
    out12 = self.mod12(x1)
    cat1 = torch.cat([out11, out12, x], dim=1)
    # transition 1
    x2 = self.inter1(cat1)
    # block 2
    out21 = self.mod21(x2)
    out22 = self.mod22(x2)
    cat2 = torch.cat([out21, out22, x], dim=1)
    # transition 2
    x3 = self.inter2(cat2)
    # block 3
    out31 = self.mod31(x3)
    out32 = self.mod32(x3)
    cat3 = torch.cat([out31, out32, x], dim=1)
    # transition 3
    x4 = self.inter3(cat3)
    # block 4
    out41 = self.mod41(x4)
    out42 = self.mod42(x4)
    cat4 = torch.cat([out41, out42, x], dim=1)
    # transition 4
    x5 = self.inter4(cat4)
    # block 5
    out51 = self.mod51(x5)
    out52 = self.mod52(x5)
    cat5 = torch.cat([out51, out52, x], dim=1)
    # transition 5
    fin_in = self.inter5(cat5)
    # final
    out = self.fin(fin_in)
    return out
'''
------------------------------------------------------------------------------------
Decoder/Reconstruction Architecture Definition
'''

class ReconstructionNet(nn.Module):
  def __init__(self):
    super(ReconstructionNet, self).__init__()

    self.expansion = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=3, 
                  kernel_size=5, padding='same'),
        nn.PReLU()
    )
    # block 1
    self.mod11 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, dilation=16, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, dilation=16, padding='same'),
        nn.PReLU()
    )
    self.mod12 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=16, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5,dilation=16, padding='same'),
        nn.PReLU()
    )
    self.inter1 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=7,dilation=16, padding='same'),
        nn.PReLU()
    )
    # block 2
    self.mod21 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=3, dilation=8, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, dilation=8, padding='same'),
        nn.PReLU()
    )
    self.mod22 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=8, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=8, padding='same'),
        nn.PReLU()
    )
    self.inter2 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=8, padding='same'),
        nn.PReLU()
    )
    # block3
    self.mod31 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, dilation=4, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=3, dilation=4, padding='same'),
        nn.PReLU()
    )
    self.mod32 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=4, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=4, padding='same'),
        nn.PReLU()
    )
    self.inter3 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=4, padding='same'),
        nn.PReLU()
    )
    # block 4
    self.mod41 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=3, dilation=2, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=3, dilation=2, padding='same'),
        nn.PReLU()
    )
    self.mod42 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=5, dilation=2, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=2, padding='same'),
        nn.PReLU()
    )
    self.inter4 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=7, dilation=2, padding='same'),
        nn.PReLU()
    )
    # block 5
    self.mod51 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=3, dilation=1, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=3, dilation=1, padding='same'),
        nn.PReLU()
    )
    self.mod52 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=5, dilation=1, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1, 
                  kernel_size=5, dilation=1, padding='same'),
        nn.PReLU()
    )
    self.inter5 = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3, 
                  kernel_size=7, dilation=1, padding='same'),
        nn.PReLU()
    )
    # final block
    self.fin = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=3,
                  kernel_size=3, padding='same'),
        nn.PReLU(),
        nn.Conv1d(in_channels=3, out_channels=1,
                  kernel_size=3, padding='same'),
        nn.PReLU()
    )

  def forward(self,x):
    # expansion block
    x1 = self.expansion(x)
    # block 1
    out11 = self.mod11(x1)
    out12 = self.mod12(x1)
    cat1 = torch.cat([out11, out12, x], dim=1)
    # transition 1
    x2 = self.inter1(cat1)
    # block 2
    out21 = self.mod21(x2)
    out22 = self.mod22(x2)
    cat2 = torch.cat([out21, out22, x], dim=1)
    # transition 2
    x3 = self.inter2(cat2)
    # block 3
    out31 = self.mod31(x3)
    out32 = self.mod32(x3)
    cat3 = torch.cat([out31, out32, x], dim=1)
    # transition 3
    x4 = self.inter3(cat3)
    # block 4
    out41 = self.mod41(x4)
    out42 = self.mod42(x4)
    cat4 = torch.cat([out41, out42, x], dim=1)
    # transition 4
    x5 = self.inter4(cat4)
    # block 5
    out51 = self.mod51(x5)
    out52 = self.mod52(x5)
    cat5 = torch.cat([out51, out52, x], dim=1)
    # transition 5
    fin_in = self.inter5(cat5)
    # final
    out = self.fin(fin_in)
    return out

'''
------------------------------------------------------------------------------------
DATASET CLASS
MIGHT NEED TO LOOK AT BETTER WAYS TO RANDOMIZE/SPLIT DATA
'''
class AudioDataset(Dataset):
  def __init__(self, mode):
    file = np.load('./inputs.npy', allow_pickle=True)
    data = list(file)
    for i in range(len(data)):
      data[i] = torch.from_numpy(data[i])
    pad_data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    pad_data = pad_data.reshape(len(data), 1, -1)
    self.data = []
    if mode == 'train':
      self.data = pad_data[:256]
    elif mode == 'val':
      self.data = pad_data[256:384]
    else:
      raise ValueError('INVALID MODE: CHOOSE EITHER TRAIN/VAL')
  
  def __len__ (self):
    return self.data.shape[0]
  
  def __getitem__(self,idx):
    if idx >= 0 and idx < self.data.shape[0]:
      return self.data[idx]
    else:
      raise ValueError('INDEX OUT OF RANGE')

'''
Data Loader Initialization
'''
train_set = AudioDataset('train')
val_set = AudioDataset('val')

train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=batch_size)

'''
------------------------------------------------------------------------------------
Loss Computation Functions
'''
'''
Threshold/Cutoff function
'''
def cutoff(p, input_data):
  return_lst = []
  for inp in input_data:
    x = inp[0]
    cut_thresh = p / 100 * torch.mean(x)
    xc = torch.where(x >= cut_thresh, x, torch.zeros(x.shape[0]).cuda())
    return_lst.append(xc)
  out = torch.cat(return_lst)
  return out.reshape(input_data.shape[0], 1, -1)

'''
Compression Loss Function (L_C)

SHOULD WE TAKE THE DIRECT L1/L2 NORM OF X_C OR TAKE THE L1/L2 NORM LOSS BETWEEN X_C & X
'''
class CompressionLoss(nn.Module):
  def __init__(self):
    super(CompressionLoss, self).__init__()

    self.l2 = nn.MSELoss()
    self.l1 = nn.L1Loss()

  def forward(self, x_c, x):
    out = self.l1(x_c, x)/self.l2(x_c, x)
    return out

'''
loss initialization
'''
compr_loss = CompressionLoss()
acc_loss = nn.MSELoss()

'''
------------------------------------------------------------------------------------
Initializing Model & Training Iteration
'''
encoder = CompressionNet()
encoder = encoder.cuda()
decoder = ReconstructionNet()
decoder = decoder.cuda()

encoder_optim = optim.Adam(encoder.parameters(), lr=elr)
decoder_optim = optim.Adam(decoder.parameters(), lr=dlr)

train_comp_epochs_loss = []
train_diff_epochs_loss = []
val_comp_epochs_loss = []
val_diff_epochs_loss = []
for e in range(epochs):
  '''
  Training
  '''
  train_comp_loss = []
  train_diff_loss = []
  encoder.train()
  decoder.train()
  for idx, batch in enumerate(train_dataloader):
    x = batch.cuda()
    # running through compression net
    xc = encoder.forward(x)
    xp = cutoff(p, xc)
    x_hat = decoder.forward(xp)
    # calculating compression & accuracy losses
    comploss = compr_loss(xc, x)
    diffloss = acc_loss(x, x_hat)
    # backpropogating the losses
    diffloss.backward(retain_graph=True)
    comploss.backward()
    # ensuring optimizer kicks in
    decoder_optim.step()
    encoder_optim.step()
    # keeping track of the losses
    train_comp_loss.append(np.array(comploss.item()))
    train_diff_loss.append(np.array(diffloss.item()))
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

  '''
  Validation
  '''
  val_comp_loss = []
  val_diff_loss = []
  with torch.no_grad():
    encoder.eval()
    decoder.eval()
    for idx, batch in enumerate(val_dataloader):
      x = batch.cuda()
      # running through compression net
      xc = encoder.forward(x)
      xp = cutoff(p, xc)
      x_hat = decoder.forward(xp)
      # calculating compression & accuracy losses
      comploss = compr_loss(xc, x)
      diffloss = acc_loss(x, x_hat)
      # keeping track of the losses
      val_comp_loss.append(np.array(comploss.item()))
      val_diff_loss.append(np.array(diffloss.item()))

  # recording training losses across epochs
  train_comp_epochs_loss.append(np.array(train_comp_loss).mean())
  train_diff_epochs_loss.append(np.array(train_diff_loss).mean())

  # recording validation losses across epochs
  val_comp_epochs_loss.append(np.array(val_comp_loss).mean())
  val_diff_epochs_loss.append(np.array(val_diff_loss).mean())

  if e % 5 == 0:
    print("Epoch : {}, Train Compression Loss: {} , Train Accuracy Loss {}".format(e, train_comp_epochs_loss[e], train_diff_epochs_loss[e]))
    print('Epoch : {}, Val Compression Loss:   {} , Val Accuracy Loss   {}\n'.format(e, val_comp_epochs_loss[e], val_diff_epochs_loss[e]))

'''
------------------------------------------------------------------------------------
Saving model and training/validation losses
'''
# saving the validation arrays
np.save('Val_Comp_Loss.npy', np.array(val_comp_epochs_loss))
np.save('Val_Acc_Loss.npy', np.array(val_diff_epochs_loss))

# saving the training arrays
np.save('Train_Comp_Loss.npy', np.array(train_comp_epochs_loss))
np.save('Train_Acc_Loss.npy', np.array(train_diff_epochs_loss))

# saving models
torch.save(encoder, './compression.pth')
torch.save(decoder, './reconstruction.pth')