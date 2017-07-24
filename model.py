import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy

cuda = True

class rn_layer_class(nn.Module):
    """docstring for rn_layer"""
    def __init__(self, mb, num, lenth, aux_lenth=0, size=256, output_size=256, name='no'):
        super(rn_layer_class, self).__init__()

        # settings
        self.mb = mb
        self.num = num
        self.lenth = lenth
        self.aux_lenth = aux_lenth
        self.size = size
        self.output_size = output_size
        self.name = name

        # NNs
        self.g_fc1 = nn.Linear((self.lenth+1)*2+self.aux_lenth, self.size).cuda()

        self.g_fc2 = nn.Linear(self.size, self.size).cuda()
        self.g_fc3 = nn.Linear(self.size, self.size).cuda()
        self.g_fc4 = nn.Linear(self.size, self.output_size).cuda()

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(self.mb, num, 1).cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((self.mb, self.num, 1))
        for i in range(self.num):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

    def cvt_coord(self, i):
        return [i]

    def forward(self, in_, aux=None):

        # add coordinates
        in_ = torch.cat([in_, self.coord_tensor],2)

        # cast all pairs against each other i
        x_i = torch.unsqueeze(in_,1) # (64x1x25x25)
        x_i = x_i.repeat(1,self.num,1,1) # (64x25x25x25)

        # cast all pairs against each other j
        x_j = torch.unsqueeze(in_,2)
        if (self.aux_lenth is 0) or (aux is None):
            x_j = x_j
        else:
            # add aux everywhere
            aux = torch.unsqueeze(aux, 1)
            aux = aux.repeat(1,self.num,1)
            aux = torch.unsqueeze(aux, 2)
            x_j = torch.cat([x_j,aux],3)
        x_j = x_j.repeat(1,1,self.num,1)

        # concatenate all together
        x_full = torch.cat([x_i,x_j],3)

        # reshape for passing through network
        x_ = x_full.view(x_full.size()[0]*x_full.size()[1]*x_full.size()[2],x_full.size()[3])

        x_ = self.g_fc1(x_)        
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(self.mb,x_.size()[0]/self.mb,self.output_size)
        x_g = x_g.sum(1).squeeze()

        return x_g

class conv_layer(nn.Module):

    def __init__(self):
        super(conv_layer, self).__init__()

        # NNs
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1).cuda()
        self.batchNorm1 = nn.BatchNorm2d(24).cuda()
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1).cuda()
        self.batchNorm2 = nn.BatchNorm2d(24).cuda()
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1).cuda()
        self.batchNorm3 = nn.BatchNorm2d(24).cuda()
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1).cuda()
        self.batchNorm4 = nn.BatchNorm2d(24).cuda()

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        return x

class f_layer(nn.Module):

    def __init__(self):
        super(f_layer, self).__init__()

        self.f_fc1 = nn.Linear(256, 256).cuda()
        self.f_fc2 = nn.Linear(256, 256).cuda()
        self.f_fc3 = nn.Linear(256, 10).cuda()

    def forward(self, x):
        
        x = self.f_fc1(x)
        x = F.relu(x)
        x = self.f_fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.f_fc3(x)
        x = F.log_softmax(x)

        return x

class rn_model(nn.Module):

    def __init__(self, mb):
        super(rn_model, self).__init__()
        
        self.mb = mb

        self.conv_layer = conv_layer()

        self.rn_config = [2,2,1]

        self.rn_layer_dic = [[0 for col in range(max(self.rn_config))] for row in range(len(self.rn_config))]

        for rn_depth in range(0,len(self.rn_config)):
            for rn_id in range(self.rn_config[rn_depth]):
                if rn_depth is 0:
                    self.temp =  rn_layer_class(mb=self.mb,
                                                num=25,
                                                lenth=24,
                                                aux_lenth=11,
                                                size=256,
                                                output_size=256,
                                                name=str(rn_depth)+'_'+str(rn_id))

                    self.rn_layer_dic[rn_depth][rn_id] = copy.deepcopy(self.temp)
                else:
                    self.temp =  rn_layer_class(mb=self.mb,
                                                num=self.rn_config[rn_depth-1],
                                                lenth=256,
                                                aux_lenth=0,
                                                size=256,
                                                output_size=256,
                                                name=str(rn_depth)+'_'+str(rn_id))
                    self.rn_layer_dic[rn_depth][rn_id] = copy.deepcopy(self.temp)

        self.f_layer = f_layer()

    def forward(self, img, qst):

        conved = self.conv_layer(img)

        conved_size_channels = conved.size()[1]
        conved_size = conved.size()[2]
        
        x = conved.view(self.mb,conved_size_channels,conved_size*conved_size).permute(0,2,1)
        
        for rn_depth in range(len(self.rn_config)):
            temp = range(self.rn_config[rn_depth])
            for rn_id in range(self.rn_config[rn_depth]):
                temp[rn_id] = [torch.unsqueeze(self.rn_layer_dic[rn_depth][rn_id](in_=x,aux=qst),1)]
            if len(temp) is 1:
                x = torch.cat(temp,1)
            else:
                x = torch.squeeze(temp[0],1)
        
        x = self.f_layer(x)

        return x


class RN(nn.Module):

    def __init__(self,args):
        super(RN, self).__init__()
        
        self.mb = 64
        self.ngpu = 1

        self.rn_model = rn_model(mb=self.mb)
        if self.ngpu > 1:
            self.rn_model = torch.nn.parallel.DataParallel(module=self.rn_model,
                                                           device_ids=range(self.ngpu),
                                                           output_device=range(self.ngpu))

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.rn_model(img,qst)
        return x


    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy


    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy


    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)
