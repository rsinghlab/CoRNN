from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch.autograd import Variable

"""
This file contains three different CNN models, they contains different number of convolutional layers.

"""


def calculate_seq_len(seq_len, kernal, pool_size, pad, stride):
	"""
	A helper function for calculating the sequence size after convolution and maxpooling
	"""


	seq_len = seq_len + 2 * pad - (kernal) + 1
	return int((seq_len - (pool_size -1) -1)/stride +1)


def calculate_seq_len_n(seq_len, params):
    seq_len = seq_len + 2 * params["pad"] - (params["conv_k"]) + 1
    if params["do_maxpool"]:
        seq_len = int((seq_len - (params["pool_k"] -1) -1)/params["stride"] +1)
    return seq_len



class Print_shape(nn.Module):
    def __init__(self):
        super(Print_shape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Print_value(nn.Module):
    def __init__(self):
        super(Print_value, self).__init__()

    def forward(self, x):
        print(x)
        return x


class cnnNlayer(nn.Module):
    def __init__(self, seq_len, args, task):
        super(cnnNlayer, self).__init__()

        self.net=nn.Sequential()
        updated_seq_len = seq_len
        last_out_channel_size = 0

        print_s = Print_shape()
        print_v = Print_value()
        # self.net.add_module("print_1",print_s)
        # self.net.add_module("print_2",print_s)
        for idx in range(args["num_layers"]):
            # self.net.add_module("print",print_s)
            params = args["layer_{}".format(idx)]
            last_out_channel_size = params["out"]
            # self.net.add_module("print",print_s)
            # print(params)
            conv =  nn.Conv1d(in_channels=params["in"],out_channels=params["out"],\
                    kernel_size=params["conv_k"],padding=params["pad"])
            # torch.nn.init.xavier_uniform(conv.weight)
            self.net.add_module("conv{}".format(idx), conv)
           
            # self.net.add_module("print_3_{}".format(idx),print_s)

            if params["do_relu"]:
                self.net.add_module("relu{}".format(idx),nn.ReLU(inplace=True))
            if params["do_maxpool"]:
                self.net.add_module("pool{}".format(idx),\
                    nn.MaxPool1d(params["pool_k"], stride=params["stride"]))
                
            # self.net.add_module("print_4_{}".format(idx),print_s)
            if params["do_norm"]:
                self.net.add_module("norm{}".format(idx),nn.BatchNorm1d(params["out"]))

            # print("layer {}, seq len:".format(idx,updated_seq_len))
            self.net.add_module("drop{}".format(idx),\
                nn.Dropout(params["dropout"]))
            
            updated_seq_len = calculate_seq_len_n(updated_seq_len,params)
            # print(updated_seq_len)
            # print("layer {}, seq len:".format(idx,updated_seq_len))


        linear_input_size = int(updated_seq_len) * last_out_channel_size
        # print("linear_input_size:",linear_input_size)


        fc_args = args["fc"]
        self.net.add_module("flatten", nn.Flatten(1,-1))


        # self.net.add_module("print_5",print_s)

        linear1_args = fc_args["linear_1"]
        linear2_args = fc_args["linear_2"]
        linear3_args = fc_args["linear_3"]

        # self.net.add_module("print",print_s)
        fc1 = nn.Linear(linear_input_size,linear_input_size,linear1_args["bias"])
        # nn.init.normal(fc1, mean=linear1_args["mean"], std=linear1_args["std"])
        self.net.add_module("linear1",fc1 )
        
        if linear1_args["do_relu"] == True:
            self.net.add_module("relu_linear1",nn.ReLU(inplace=True))
        self.net.add_module("drop_linear1",\
                nn.Dropout(linear1_args["dropout"]))

        # self.net.add_module("print_6",print_s)

        fc2 = nn.Linear(linear_input_size,linear2_args["kernal"],linear2_args["bias"])
        # nn.init.normal(fc2, mean=linear2_args["mean"], std=linear2_args["std"])
        self.net.add_module("linear2", fc2)

        if linear2_args["do_relu"] == True:
            self.net.add_module("relu_linear2",nn.ReLU(inplace=True))
        self.net.add_module("drop_linear2".format(idx),\
                nn.Dropout(linear2_args["dropout"]))

        # self.net.add_module("print_7",print_s)

        
        if task == "cla":
            fc3 = nn.Linear(linear2_args["kernal"],2,linear3_args["bias"])
        elif task == "reg":
            fc3 = nn.Linear(linear2_args["kernal"],1,linear3_args["bias"])
            
        # nn.init.normal(fc3, mean=linear3_args["mean"], std=linear3_args["std"])
        self.net.add_module("output", fc3)

        if linear3_args["do_relu"] == True:
            self.net.add_module("relu_linear3",nn.ReLU(inplace=True))
        
        self.net.add_module("drop_linear3",\
                nn.Dropout(linear3_args["dropout"]))

        # l = [module for module in self.net if type(module) != nn.Sequential]
        # print(l)

    def forward(self, inputs):


        x = self.net(inputs)

        return x

class lstm(nn.Module):
    def __init__(self, seq_len, args):
        super(lstm, self).__init__()
        self.input_dim = 6
        self.hidden_dim = args["hidden_dim"]
        self.n_layers = args["n_layers"]
        self.linear_dim = args["hidden_dim"]
        self.rnn_type = 'LSTM'
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.classifier = nn.Linear(self.linear_dim,2)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        #print(inputs.size()) #[batch,100,6]
        hidden = self.init_hidden(len(inputs))
        lstm_out, (ht, ct) = self.lstm_layer(inputs, hidden)
        # print(ht.size())
        # print(ht[:,-1,:].size())
        # print(ht[-1].size())

        classifier_input = ht[-1]
        x = self.classifier(classifier_input)
        return x
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()))
        else:
            return Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_())


class gru(nn.Module):
    def __init__(self, seq_len, args):
        super(gru, self).__init__()
        self.input_dim = 6
        self.hidden_dim = args["hidden_dim"]
        self.n_layers = args["n_layers"]
        self.linear_dim = args["hidden_dim"]
        self.rnn_type = 'GRU'

        try:
            self.add_mean_evec = args["mean_evec"]
        except:
            self.add_mean_evec = False

        if self.add_mean_evec == True:
            self.linear_dim = self.linear_dim + 1
        try:
            self.bidirectional = args["bidirectional"]
        except:
            self.bidirectional = False
        self.gru_layer = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional = self.bidirectional)
        self.classifier = nn.Linear(self.linear_dim,2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, inputs, mean_evec = None, use_mask = False, mask = None):
        # print(mean_evec)
        # print("1",inputs)
        inputs = torch.transpose(inputs, 1, 2)
        # print(inputs.size()) #[batch,100,6]
        if use_mask:
            inputs = torch.mul(inputs,mask)
        # print(mask)
        # print(inputs)
        hidden = self.init_hidden(len(inputs))
        gru, ht = self.gru_layer(inputs, hidden)
        # print("2",ht)

        classifier_input = ht[-1]
        # print("3",classifier_input)
        # print(classifier_input.size())
        # print(mean_evec.size())
        if self.add_mean_evec:
            # print(mean_evec)
            mean_evec = torch.unsqueeze(mean_evec,1)
            classifier_input = torch.cat((classifier_input,mean_evec),1)
        # print("4",classifier_input)
        # print(classifier_input.size())
        x = self.classifier(classifier_input)
        # x = self.softmax(x)
        # print(x)
        # exit()
        return x
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()))
        else:
            if self.bidirectional == True:
                return Variable(weight.new(self.n_layers*2, bsz, self.hidden_dim).zero_())
            else:
                return Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_())
            


#========================= legacy code ==========================================



class gru_debug(nn.Module):
    def __init__(self, seq_len, args):
        super(gru_debug, self).__init__()
        self.input_dim = 6
        self.hidden_dim = args["hidden_dim"]
        self.n_layers = args["n_layers"]
        self.linear_dim = args["hidden_dim"]
        self.rnn_type = 'GRU'

        try:
            self.add_mean_evec = args["mean_evec"]
        except:
            self.add_mean_evec = False

        if self.add_mean_evec == True:
            self.linear_dim = self.linear_dim + 1
        try:
            self.bidirectional = args["bidirectional"]
        except:
            self.bidirectional = False
        self.gru_layer = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional = self.bidirectional)
        self.classifier = nn.Linear(self.linear_dim,2)
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, inputs, mean_evec = None):
        # print("1",inputs)
        inputs = torch.transpose(inputs, 1, 2)
        # print(inputs.size()) #[batch,100,6]
        hidden = self.init_hidden(len(inputs))
        gru, ht = self.gru_layer(inputs, hidden)
        # print("2",ht)

        ht_mod = ht[-1]
        # print("3",classifier_input)
        # print(classifier_input.size())
        # print(mean_evec.size())
        if self.add_mean_evec:
            mean_evec = torch.unsqueeze(mean_evec,1)
            classifier_input = torch.cat((ht_mod,mean_evec),1)
        # print("4",classifier_input)
        # print(classifier_input.size())
        x = self.classifier(classifier_input)
        x = self.softmax(x)


        # if "nan" in mean_evec:
        #     print("nan in mean evec")
        #     print(mean_evec)
        # if "nan" in inputs:
        #     print("nan in inputs")
        # if "nan" in ht:
        #     print("nan in ht")

        # if "nan" in classifier_input:
        #     print("nan in classifier input")

        # if "nan" in x:
        #     print("nan in x")
                # print("1",inputs)
                # print("2",ht)
                # print("3",mean_evec)
                # print("4",ht_mod)
                # print("5",classifier_input)
                # print("6",x)
                # exit()
        return x
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_()))
        else:
            if self.bidirectional == True:
                return Variable(weight.new(self.n_layers*2, bsz, self.hidden_dim).zero_())
            else:
                return Variable(weight.new(self.n_layers, bsz, self.hidden_dim).zero_())
class cnn1layer(nn.Module):
    def __init__(self,seq_len, args, task):
        super(cnn1layer, self).__init__()
        """
        This model has only 1 convolutionary layer. 

        Tunable parameters: 

            ====Tune these variables in hm2ab.py===

            outX: number of output channel in the convolutional layer X
            kX : kernal size of convolutional layer X
            poolX: kernal size of maxpooling after convolutional layer X
            strideX: maxpooling step size of after convolutional layer X
            dropoutX: dropout rate after first cnn layer
            linear1: the output size of first linear layer (feed forward layer)
            linear2: the output size of second linear layer (feed forward layer)

        """
        self.conv1 = nn.Conv1d(in_channels=6,out_channels=args["out1"],kernel_size=args["k1"])

        self.pool1 = nn.MaxPool1d(args["pool1"], stride=args["stride1"])
        self.drop1 = nn.Dropout(args["dropout1"])
        seq_len1 = calculate_seq_len(seq_len, args["k1"], args["pool1"], 0, args["stride1"])

        linear_input_size = seq_len1 * args["out1"]

        self.linear1 = nn.Linear(linear_input_size,args["linear1"])
        self.linear2 = nn.Linear(args["linear1"],args["linear2"])

        if task == "cla":
            self.output = nn.Linear(args["linear2"],2)
        elif task == "reg":
            self.output = nn.Linear(args["linear2"],1)

    def forward(self, inputs):
        # inputs shape (batch, 100, 6)
        x = self.conv1(inputs)  
        x = F.relu(x)
        x = self.pool1(x) 
        
        x = self.drop1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)

        return x

class cnn2layer(nn.Module):
    def __init__(self, seq_len, args, task):
        super(cnn2layer, self).__init__()
        """
        This model has 2 convolutionary layers. 
        """
        self.conv1 = nn.Conv1d(in_channels=6,out_channels=args["out1"],kernel_size=args["k1"])
        self.pool1 = nn.MaxPool1d(args["pool1"], stride=args["stride1"])
        self.drop1 = nn.Dropout(args["dropout1"])

        seq_len1 = calculate_seq_len(seq_len, args["k1"], args["pool1"], 0,  args["stride1"])

        self.conv2 = nn.Conv1d(in_channels=args["out1"],out_channels=args["out2"],kernel_size=args["k2"])
        self.drop2 = nn.Dropout(args["dropout2"])
        
        seq_len2 = (seq_len1-args["k2"]+1)
        linear_input_size = int(seq_len2)*args["out2"]

        self.linear1 = nn.Linear(linear_input_size,args["linear1"])
        self.linear2 = nn.Linear(args["linear1"],args["linear2"])
        if task == "cla":
            self.output = nn.Linear(args["linear2"],2)
        elif task == "reg":
            self.output = nn.Linear(args["linear2"],1)

    def forward(self, inputs):
        # inputs shape (batch, 100, 6)
        x = self.conv1(inputs) 
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        # print(x.shape)

        x = self.conv2(x)  
        x = F.relu(x)
        x = self.drop2(x)
        # print(x.shape)
        
        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)

        return x

class cnn3layer(nn.Module):
    def __init__(self, seq_len, args, task):
        super(cnn3layer, self).__init__()
        # print(args)
        self.conv1 = nn.Conv1d(in_channels=6,out_channels=args["out1"],kernel_size=args["k1"])
        #torch.nn.init.xavier_uniform(self.conv1.weight)

        self.pool1 = nn.MaxPool1d(args["pool1"], stride=args["stride1"])
        self.drop1 = nn.Dropout(args["dropout1"])

        # (batch,(seq_len-k+1)/pool_size,out_size)
        seq_len1 = calculate_seq_len(seq_len, args["k1"], args["pool1"], 0, args["stride1"])
        # print("seq_len1:",seq_len1)

        self.conv2 = nn.Conv1d(in_channels=args["out1"],out_channels=args["out2"],kernel_size=args["k2"])
        #torch.nn.init.xavier_uniform(self.conv2.weight)
        self.pool2 = nn.MaxPool1d(args["pool2"], stride=args["stride2"])
        self.drop2 = nn.Dropout(args["dropout2"])
        seq_len2 = calculate_seq_len(seq_len1, args["k2"], args["pool2"], 0, args["stride2"])
        # (batch,(((seq_len-k+1)/pool_size)-k+1）/pool_size,out_size*1.5)

        # print("seq_len2:",seq_len2)

        self.conv3 = nn.Conv1d(in_channels=args["out2"],out_channels=args["out3"],kernel_size=args["k3"])
        #torch.nn.init.xavier_uniform(self.conv3.weight)
        # self.pool2 = nn.MaxPool1d(pool_size, stride=pool_size)
        self.drop3 = nn.Dropout(args["dropout3"])
         # (batch,(((seq_len-k+1)-k+1)/pool_size),out_size*1.5)
        seq_len3 = (seq_len2-args["k3"]+1)

        # print("seq_len3:",seq_len3)

        linear_input_size = int(seq_len3)*args["out3"]

        # print(linear_input_size)

        self.linear1 = nn.Linear(linear_input_size,args["linear1"])
        self.linear2 = nn.Linear(args["linear1"],args["linear2"])

        if task == "cla":
            self.output = nn.Linear(args["linear2"],2)
        elif task == "reg":
            self.output = nn.Linear(args["linear2"],1)

    def forward(self, inputs):
        # inputs shape (batch, 100, 6)
        x = self.conv1(inputs) 
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        # print(x.shape)

        x = self.conv2(x)  
        x = F.relu(x)
        x = self.pool2(x) 
        x = self.drop2(x)
        # print(x.shape)

        x = self.conv3(x) 
        x = F.relu(x)
        x = self.drop3(x)
        # print(x.shape)
        
        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)

        return x