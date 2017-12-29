import numpy as np
import math
import random
import time
import json
import pprint

class Backpropagation:
    def __init__(self, dataset, hidden, E=None, v=None, w=None, alpha=None, epoh=None):
        self.dataset = dataset
        self.alpha = alpha
        self.E = E
        self.hidden = hidden
        self.jumlah_epoh = 0
        self.epoh = epoh

        length = []
        for i in dataset:
            length.append(i[0])
            
        x, y = np.asarray(length).shape

        if v==None:
            v_temp = np.random.rand(x, y)
            v = []
            for i in v_temp:
                v.append([i.tolist(), 1])
        if w==None:
            w_temp = np.random.rand(hidden)
            w = []
            w.append(w_temp.tolist())
            w.append(1)

        self.v = v
        self.w = w
    
    def z_net(self, data_input, weight):
        z = []
        for i in range(self.hidden):
            z.append(np.sum(map(lambda x, y: float(x)*float(y), data_input[0], weight[0])))
        z_net = (np.asarray(z) + weight[1]).tolist()
        return z_net
    
    def f_z_net(self, z_net):
        f_z_net = map(lambda x: float(1)/(float(1) + math.e**float(-x)), z_net)
        return f_z_net

    def y_net(self, data_input, weight):
        y_net = np.sum(map(lambda x, y: float(x) * float(y), data_input, weight[0]))
        y_net = y_net + float(weight[1])
        return y_net

    def f_y_net(self, y_net):
        f_y_net = 1/(1+math.e**(-y_net))
        return f_y_net

    def teta(self, target, f_y_net):
        return (target[2] - f_y_net) * f_y_net * (1-f_y_net)

    def delta_w(self, teta, f_z_net, data_input):
        net = (np.asarray(f_z_net) * teta * self.alpha)
        bias = data_input[1] * alpha * teta
        return [net.tolist(), bias] # bias dihitung

    def teta_net(self, teta, weight):
        return (teta * np.asarray(weight[0])).tolist() # bias diabaikan
    
    def teta_net_zet(self, teta_net, f_z_net):
        first = map(lambda x, y: float(x) * float(y), teta_net, f_z_net)
        second = map(lambda x, y: float(x) * float(1 - y), first, f_z_net)
        return second
    
    def delta_v(self, teta_net_zet, data_input):
        net = []
        for d in data_input[0]:
            net.append(map(lambda x: self.alpha * float(x) * float(d), teta_net_zet))
        bias = map(lambda x: self.alpha * float(x) * float(data_input[1]), teta_net_zet)
        net.append(bias)
        return net

    def new_w(self, old_w, delta_w):
        w = []
        w.append(map(lambda x, y: float(x) + float(y), old_w[0], delta_w[0]))
        w.append(old_w[1] + delta_w[1])
        return w

    def new_v(self, old_v, delta_v):
        v = []
        hitung = []
        temp = []
        bias = []
        for i in old_v:
            bias.append(i[1])
            temp.append(i[0])
        temp.append(bias)
        for i, t in enumerate(temp):
            hitung.append(map(lambda x, y: float(x) + float(y), temp[i], delta_v[i]))   
        for i, t in enumerate(old_v):
            v.append([hitung[i], max(hitung)[i]])    
        return v
    
    def get_MSE(self, target, y_net, data_input):
        return ((target - y_net)**2)/len(data_input[0])
    
    def apa_lanjut(self, MSE):
        return False if MSE < self.E else True
    
    def learn_process(self):
        for i in self.dataset:
            inp = i[0]
            bias = i[1]
            target = i[2]
            result = []
            z_net = []
            f_zet_net = []
            y_net = []
            f_y_net = []
            znet = []
            fznet = []
                
            for j in self.v:
                z_net = self.z_net(i, j)
                f_zet_net = self.f_z_net(z_net)

            y_net = self.y_net(f_zet_net, self.w)
                
            f_y_net = self.f_y_net(y_net)
                
            teta = self.teta(i, f_y_net)
                
            delta_w = self.delta_w(teta, f_zet_net, i)
                
            teta_net = self.teta_net(teta, self.w)
            
            teta_net_z = self.teta_net_zet(teta_net, f_zet_net)
                
            delta_v = self.delta_v(teta_net_z, i)
                
            new_w = self.new_w(self.w, delta_w)
                
            new_v = self.new_v(self.v, delta_v)
                
            if self.E != None:
                get_MSE = self.get_MSE(i[2], f_y_net, self.dataset)
            
        self.v = new_v
        self.w = new_w

        if self.E != None:
            apa_lanjut = self.apa_lanjut(get_MSE)
            return apa_lanjut
        else:
            return True

    def train(self):
        if self.epoh == None:
            epoh = 1
            while True:
                learn = self.learn_process()
                if learn:
                    epoh = epoh+1
                else:
                    break
            self.jumlah_epoh = epoh
        else:
            for i in range(self.epoh):
                learn = self.learn_process()
                if learn:
                    self.jumlah_epoh = i+1
                else:
                    break
        return self.v, self.w, self.jumlah_epoh

        
        
if __name__ == '__main__':
    timestart = time.time()
    dataset = [
        [[0.4,0.6,0.3,0.6],1,1],
        [[0.6,0.6,0.3,0.6],1,1],
        [[0.2,0.4,0.6,1],1,0],
        [[0.2,0.4,0.3,1],1,0]
    ]

    alpha = 0.2

    b = Backpropagation(dataset, alpha=alpha, hidden=4, E=0.0001, epoh=100)

    train = b.train()
    data = {
        "new_w":train[1],
        "new_v":train[0],
        "epoh":train[2]
    }
    pprint.pprint(data)
    print "Excecution Time : %s seconds" % (time.time() - timestart)