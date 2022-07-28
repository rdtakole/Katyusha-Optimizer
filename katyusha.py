from torch.optim import Optimizer
import copy


class Katyusha_k(Optimizer):
    """Optimizer for calculating the gradient step in the inner loop of the algorithm
    Args:
        params (iterable): Network parameters
        lips : Lipschitz constant
        m : No of iterations to be performed in the inner loop    
    """
    def __init__(self, params, lips = 10, m = 500):
        if lips < 0.0:
            raise ValueError("Invalid lipschitz constant : {}".format(lips))
        if m < 0.0:
            raise ValueError("Invalid frequency value: {}".format(m))
        self.x_tilda = None
        self.tau_2 = 0.5
        self.counter = 0
        self.lips = lips
        self.m = m    
        defaults = dict(lips=lips, m=m)
        super(Katyusha_k, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.avg_y

    def set_outparam(self, new_param):
        """ Set the out parameter for the epoch
        """
        self.tau_1 = 2/(self.counter+4)
        self.counter += 1
        self.alpha = 1/(3*self.tau_1*self.lips) 
        if self.x_tilda is None:
            self.x_k = copy.deepcopy(new_param)
            self.y_k = copy.deepcopy(new_param)
            self.z_k = copy.deepcopy(new_param)
            self.avg_y = copy.deepcopy(new_param)
        self.x_tilda = copy.deepcopy(new_param)
        for xtil_group, new_group in zip(self.x_tilda, new_param):  
            for xtil, new_u in zip(xtil_group['params'], new_group['params']):
                xtil.grad = new_u.grad.clone()
        for avg_grp, param in zip(self.avg_y,self.y_k):
          for avg, par in zip(avg_grp['params'],param['params']):
            avg.data = torch.zeros_like(par.data)        

    def compute_xk(self):
        for x,y,z,x_tilda in zip(self.param_groups,self.y_k,self.z_k,self.x_tilda):
            for xk,yk,zk,xtil in zip(x['params'],y['params'],z['params'],x_tilda['params']):
                xk.data = (self.tau_1*zk.data) + (self.tau_2*xtil.data) + ((1-self.tau_1-self.tau_2)*yk.data)
            

    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, xtilda_group, yk, zk, avg in zip(self.param_groups, params, self.x_tilda, self.y_k, self.z_k, self.avg_y):
            for p, q, u, y, z, av in zip(group['params'], new_group['params'], xtilda_group['params'], yk['params'], zk['params'], avg['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # Katyusha gradient update 
                new_d = p.grad.data - q.grad.data + u.grad.data
                y.data = p.data - (1/(3*self.lips))*new_d
                z.data = z.data - (self.alpha*new_d)
                av.data = av.data + (1/self.m)*y.data


class Katyusha_Snapshot(Optimizer):
    """Optimizer for calculating the mean gradient w.r.t snapshot parameters (snapshot)
    Args:
        params (iterable): Network parameters
    """
    def __init__(self, params):
        defaults = dict()
        super(Katyusha_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_parameters):
        """Copies the parameters from Katyusha_k optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_parameters): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]
