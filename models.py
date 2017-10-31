import numpy

# enum
class Opt():
    SGD = 'sgd'
    SAGA = 'saga'

class Model:
    def update_step(self, training_point, step_size, mu):
        if self.opt == Opt.SGD:
            self.sgd_step(training_point, step_size, mu)
        elif self.opt == Opt.SAGA:
            self.saga_step(training_point, step_size, mu)
    
    def sgd_step(self, training_point, step_size, mu):
        raise NotImplementedError()
        
    def saga_step(self, training_point, step_size, mu):
        raise NotImplementedError()
        
    def loss(self, data):
        raise NotImplementedError()
        
    def reg_loss(self, data, mu):
        raise NotImplementedError()
        

class MatrixFactorization(Model):
    def __init__(self, init_L, init_R, opt):
        self.L = numpy.copy(init_L)
        self.R = numpy.copy(init_R)
        self.opt = opt
        
        self.tableL = {}
        self.tableR = {}
        self.table_sumL = numpy.zeros(self.L.shape)
        self.table_sumR = numpy.zeros(self.R.shape)
        
    def sgd_step(self, training_point, step_size, mu):
        (i, j, val) = training_point
        pred_err = numpy.dot(self.L[i,:], self.R[:,j]) - val
        L_temp      = (1-step_size*mu)*self.L[i,:] - step_size*(pred_err*self.R[:,j])
        self.R[:,j] = (1-step_size*mu)*self.R[:,j] - step_size*(pred_err*self.L[i,:])
        self.L[i,:] = L_temp
    
    # it is possible to do this significantly faster with only sparse updates to L, R    
    def saga_step(self, training_point, step_size, mu):
        (i, j, val) = training_point
        pred_err = numpy.dot(self.L[i,:], self.R[:,j]) - val
        gL = pred_err*self.R[:,j]
        gR = pred_err*self.L[i,:]
        m = len(self.tableL) if len(self.tableL)!=0 else 1
        
        if (i, j) in self.tableL:
            alphaL = self.tableL[(i, j)]
            alphaR = self.tableR[(i, j)]
        else:
            alphaL = numpy.zeros(gL.shape)
            alphaR = numpy.zeros(gR.shape)
            
        self.L[i,:] = (1-step_size*mu)*self.L[i,:] - step_size*(gL - alphaL)
        self.R[:,j] = (1-step_size*mu)*self.R[:,j] - step_size*(gR - alphaR)
        
        self.L -= step_size*(1./m)*self.table_sumL
        self.R -= step_size*(1./m)*self.table_sumR
        
        self.tableL[(i, j)] = gL
        self.tableR[(i, j)] = gR
        self.table_sumL[i,:] += gL - alphaL
        self.table_sumR[:,j] += gR - alphaR
        
    def loss(self, data):
        return sum( (val-numpy.dot(self.L[i,:], self.R[:,j]))**2 for (i, j, val) in data )/len(data)
        
    def reg_loss(self, data, mu):
        return sum( (val-numpy.dot(self.L[i,:], self.R[:,j]))**2 + mu*(numpy.dot(self.L[i,:], self.L[i,:]) + numpy.dot(self.R[:,j], self.R[:,j])) for (i, j, val) in data )/len(data)
        
class LogisticRegression(Model):
    def __init__(self, init_w, opt):
        self.w = numpy.copy(init_w)
        self.opt = opt
        
        self.table = {}
        self.table_sum = numpy.zeros(self.w.shape)
        
    # dot product with a sparse feature vector
    def dot_product(self, x):
        return sum(self.w[k]*v for (k,v) in x.iteritems())
 
    def sgd_step(self, training_point, step_size, mu):
        (i, x, y) = training_point
        p = 1./(1 + numpy.exp(y*self.dot_product(x)))
        for (k, v) in x.iteritems():
            self.w[k] = (1-step_size*mu)*self.w[k] - step_size*(-1*p*y*v)
        
    def saga_step(self, training_point, step_size, mu):
        (i, x, y) = training_point
        p = 1./(1 + numpy.exp(y*self.dot_product(x)))
        g = -1*p*y
        alpha = self.table[i] if i in self.table else 0
        m = len(self.table) if len(self.table)!= 0 else 1
        
        for (k, v) in x.iteritems():
            self.w[k] = (1-step_size*mu)*self.w[k] - step_size*( (g-alpha)*v + self.table_sum[k]/m )
        
        self.table[i] = g
        for (k, v) in x.iteritems():
            self.table_sum[k] += (g-alpha)*v
        
    def loss(self, data):
        return sum( numpy.log(1+numpy.exp(-1*y*self.dot_product(x))) for (x,y) in data )/len(data)
        
    def reg_loss(self, data, mu):
        return self.loss(data) + 0.5*mu*numpy.dot(w, w)

