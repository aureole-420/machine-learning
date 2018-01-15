import numpy as np 

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        u = np.random.uniform()
        v = np.random.uniform()
        return np.sqrt(-2 * np.log(u)) * np.cos(2*np.pi*v)
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu
        self.Sigma = Sigma
        U,s,V = np.linalg.svd(Sigma)
        # print np.allclose(U,V.T)
        self.A = U.dot(np.sqrt(np.diag(s)))
        self.normSpl = UnivariateNormal(0, 1)
        
    def sample(self):
        Z = np.zeros((self.Mu.shape[0],))
        for i in range(Z.shape[0]):
            Z[i] = self.normSpl.sample()
        return self.Mu + self.A.dot(Z)
        
        
        
        
    


    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    __cumProbList = []
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap
        # print "<Categorical>:", len(ap)
        cumProb = 0.0
        self.__cumProbList.append(cumProb)
        for prob in self.ap:
            cumProb += prob
            self.__cumProbList.append(cumProb)
        # print self.__cumProbList

    def sample(self):
        x = np.random.uniform()
        for index in range(len(self.__cumProbList)-1):
            lowB = self.__cumProbList[index]
            upB = self.__cumProbList[index+1]
            if (lowB <= x and x <= upB):
                return index 
        return len(self.__cumProbList)-1

        


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm
    
    def sample(self):
        for i in range(len(self.ap)):
            weight = self.ap[i]
            # print weight*self.pm[0].sample()
            if (i == 0): 
                sample = weight*self.pm[0].sample()
            else: 
                sample += weight*self.pm[i].sample()
              
        return sample
