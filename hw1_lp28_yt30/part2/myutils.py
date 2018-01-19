# from sklearn import linear_model
import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
# import plot_utils

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval,reg_vec):
  
  #reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
  error_train = np.zeros((len(reg_vec),))
  error_val = np.zeros((len(reg_vec),))

  for i in range(len(reg_vec)):
      reglinear_reg = RegularizedLinearReg_SquaredLoss()
      theta = reglinear_reg.train(X, y, reg_vec[i], num_iters=1000)
      error_train[i] = reglinear_reg.loss(theta, X, y, 0.0)
      error_val[i] = reglinear_reg.loss(theta, Xval, yval, 0.0)
    
  return reg_vec, error_train, error_val