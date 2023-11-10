import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def logistic(x, beta):
    return 1/(1+np.exp(-(beta[0]+beta[1]*x)))

surveyInfo = pd.read_csv('survey.csv')  
ages = surveyInfo['age'].to_numpy()
responses = surveyInfo['recognized_it'].to_numpy()
age_sort = np.argsort(ages)
ages = ages[age_sort]
responses = responses[age_sort]

def little_neg_log_likelihood(beta, ages, responses):
    epsilon = 1e-16
    l_list = [-1*responses[i]*np.log(logistic(ages[i], beta)/(1-logistic(ages[i], beta)+epsilon)) 
              - np.log(1-logistic(ages[i], beta)+epsilon) for i in range(len(ages))]
    return l_list

# Covariance matrix of parameters
def Covariance(hess_inverse, resVariance):
    return hess_inverse * resVariance

#Error of parameters
def error(hess_inverse, resVariance):
    covariance = Covariance(hess_inverse, resVariance)
    return np.sqrt( np.diag( covariance ))

betaStart = np.array([0,0])

result = optimize.minimize(lambda beta,xs,ys: np.sum(little_neg_log_likelihood(beta, xs, ys)), betaStart,  args=(ages, responses))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(responses)-len(betaStart)) 
errorFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', errorFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))

Ages = np.linspace(0,80,1000)

plt.plot(ages, responses, '*', label = 'Survey Data')
plt.plot(Ages,logistic(Ages, result.x), label = 'Optimal Model')
plt.xlabel('Age in Years')
plt.ylabel('Recognized (1) or Did Not Recognize (0)')
plt.legend()
plt.title('Responses and Ages Actual and Optimal Model')
plt.savefig('SurveyDataVersusOptimalModel.png')
plt.show()

negLogLikeOptParam = np.sum(little_neg_log_likelihood(result.x, ages, responses))
print('The optimized negative log likelihood associated with these optimal parameters is ',negLogLikeOptParam)