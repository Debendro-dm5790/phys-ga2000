import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def logistic(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))

surveyInfo = pd.read_csv('survey.csv')  
ages = surveyInfo['age'].to_numpy()
responses = surveyInfo['recognized_it'].to_numpy()
age_sort = np.argsort(ages)
ages = ages[age_sort]
responses = responses[age_sort]


x = 50
beta_0 = np.linspace(-5,5, 100)
beta_1 = np.linspace(-5,5, 100)
beta = np.meshgrid(beta_0, beta_1)
logistic_grid = logistic(x, *beta)

plt.pcolormesh(*beta, logistic_grid)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$p(y_i|x_i=50,\beta_0, \beta_1)$', fontsize = 16)
plt.savefig('LogisticMesh.png')
plt.show()


plt.plot(ages, responses, '*', label = 'Survey Data')
plt.xlabel('Age in Years')
plt.ylabel('Recognized (1) or Did Not Recognize (0)')
plt.legend()
plt.title('Responses and Ages')
plt.savefig('SurveyData.png')
plt.show()

def calculate_log_likelihood(beta, ages, responses):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [responses[i]*np.log(logistic(ages[i], beta_0, beta_1)/(1-logistic(ages[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-logistic(ages[i], beta_0, beta_1)+epsilon) for i in range(len(ages))]
    logLike = np.sum(np.array(l_list), axis = -1)
    return -1*logLike # return log likelihood

    
logLike = calculate_log_likelihood(beta, ages, responses)
plt.pcolormesh(*beta, logLike)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.savefig('LogLikePlot.png')
plt.show()


grad_ll_arr = np.gradient(logLike)

plt.pcolormesh(*beta, grad_ll_arr[0])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial}{\partial \beta_0} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()
plt.savefig('GradOfLogLikeBeta0.png')
plt.show()

plt.pcolormesh(*beta, grad_ll_arr[1])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial}{\partial \beta_1} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()
plt.savefig('GradOfLogLikeBeta1.png')
plt.show()

def hessian(x):
    """
    https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

hess_ll = hessian(logLike)

plt.pcolormesh(*beta, hess_ll[0, 0, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial^2}{\partial \beta_0^2} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()
plt.savefig('HessianBeta00.png')
plt.show()

plt.pcolormesh(*beta, hess_ll[1, 1, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial^2}{\partial \beta_1^2} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()
plt.savefig('HessianBeta11.png')
plt.show()

plt.pcolormesh(*beta, hess_ll[0, 1, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial^2}{\partial \beta_0 \beta_1} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()
plt.savefig('HessianBeta01.png')
plt.show()

plt.pcolormesh(*beta, hess_ll[1, 0, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 14)
plt.ylabel(r'$\beta_1$ inverse years', fontsize = 14)
plt.title(r'$\frac{\partial^2}{\partial \beta_1 \beta_0} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()
plt.savefig('HessianBeta10.png')
plt.show()