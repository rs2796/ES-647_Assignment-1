import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.patches as mpatches
from sklearn.model_selection import cross_validate
import itertools as it
import sys

l = [0,.0001,.001,.01,.1,1,1.5,2,3,4,5]
l1 = [70,80,90,99]
lk = [0,.01,.1,1]
x = input('Enter either B or D, B for Boston and D for dataset: ')
y = int(input('Enter 1,2,3 or 4: '))


def estimator(y,a):
    if y == 2:
        
        dataset_reg = linear_model.Ridge(alpha=a, copy_X=True, fit_intercept=True, max_iter=1000,normalize=True, random_state=None, solver='auto', tol=0.001)
    else:
        dataset_reg = linear_model.Lasso(alpha=a, fit_intercept=True, normalize=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        
    return dataset_reg

def loaddata(x):
    if x == 'D':
        dataset = datasets.load_diabetes()
    else:
        dataset = datasets.load_boston()
    dataset_X = dataset.data
    dataset_Y = dataset.target
    dataset_Y = dataset_Y.reshape(dataset_Y.shape[0],1)
    data = np.hstack((dataset_X,dataset_Y))
    return data
    


def q23(x,y):
 

    data = loaddata(x)
    fig, s  = plt.subplots(2,4,sharey = 'row')

    for j in range(1,5):
        np.random.shuffle(data)
        xa = [];xb = [];ya = [];yb = [];yf = [];yc = []
        
        a = (data.shape[0]*l1[j-1])//100
        dataset_X_train = data[:a+1,:-1]
        dataset_X_test = data[a+1:,:-1]
        dataset_Y_train = data[:a+1,-1:]
        dataset_Y_test = data[a+1:,-1:]
        
    
        for i in range(0,11):
            if i == 0:
                dataset_reg = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
            else:
                
                dataset_reg = estimator(y,l[i])
           
            result = cross_validate(dataset_reg,dataset_X_train, dataset_Y_train,cv =5, scoring = 'neg_mean_squared_error')
            
            
            
            r = result['test_score']

               

            dataset_reg.fit(dataset_X_train, dataset_Y_train)
           
            dataset_model_intercept = dataset_reg.intercept_
            dataset_model_coeff = dataset_reg.coef_

            dataset_Y_pred = dataset_reg.predict(dataset_X_test)
            dataset_Y_train_pred = dataset_reg.predict(dataset_X_train)
            dataset_total_variance = np.var(dataset_Y_train)
            dataset_explained_variance = np.var(dataset_Y_train_pred)
            dataset_train_error = mean_squared_error(dataset_Y_train, dataset_Y_train_pred)
            dataset_test_error = mean_squared_error(dataset_Y_test, dataset_Y_pred)
            dataset_R2_score = r2_score(dataset_Y_train, dataset_Y_train_pred)
            xa.append(i)
            xb.append(l[i])
            ya.append(dataset_train_error)
            yb.append(dataset_test_error)
            yc.append(dataset_R2_score)
            yf.append(np.mean(r)*-1)

    
        line1, = s[0,j-1].plot(xb,ya,'b-',label='Training Error')
        line2, = s[0,j-1].plot(xb,yb,'r-',label = 'Test Error')
        line3, = s[0,j-1].plot(xb,yf,'r--',label = 'Validation Error')

        s[0,j-1].legend((line1,line2,line3),('Training Error','Test Error','Validation Error'),loc = 0)

        s[0,j-1].grid()
        if j == 1:
            s[0,j-1].set_ylabel('TrainingError & TestError',fontsize = 12,fontweight = 'bold')
    

        s[0,j-1].set_title('Training Data = '+str(l1[j-1])+'%')
    

        line4, = s[1,j-1].plot(xb,yc,'k--',label = 'R^2 Score')
        s[1,j-1].legend(loc = 0)
    
        s[1,j-1].grid()
        if j == 1:
            s[1,j-1].set_ylabel('R^2 Score',fontsize = 12,fontweight = 'bold')
    
        s[1,j-1].set_xlabel(r'$\lambda$')

    if x == 'D' and y == 2:
        fig.suptitle('RidgeCV(Diabetes)', fontsize=14, fontweight='bold')
    elif x == 'D' and y == 3:
        fig.suptitle('Lasso(Diabetes', fontsize=14, fontweight='bold')
    elif x == 'B' and y == 2:
        fig.suptitle('RidgeCV(Boston)', fontsize=14, fontweight='bold')
    elif x == 'B' and y == 3:
        fig.suptitle('Lasso(Boston)', fontsize=14, fontweight='bold')
   
  
    plt.show(block = False)


    
def q1(x):
   

    data = loaddata(x)
    fig, s  = plt.subplots(2,4,sharey = 'row')



    
    for j in range(1,5):
        xa = [];xb = [];ya = [];yb = [];yc = []
    
       
        for i in it.chain(range(50,91,10),range(95,100,4)):

            a = (data.shape[0]*i)//100
            np.random.shuffle(data)
    
    
            dataset_X_train = data[:a+1,:-1]
            dataset_X_test = data[a+1:,:-1]
            dataset_Y_train = data[:a+1,-1:]
            dataset_Y_test = data[a+1:,-1:]
            dataset_reg = linear_model.Ridge(alpha=lk[j-1], copy_X=True, fit_intercept=True, max_iter=None,normalize=True, random_state=None, solver='auto', tol=0.001)
        
            dataset_reg.fit(dataset_X_train, dataset_Y_train)
            dataset_model_intercept = dataset_reg.intercept_
            dataset_model_coeff = dataset_reg.coef_
            dataset_Y_pred = dataset_reg.predict(dataset_X_test)
            dataset_Y_train_pred = dataset_reg.predict(dataset_X_train)
            dataset_total_variance = np.var(dataset_Y_train)
            dataset_explained_variance = np.var(dataset_Y_train_pred)
            dataset_train_error = mean_squared_error(dataset_Y_train, dataset_Y_train_pred)
            dataset_test_error = mean_squared_error(dataset_Y_test, dataset_Y_pred)
            dataset_R2_score = r2_score(dataset_Y_train, dataset_Y_train_pred)
            xa.append(a)
            xb.append(i)
            ya.append(dataset_train_error)
            yb.append(dataset_test_error)
            yc.append(dataset_R2_score)

    
        line1, = s[0,j-1].plot(xb,ya,'b--',label='Training Error')
        line2, = s[0,j-1].plot(xb,yb,'r--',label = 'Test Error')
        s[0,j-1].legend((line1,line2),('Training Error','Test Error'),loc = 0)

        s[0,j-1].grid()
        if j == 1:
            s[0,j-1].set_ylabel('TrainingError & TestError',fontsize = 12,fontweight = 'bold')
    
        s[0,j-1].set_xlabel('Training Data(in percentage)')
        s[0,j-1].set_title(r'$\lambda$ = '+str(lk[j-1]))

        line3 = s[1,j-1].plot(xa,yc,'k--',label = 'R^2 Score')
        s[1,j-1].legend(loc = 0)
    
        s[1,j-1].grid()
        if j == 1:
            s[1,j-1].set_ylabel('R^2 Score',fontsize = 12,fontweight = 'bold')
    
        s[1,j-1].set_xlabel('# of training examples')
    

    if x == 'D' and y == 1:
        fig.suptitle('l_2-regularization(Diabetes)', fontsize=14, fontweight='bold')
    elif x == 'B' and y == 1:
        fig.suptitle('l_2-regularization(Boston)', fontsize=14, fontweight='bold')
  
    plt.show(block = False)

    

def q4(x):
    data = loaddata(x)


    for i in range(0,data.shape[1]-1):
        av = np.mean(data[:,i])
        sd = np.std(data[:,i])

        data[:, i] = (data[:, i]-av)/sd
            
  
    b = np.full((data.shape[0],data.shape[1]+1),1)
    b[:,1:] = data
    
    fig, s  = plt.subplots(2,4,sharey = 'row')
    for j in range(1,5):
        xa = []; xb = [];ya = [];yb = [];yc = []
        

        for i in it.chain(range(50,91,10),range(95,100,4)):
            

            np.random.shuffle(b)

            
                

            a = (b.shape[0]*i)//100

            dataset_X_train = b[:a+1,:-1]
            dataset_X_test = b[a+1:,:-1]
            dataset_Y_train = b[:a+1,-1]

            dataset_Y_test = b[a+1:,-1]
            
            B = np.linalg.pinv(np.matmul(dataset_X_train.transpose(),dataset_X_train)+lk[j-1]*np.eye(dataset_X_train.shape[1]))
            w = np.matmul(B,np.matmul(dataset_X_train.transpose(),dataset_Y_train))
            
            ypredictedtrain = np.matmul(dataset_X_train,w)
      
            ypredictedtest = np.matmul(dataset_X_test,w)
            trainingerror = mean_squared_error(dataset_Y_train, ypredictedtrain)
            testerror = mean_squared_error(dataset_Y_test, ypredictedtest)
            r2score = 1  -  trainingerror/(np.var(dataset_Y_train))
            xa.append(a)
            xb.append(i)
            ya.append(trainingerror)
            yb.append(testerror)
            yc.append(r2score)

        line1, = s[0,j-1].plot(xb,ya,'b--',label='Training Error')
        line2, = s[0,j-1].plot(xb,yb,'r--',label = 'Test Error')
        s[0,j-1].legend((line1,line2),('Training Error','Test Error',),loc = 0)

        s[0,j-1].grid()
        if j == 1:
            s[0,j-1].set_ylabel('TrainingError & TestError',fontsize = 12,fontweight = 'bold')
    
        s[0,j-1].set_xlabel('Training Data(in percentage)')
        s[0,j-1].set_title(r'$\lambda$ = '+str(lk[j-1]))

        line3 = s[1,j-1].plot(xa,yc,'k--',label = 'R^2 Score')
        s[1,j-1].legend(loc = 0)
    
        s[1,j-1].grid()
        if j == 1:
            s[1,j-1].set_ylabel('R^2 Score',fontsize = 12,fontweight = 'bold')
    
        s[1,j-1].set_xlabel('# of training examples')
    

    if x == 'D' and y == 4:
        fig.suptitle('Own l_2-regularization(Diabetes)', fontsize=14, fontweight='bold')
    elif x == 'B' and y == 4:
        fig.suptitle('Own l_2-regularization(Boston)', fontsize=14, fontweight='bold')
  
    
  
    plt.show(block = False)

    
  
    fig1, s1  = plt.subplots(2,4,sharey = 'row')
    for j in range(1,5):
        xa = [];xb = [];ya = [];yb = [];yc = [];yf = []
        np.random.shuffle(b)
        
        a = (b.shape[0]*l1[j-1])//100
        dataset_X_train = b[:a+1,:-1]
        dataset_X_test = b[a+1:,:-1]
        dataset_Y_train = b[:a+1,-1]
        dataset_Y_test = b[a+1:,-1]
    
        for i in range(0,11):
          
            B = np.linalg.pinv(np.matmul(dataset_X_train.transpose(),dataset_X_train)+l[i]*np.eye(dataset_X_train.shape[1]))
            
            
            w = np.matmul(B,np.matmul(dataset_X_train.transpose(),dataset_Y_train))
            le = dataset_X_train.shape[0]
            te = 0

            for p in range(1,6):
                
                s = ((p-1)*le)//5
                e = (p*le)//5
                d0 = np.delete(dataset_X_train,np.s_[s:e],axis = 0)
                d1 = dataset_X_train[s:e,:]
                k0 = np.delete(dataset_Y_train,np.s_[s:e])
                k1 = dataset_Y_train[s:e]
                B1 = np.linalg.pinv(np.matmul(d0.transpose(),d0)+l[i]*np.eye(d0.shape[1]))
                w1 = np.matmul(B1,np.matmul(d0.transpose(),k0))
                ytp = np.matmul(d1,w1)
                tee = mean_squared_error(k1, ytp)
                te = te + tee
           
            te = te/5
            
                
            
            ypredictedtrain = np.matmul(dataset_X_train,w)
            ypredictedtest = np.matmul(dataset_X_test,w)
            trainingerror = mean_squared_error(dataset_Y_train, ypredictedtrain)
            testerror = mean_squared_error(dataset_Y_test, ypredictedtest)
            r2score = 1  -  trainingerror/(np.var(dataset_Y_train))
            xa.append(i)
            xb.append(l[i])
            ya.append(trainingerror)
            yb.append(testerror)
            yc.append(r2score)
            yf.append(te)
            
        
     
    
        line1, = s1[0,j-1].plot(xb,ya,'b-',label='Training Error')
        line2, = s1[0,j-1].plot(xb,yb,'r-',label = 'Test Error')
        line3, = s1[0,j-1].plot(xb,yf,'r--',label = 'Validation Error')
        s1[0,j-1].legend((line1,line2,line3),('Training Error','Test Error','Validation Error'),loc = 0)
        

        s1[0,j-1].grid()
        if j == 1:
            s1[0,j-1].set_ylabel('TrainingError & TestError',fontsize = 12,fontweight = 'bold')
    
       
        s1[0,j-1].set_title('Training Data = '+str(l1[j-1])+'%')

        line3 = s1[1,j-1].plot(xb,yc,'k--',label = 'R^2 Score')
        s1[1,j-1].legend(loc = 0)
    
        s1[1,j-1].grid()
        if j == 1:
            s1[1,j-1].set_ylabel('R^2 Score',fontsize = 12,fontweight = 'bold')
    
        s1[1,j-1].set_xlabel(r'$\lambda$')

    if x == 'D' and y == 4:
        fig1.suptitle('Own RidgeCV(Diabetes)', fontsize=14, fontweight='bold')
    elif x == 'B' and y == 4:
        fig1.suptitle('Own RidgeCV(Boston)', fontsize=14, fontweight='bold')
  
   
  
    plt.show(block = False)

if y == 2 or y == 3:
    q23(x,y)


if y == 1:
    q1(x)

if y == 4:
    q4(x)



    
            
            
            
     
    
        
    

