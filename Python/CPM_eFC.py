import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from tqdm import tqdm


# fighting!! 

class CPM:
    
    def __init__(self,k,subjlist,method):
        self.kf = KFold(n_splits=k)
        self.method = method
        
        self.cv_list_train = []
        self.cv_list_test = []
        self.subjlist = subjlist

        for train, test in self.kf.split(subjlist):
            self.cv_list_train.append(train)
            self.cv_list_test.append(test)

    def get_pearsonr(self,beh,connectivity):
        return [stats.pearsonr(beh,im) for im in connectivity.T]


    def summarize(self,subject,mask,path,extension):
        """
        Summarize data
    
        * The reason why I summarize the connectivity using for loop is because the dimension of eFC is too big.
    
        """
        summary_list = []
        
        print(subject[0])
        for sub in subject: 
            scan_i = np.load(path+str(sub)+extension)
            summation = 0
            for j in mask:
                summation += scan_i[j]
            summary_list.append(summation)
            del scan_i
        return np.array(summary_list)
    

    def train(self,PATH,subid,dimension,divideby,pheno,threshold,extension=''):
        
        
        """ 
        dimension : Total dimension of the connectivity 
        subid : subject id
        PATH : file path
        extension : extension of the file
        divideby : connectivity are divided into subj by (dimension/divideby)
        
        """
        
        predict_pheno_p = np.zeros((1,len(pheno)))
        predict_pheno_n = np.zeros((1,len(pheno)))
        ratio_by_threshold_p = []
        ratio_by_threshold_n = []
        
        
        connectivity_dimension = int(dimension/divideby)
        
        
        for f in range(len(self.cv_list_train)):
            
            
            train_behavior = pheno[self.cv_list_train[f]]
            
            pearson_list = []
            
            for nth in tqdm(range(divideby)):#197995050
            
                scan_list = []
                for sub in (subid[self.cv_list_train[f]]): #4second/sub ==> 1200second
                    scan_i = np.load('/data03/moon_repo/efc_1200/efc_restricted_scan2/patient_'+str(sub)+'.npy')
                    scan_list.append(scan_i[nth*connectivity_dimension:(nth+1)*connectivity_dimension])
                    del scan_i
                scan_list = np.array(scan_list)
                train_behavior = np.array(train_behavior)
                
                pearson_list_nth = self.get_pearsonr(train_behavior,scan_list) 
                pearson_list.append(pearson_list_nth)
                print(len(pearson_list))
                del pearson_list_nth
                #print(len(pearson_list[0]))
                
        
           # train_connectivity = connectivity[self.cv_list_train[f]]
            
            #perason_list = np.array(pearson_list).reshape(1,-1)
            #test_connectivity = connectivity[self.cv_list_test[f]]
            test_behavior = pheno[self.cv_list_test[f]]   
            pearson_list = [y for x in pearson_list for y in x]
            
            print(len(pearson_list))
            print(pearson_list[0])
    
            pearson_filter_p = [j for j,i in enumerate(pearson_list) if i[1] < threshold and i[0]>0]
            pearson_filter_n = [j for j,i in enumerate(pearson_list) if i[1] < threshold and i[0]<0]
    
    
            ratio_by_threshold_p.append(len(pearson_filter_p)/len(pearson_list))
            ratio_by_threshold_n.append(len(pearson_filter_n)/len(pearson_list))

    
            summary_list_p = self.summarize(subid[self.cv_list_train[f]],pearson_filter_p,PATH,extension)
            summary_list_n = self.summarize(subid[self.cv_list_train[f]],pearson_filter_n,PATH,extension)
            
            regressor_p = self.train_cpm(summary_list_p,train_behavior)
            regressor_n = self.train_cpm(summary_list_n,train_behavior)
            
            
            summary_test_p = self.summarize(subid[self.cv_list_test[f]],pearson_filter_p,PATH,extension) 
            summary_test_n = self.summarize(subid[self.cv_list_test[f]],pearson_filter_n,PATH,extension) 
            
            cv_predict_p =regressor_p.predict(summary_test_p.reshape(-1,1))
            cv_predict_n =regressor_n.predict(summary_test_n.reshape(-1,1))
            
            
            predict_pheno_p[0,self.cv_list_test[f]] = cv_predict_p
            predict_pheno_n[0,self.cv_list_test[f]] = cv_predict_n
        

        return predict_pheno_p, predict_pheno_n,ratio_by_threshold_p,ratio_by_threshold_n


    
    
    def temp(self,PATH,subid,dimension,divideby,pheno,threshold,extension=''):
        
        
        """ 
        dimension : Total dimension of the connectivity 
        subid : subject id
        PATH : file path
        extension : extension of the file
        divideby : connectivity are divided into subj by (dimension/divideby)
        
        """
        
        predict_pheno_p = np.zeros((1,len(pheno)))
        predict_pheno_n = np.zeros((1,len(pheno)))
        ratio_by_threshold_p = []
        ratio_by_threshold_n = []
        
        
        connectivity_dimension = int(dimension/divideby)
        
        
        train_behavior = pheno[self.cv_list_train[0]]
            
        pearson_list = []
            
        for nth in tqdm(range(divideby)):#197995050
            
            scan_list = []
            for sub in (subid[self.cv_list_train[0]]): #4second/sub ==> 1200second
                scan_i = np.load('/data03/moon_repo/efc_1200/efc_restricted_scan2/patient_'+str(sub)+'.npy')
                scan_list.append(scan_i[nth*connectivity_dimension:(nth+1)*connectivity_dimension])
                del scan_i
            scan_list = np.array(scan_list)
            train_behavior = np.array(train_behavior)
                
            pearson_list_nth = self.get_pearsonr(train_behavior,scan_list) 
            pearson_list.append(pearson_list_nth)
            print(len(pearson_list))
            del pearson_list_nth
                #print(len(pearson_list[0]))
                
        
           # train_connectivity = connectivity[self.cv_list_train[f]]
            
            #perason_list = np.array(pearson_list).reshape(1,-1)
            #test_connectivity = connectivity[self.cv_list_test[f]]
        test_behavior = pheno[self.cv_list_test[0]]   
        pearson_list = [y for x in pearson_list for y in x]
            
        print(len(pearson_list))
    
        pearson_filter_p = [j for j,i in enumerate(pearson_list) if i[1] < threshold and i[0]>0]
        pearson_filter_n = [j for j,i in enumerate(pearson_list) if i[1] < threshold and i[0]<0]
    
    
        ratio_by_threshold_p.append(len(pearson_filter_p)/len(pearson_list))
        ratio_by_threshold_n.append(len(pearson_filter_n)/len(pearson_list))

    
        summary_list_p = self.summarize(subid[self.cv_list_train[0]],pearson_filter_p,PATH,extension)
        summary_list_n = self.summarize(subid[self.cv_list_train[0]],pearson_filter_n,PATH,extension)
            
        regressor_p = self.train_cpm(summary_list_p,train_behavior)
        regressor_n = self.train_cpm(summary_list_n,train_behavior)
            
            
        summary_test_p = self.summarize(subid[self.cv_list_test[0]],pearson_filter_p,PATH,extension) 
        summary_test_n = self.summarize(subid[self.cv_list_test[0]],pearson_filter_n,PATH,extension) 
            
        cv_predict_p =regressor_p.predict(summary_test_p.reshape(-1,1))
        cv_predict_n =regressor_n.predict(summary_test_n.reshape(-1,1))   
        
        return cv_predict_p,cv_predict_n,ratio_by_threshold_p,ratio_by_threshold_n
    
    
    
    def train_cpm(self,summary, pheno):
        alphas = 10**np.linspace(5,-6,100)*0.2 
        
        if self.method == "Ridge":
            regressor = RidgeCV(alphas=alphas)
            regressor.fit(summary.reshape(-1,1), pheno)
      
        elif self.method == "linear":
        
            regressor = LinearRegression()
            regressor.fit(summary.reshape(-1,1), pheno)
        elif self.method == "Lasso":
        
            regressor = LassoCV(alphas=aphas)
            regressor.fit(summary.reshape(-1,1), pheno)
        
        
        return regressor



