#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:48:04 2021

@author: raymondlei
"""
 
_version='20210515'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype
import pickle
import os
from sklearn.feature_selection import RFE
import sys




def drop_high_corr(df,col,threshold=0.95):
    num_f=[i for i in col if is_numeric_dtype(df[i])==True ]
    corr_matrix = df.loc[:,num_f].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = set([column for column in upper.columns if any(upper[column] > threshold)])

    print('dropped '+str(len(to_drop))+' highly correlated features out of '+str(len(col))+'!')

    return to_drop


class vaccine_classifier:
    
    def __init__ (self
                  ,file_name
                  ,submit
                  ,top_k
                  ,run_from_beginning 
                  ,n_features_to_select
                  ,show_n=20
                  ,repeat=5
                  ,nfold=5
                  ,concervative=True
                  ,drop_corr=True
                  ,root='/Users/raymondlei/Downloads/project/'
                  ,recursive_selection=True
                  ,debug=True
                  ,n_select=100
                      
                  ):
        
        self.root=root
        self.file_name=file_name
        self.submit=submit
        self.show_n=show_n
        self.repeat=repeat
        self.nfold=nfold
        self.concervative=concervative
        self.drop_corr=drop_corr
        self.top_k=top_k
        self.run_from_beginning=run_from_beginning
        self.cur_check_name=root+'checkpoint.p'
        self.new_check_name=root+'checkpoint_sucess.p'
        self.n_features_to_select=n_features_to_select #duplicate
        self.recursive_selection=recursive_selection
        self.debug=debug
        self.n_select=n_features_to_select 


    def checkpoint(self):
        
        with open(self.cur_check_name, 'wb') as f:
             pickle.dump(self, f)
        #double check to make sure saving sucessfully     
        os.rename(self.cur_check_name ,self.new_check_name)
        print('saved!')     
 
    
    def resume(self):
        self = pickle.load( open( self.new_check_name, "rb" ) ) #?
        print('resumed from {}'.format(self.step_log[self.log_step]))
        
        return self 
        
    def execute(self):

        self.step_log={
         1:'self.load()'
        ,2:'self.create_multiclass()'
        ,3:'self.cast()'
        ,4:'self.split()'
        ,5:'self.create_combination()'
        ,6:'self.create_num_age()'
        ,7:'self.create_num_edu()'
        ,8:'self.create_num_income()'
        ,9:'self.create_log()'
        ,10:'self.create_index()'
        ,11:'self.create_agg()' 
        ,12:'self.drop_unused()'
        ,13:'self.feature_selection()'
        ,14:'self.feature_selection_recursive()'
        ,15:'self.model_train()'
        }        
        self.log_step=1    

        if self.run_from_beginning==False and os.path.exists(self.new_check_name):
             self=self.resume()
             while self.log_step<=max(self.step_log.keys()):
                 if self.recursive_selection==False and self.log_step==14:
                     self.log_step+=1
                 else:
                     exec(self.step_log[self.log_step])
                     self.log_step+=1
                     self.checkpoint()
        else:
            if os.path.exists(self.new_check_name):
                os.remove(self.new_check_name)
            for k, v in self.step_log.items():
                print(14)
                if self.recursive_selection==False and k==14:
                     self.log_step+=1
                else:     
                    exec(v)
                    if k<15:
                        self.log_step+=1
                        self.checkpoint()
                    else:
                        self.checkpoint()
                        print("This is the end of process!")
        
 
        
    def load (self):
        if self.debug:
            self.train_x=pd.read_csv(self.root+'train_x.csv',index_col='respondent_id').sample(frac=0.1, random_state=1)
            self.test_x=pd.read_csv(self.root+'test_x.csv').sample(frac=0.1, random_state=1)
        else:
            self.train_x=pd.read_csv(self.root+'train_x.csv',index_col='respondent_id')
            self.test_x=pd.read_csv(self.root+'test_x.csv')
        self.train_y=pd.read_csv(self.root+'train_y.csv',index_col='respondent_id')
        self.train=pd.merge(left=self.train_x, right=self.train_y,how='inner',left_index=True,right_index=True)
        self.submission=pd.read_csv(self.root+'submission.csv')
        self.binary_col= [
            'behavioral_antiviral_meds'  
            ,'behavioral_avoidance'  
            ,'behavioral_face_mask'  
            ,'behavioral_wash_hands'  
            ,'behavioral_large_gatherings'  
            ,'behavioral_outside_home'  
            ,'behavioral_touch_face'  
            ,'doctor_recc_h1n1'  
            ,'doctor_recc_seasonal'  
            ,'chronic_med_condition'  
            ,'child_under_6_months'  
            ,'health_worker'  
            ,'health_insurance'  
            ]
        self.num_col=[   
            'h1n1_concern'  
            ,'h1n1_knowledge'  
            ,'opinion_h1n1_vacc_effective'  
            ,'opinion_h1n1_risk'  
            ,'opinion_h1n1_sick_from_vacc'  
            ,'opinion_seas_vacc_effective'  
            ,'opinion_seas_risk'  
            ,'opinion_seas_sick_from_vacc'    
            ,'age_group_t'
            ,'education_t'
            ,'income_poverty_t'
            ]
        self.cat_col=[
            'age_group'
            ,'education'
            ,'race'
            ,'income_poverty'
            ,'marital_status'
            ,'rent_or_own'
            ,'employment_status'
            ,'hhs_geo_region'
            ,'census_msa'
            ,'employment_industry'
            ,'employment_occupation'
            ,'sex'

            ]
        
        self.order_col=[
            
            'h1n1_concern'
            ,'h1n1_knowledge'
            ,'opinion_h1n1_vacc_effective'
            ,'opinion_h1n1_risk'
            ,'opinion_h1n1_sick_from_vacc'
            ,'opinion_seas_vacc_effective'
            ,'opinion_seas_risk'
            ,'opinion_seas_sick_from_vacc' 
            
            ]
        
        self.raw_target=[
            
            'h1n1_vaccine'
            ,'seasonal_vaccine'
            
            ]

        #target ratios (assuming no duplicates)
        print('h1n1_vaccine_target_ratios: ' + str(self.train_y.h1n1_vaccine.sum()/len(self.train_y)))
        print('seasonal_Vaccine_target_ratios: ' + str(self.train_y.seasonal_vaccine.sum()/len(self.train_y)))
        print('number of features: {}'.format(str(len(self.train_x.columns))))

        return self

    def create_multiclass(self,drop_orig=False):  
    
        label1= self.raw_target[0]
        label2= self.raw_target[1]
    
        def f(row):
            v=''
            if row[label1]==1 and row[label2]==1:
                v='HS'
            elif row[label1]==1 and row[label2]==0:
                v='H'
            elif row[label1]==0 and row[label2]==1:
                v='S'
            else:
                v='N'
        
            return v
    
        self.train['new_target']=self.train.apply(f,axis=1)
        
        if drop_orig:
            self.train.drop([label1,label2],axis=1,inplace=True)
            self.new_target=['new_target']
        else:
            self.new_target=[label1,label2,'new_target']
        
        return self
        
    def cast(self):
        for i in self.train.select_dtypes(exclude=['float64']).columns:
            if i not in ['seasonal_vaccine','h1n1_vaccine']:
                self.train[i]=self.train[i].astype('category')
            if i not in ['seasonal_vaccine','h1n1_vaccine','new_target']:
                self.test_x[i]=self.test_x[i].astype('category')
    
        return self
    
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.train.drop(self.new_target,axis=1),self.train.loc[:,self.new_target],test_size=0.2,random_state=100)
        
        return self


    def create_combination(self):
    
        t=self.X_train.copy()
        v=self.X_test.copy()
        te=self.test_x.copy()
        columns=self.binary_col
    
        toolbar_width = 7
        sys.stdout.write("[%s]" % ("Create combination"))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))  

        columns_a=columns.copy()
        for a in columns:
            for b in columns_a:
                if a!=b:
                    t[a+'_or_'+b]=t[a]+t[b]
                    v[a+'_or_'+b]=v[a]+v[b]
                    te[a+'_or_'+b]=te[a]+te[b]
                    
                    t[a+'_or_'+b+'_comb']=t[a+'_or_'+b].map(lambda x: 1 if x>1 else x)
                    v[a+'_or_'+b+'_comb']=v[a+'_or_'+b].map(lambda x: 1 if x>1 else x)
                    te[a+'_or_'+b+'_comb']=te[a+'_or_'+b].map(lambda x: 1 if x>1 else x)
                    
                else:   
                    columns_a.remove(b)
    
        sys.stdout.write("-")
        sys.stdout.flush()
    
        columns_b1=columns.copy()
        columns_b2=columns.copy()
        for a in columns:
            for b in columns_b1:
                if a!=b:
                    for c in columns_b2:
                        if b!=c:
                            t[a+'_or_'+b+'_or_'+c]=t[a]+t[b]+t[c]
                            v[a+'_or_'+b+'_or_'+c]=v[a]+v[b]+v[c]
                            te[a+'_or_'+b+'_or_'+c]=te[a]+te[b]+te[c]
                
                            t[a+'_or_'+b+'_or_'+c+'_comb'] = t[a+'_or_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_or_'+b+'_or_'+c+'_comb'] = v[a+'_or_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_or_'+b+'_or_'+c+'_comb']=te[a+'_or_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:
                            columns_b2.remove(c)
                else:
                    columns_b1.remove(b)

        sys.stdout.write("-")
        sys.stdout.flush()

        columns_c1=columns.copy()
        columns_c2=columns.copy()
        for a in columns:
            for b in columns_c1:
                if a!=b:
                    for c in columns_c2:
                        if b!=c:
                            t[a+'_and_'+b+'_or_'+c]=t[a]*t[b]+t[c]
                            v[a+'_and_'+b+'_or_'+c]=v[a]*v[b]+v[c]
                            te[a+'_and_'+b+'_or_'+c]=te[a]*te[b]+te[c]
                            
                            t[a+'_and_'+b+'_or_'+c+'_comb'] = t[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_and_'+b+'_or_'+c+'_comb'] = v[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_and_'+b+'_or_'+c+'_comb']=te[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:
                            columns_c2.remove(c)
                else:
                    columns_c1.remove(b)   
                
     
        sys.stdout.write("-")
        sys.stdout.flush()          
                    
        columns_d1=columns.copy()
        columns_d2=columns.copy()
        for a in columns:
            for b in columns_d1:
                if a!=b:
                    for c in columns_d2:
                        if b!=c:
                            t[a+'_or_not'+b+'_or_'+c]=t[a]+(1-t[b])+t[c]
                            v[a+'_or_not'+b+'_or_'+c]=v[a]+(1-v[b])+v[c]
                            te[a+'_or_not'+b+'_or_'+c]=te[a]+(1-te[b])+te[c]
                
                            t[a+'_or_not'+b+'_or_'+c+'_comb'] = t[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_or_not'+b+'_or_'+c+'_comb'] = v[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_or_not'+b+'_or_'+c+'_comb']=te[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:
                            columns_d2.remove(c)
                else:
                    columns_d1.remove(b)      
                    
    
        sys.stdout.write("-")
        sys.stdout.flush()
    
    
        columns_e1=columns.copy()
        columns_e2=columns.copy()
        for a in columns:
            for b in columns_e1:
                if a!=b:
                    for c in columns_e2:
                        if b!=c:
                            t[a+'_or_not'+b+'_or_not'+c]=t[a]+(1-t[b])+(1-t[c])
                            v[a+'_or_not'+b+'_or_not'+c]=v[a]+(1-v[b])+(1-v[c])
                            te[a+'_or_not'+b+'_or_not'+c]=te[a]+(1-te[b])+(1-te[c])
                            
                            t[a+'_or_not'+b+'_or_not'+c+'_comb'] = t[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_or_not'+b+'_or_not'+c+'_comb'] = v[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_or_not'+b+'_or_not'+c+'_comb']=te[a+'_or_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:    
                            columns_e2.remove(c)
                else:
                    columns_e1.remove(b)   

        sys.stdout.write("-")
        sys.stdout.flush()


        columns_f1=columns.copy()
        columns_f2=columns.copy()
        for a in columns:
            for b in columns_f1:
                if a!=b:
                    for c in columns_f2:
                        if b!=c:
                            t[a+'_and_not'+b+'_or_'+c]=t[a]*(1-t[b])+t[c]
                            v[a+'_and_not'+b+'_or_'+c]=v[a]*(1-v[b])+v[c]
                            te[a+'_and_not'+b+'_or_'+c]=te[a]*(1-te[b])+te[c]
                            
                            t[a+'_and_not'+b+'_or_'+c+'_comb'] = t[a+'_and_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_and_not'+b+'_or_'+c+'_comb'] = v[a+'_and_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_and_not'+b+'_or_'+c+'_comb']=te[a+'_and_not'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:    
                            columns_f2.remove(c)
                else:
                    columns_f1.remove(b)   
     
        sys.stdout.write("-")
        sys.stdout.flush()             
    
    
        columns_g1=columns.copy()
        columns_g2=columns.copy()
        for a in columns:
            for b in columns_g1:
                if a!=b:
                    for c in columns_g2:
                        if b!=c :
                            t[a+'_and_'+b+'_or_not'+c]=t[a]*t[b]+(1-t[c])
                            v[a+'_and_'+b+'_or_not'+c]=v[a]*v[b]+(1-v[c])
                            te[a+'_and_'+b+'_or_not'+c]=te[a]*te[b]+(1-te[c])
                
                            t[a+'_and_'+b+'_or_not'+c+'_comb'] = t[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            v[a+'_and_'+b+'_or_not'+c+'_comb'] = v[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                            te[a+'_and_'+b+'_or_not'+c+'_comb']=te[a+'_and_'+b+'_or_'+c].map(lambda x: 1 if x>1 else x)
                        else:
                            columns_g2.remove(c)
                else:
                    columns_g1.remove(b)   
                    
        sys.stdout.write("-")
        sys.stdout.flush()


        self.X_train=t.copy()
        self.X_test=v.copy()
        self.test_x=te.copy()
        
        self.comb_col=[i for i in self.X_train.columns if '_comb' in i ]
        
        sys.stdout.write("Done! Number of features: " + str(len(self.X_train.columns))+ "  ")  
    
        
        return self

    def create_num_age(self):
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        int_train['age_group_t']=int_train['age_group'].map(lambda x: 1 if x=='18 - 34 Years' else (2 if x=='35 - 44 Years' else (3 if x=='45 - 54 Years' else (4 if x=='55 - 64 Years' else (5 if '65+ Years' else -1))))) 
        ext_val['age_group_t']=ext_val['age_group'].map(lambda x: 1 if x=='18 - 34 Years' else (2 if x=='35 - 44 Years' else (3 if x=='45 - 54 Years' else (4 if x=='55 - 64 Years' else (5 if '65+ Years' else -1))))) 
        test['age_group_t']=test['age_group'].map(lambda x: 1 if x=='18 - 34 Years' else (2 if x=='35 - 44 Years' else (3 if x=='45 - 54 Years' else (4 if x=='55 - 64 Years' else (5 if '65+ Years' else -1))))) 
        
        int_train['age_group_t']=int_train['age_group_t'].map(lambda x: float(x)).astype('float64')    
        ext_val['age_group_t']=ext_val['age_group_t'].map(lambda x: float(x)).astype('float64')     
        test['age_group_t']=test['age_group_t'].map(lambda x: float(x)).astype('float64')    
 
        self.X_train=int_train.copy()
        self.X_test=ext_val.copy()
        self.test_x=test.copy()
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
           
        return self
 
    def create_num_edu(self):
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        int_train['education_t']=int_train['education'].map(lambda x: 1 if x=='< 12 Years' else (2 if x=='12 Years' else (3 if x=='Some College' else (4 if x=='College Graduate' else -1)))) 
        ext_val['education_t']=ext_val['education'].map(lambda x: 1 if x=='< 12 Years' else (2 if x=='12 Years' else (3 if x=='Some College' else (4 if x=='College Graduate' else -1)))) 
        test['education_t']=test['education'].map(lambda x: 1 if x=='< 12 Years' else (2 if x=='12 Years' else (3 if x=='Some College' else (4 if x=='College Graduate' else -1)))) 
    
        int_train['education_t']=int_train['education_t'].map(lambda x: float(x)).astype('float64')     
        ext_val['education_t']=ext_val['education_t'].map(lambda x: float(x)).astype('float64')    
        test['education_t']=test['education_t'].map(lambda x: float(x)).astype('float64')    
    
        self.X_train=int_train.copy()
        self.X_test=ext_val.copy()
        self.test_x=test.copy()
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))

        return self
    
    def create_num_income(self):
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        int_train['income_poverty_t']=int_train['income_poverty'].map(lambda x: 1 if x=='Below Poverty' else (2 if x=='<= $75,000, Above Poverty' else (3 if x=='> $75,000' else -1))) 
        ext_val['income_poverty_t']=ext_val['income_poverty'].map(lambda x: 1 if x=='Below Poverty' else (2 if x=='<= $75,000, Above Poverty' else (3 if x=='> $75,000' else -1))) 
        test['income_poverty_t']=test['income_poverty'].map(lambda x: 1 if x=='Below Poverty' else (2 if x=='<= $75,000, Above Poverty' else (3 if x=='> $75,000' else -1))) 
  
        int_train['income_poverty_t']=int_train['income_poverty_t'].map(lambda x: float(x)).astype('float64')     
        ext_val['income_poverty_t']=ext_val['income_poverty_t'].map(lambda x: float(x)).astype('float64')    
        test['income_poverty_t']=test['income_poverty_t'].map(lambda x: float(x)).astype('float64')      
  
        self.X_train=int_train.copy()
        self.X_test=ext_val.copy()
        self.test_x=test.copy()
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))

        return self

    def create_log(self):
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        for col in self.num_col:
            int_train[col+'_log_t']=np.log(int_train[col]+2) 
            ext_val[col+'_log_t']  =np.log(ext_val[col]+2) 
            test[col+'_log_t']     =np.log(test[col]+2) 
  
        self.X_train=int_train.copy()
        self.X_test=ext_val.copy()
        self.test_x=test.copy()
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))

        return self
    
    def create_index(self,targets=['new_target','h1n1_vaccine','seasonal_vaccine'],drop_orig=False,keep_unweighted=False):
        
        columns=self.cat_col    
        train=pd.merge(left=self.X_train,right=self.y_train,how='inner',left_index=True,right_index=True)
        drop_corr=self.drop_corr
        
        int_train=pd.get_dummies(train.copy(),columns= list(set(targets)-set(['h1n1_vaccine','seasonal_vaccine'])),prefix='',prefix_sep='')
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        new_name=[]
        for target in set(train.copy()[targets[0]]):
            int_train['no_'+target] = int_train[target].map(lambda x: 1-x)
            
            toolbar_width = len(columns)
            sys.stdout.write("[%s]" % ("Create lvl1 index for target "+target))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1))  
            
            for col in columns:
                grp=pd.merge( 
                        left=int_train.loc[:,[col,target]].groupby([col]).agg({col:'count',target:'sum'}).rename(columns={col:col+'_cnt'}) 
                      ,right=int_train.loc[:,[col,'no_'+target]].groupby([col]).agg({col:'count','no_'+target:'sum'}).rename(columns={col:col+'_cnt_no'})
                      ,how='inner'
                      ,left_index=True
                      ,right_index=True
                      ).reset_index()

                keep_col=[col]
                if keep_unweighted:
                    grp[col+'_'+target+'_ratios']=grp[target]/grp['no_'+target]
                    keep_col.append(col+'_'+target+'_ratios')
            
                #weighted and rescale to a more readable rank ordering
                name=col+'_'+target+'_ratios_weighted'
                new_name.append(name)
                grp[name]=(grp[target] / grp[col+'_cnt']) / (grp['no_'+target] / grp[col+'_cnt_no'])
                m=max(grp[name])
                grp[name]=grp[name].map(lambda x: round((x/m)*100))
                keep_col.append(name)
                
                int_train=pd.merge(left=int_train,right=grp.loc[:,keep_col],how='left',on=col)
                ext_val  =pd.merge(left=ext_val  ,right=grp.loc[:,keep_col],how='left',on=col)
                test     =pd.merge(left=test     ,right=grp.loc[:,keep_col],how='left',on=col)
                
                sys.stdout.write("-")
                sys.stdout.flush()     
            
            sys.stdout.write("Done! Number of features: " + str(len(int_train.columns)))  
            
            
            columns_a=columns.copy()
            columns_b=columns.copy() + self.binary_col 
            
            toolbar_width = len(columns_a) ###
            sys.stdout.write("[%s]" % ("Create lvl2 index for target: "+target))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1))  
        
            for col1 in columns_a:
                for col2 in columns_b  :
                    if col1 != col2 and int_train[col1].isna().sum()==0 and int_train[col2].isna().sum()==0:
                        grp=pd.merge( 
                            left=int_train.loc[:,[col1,col2,target]].groupby([col1,col2]).agg({col1:'count',target:'sum'}).rename(columns={col1:col1+'_cnt'}) 
                            ,right=int_train.loc[:,[col1,col2,'no_'+target]].groupby([col1,col2]).agg({col1:'count','no_'+target:'sum'}).rename(columns={col1:col1+'_cnt_no'})
                            ,how='inner'
                            ,left_index=True
                            ,right_index=True
                            ).reset_index()     
            
                        keep_col=[col1,col2]
                        if keep_unweighted:
                            grp[col1+'_'+col2+'_'+target+'_ratios']=grp[target]/grp['no_'+target]
                            keep_col.append(col1+'_'+col2+'_'+target+'_ratios')
            
                        #weighted and rescale to a more readable rank ordering
                        name2=col1+'_'+col2+'_'+target+'_ratios_weighted'
                        new_name.append(name2)
                
                        grp[name2]=(grp[target] / grp[col1+'_cnt']) / (grp['no_'+target] / grp[col1+'_cnt_no'])
                        m=max(grp[name2])
                        grp[name2]=grp[name2].map(lambda x: round((x/m)*100))
                        keep_col.append(name2)
                    
                        int_train=pd.merge(left=int_train,right=grp.loc[:,keep_col],how='left',on=[col1,col2])
                        ext_val  =pd.merge(left=ext_val  ,right=grp.loc[:,keep_col],how='left',on=[col1,col2])
                        test     =pd.merge(left=test     ,right=grp.loc[:,keep_col],how='left',on=[col1,col2])
                
                    else:
                        columns_b.remove(col2)
                    
                    sys.stdout.write("-")
                    sys.stdout.flush()    
                
                sys.stdout.write("-")
                sys.stdout.flush()    
                    
            sys.stdout.write("]\n")
            sys.stdout.flush()
            sys.stdout.write("Done! Number of features: " + str(len(int_train.columns)))  
                    

        int_train.drop(['no_'+target,target],axis=1,inplace=True)
        
        if drop_orig:
            int_train.drop(columns=columns,axis=1,inplace=True)
            ext_val.drop(columns=columns,axis=1,inplace=True)
            test.drop(columns=columns,axis=1,inplace=True)
 
        if drop_corr:
            to_drop=drop_high_corr(int_train,new_name,threshold=0.99)
            int_train.drop(columns=to_drop,axis=1,inplace=True)
            ext_val.drop(columns=to_drop,axis=1,inplace=True)
            test.drop(columns=to_drop,axis=1,inplace=True)
        
        self.X_train=int_train.copy()
        self.X_test=ext_val.copy()
        self.test_x=test.copy()
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
        
        self.index_col=[i for i in self.X_train.columns if '_ratios_weighted' in i]

        return self

 

    def create_agg(self):
        
        cat_col=self.cat_col
        num_col=[   self.num_col 
                # , ['age_group_t','education_t','income_poverty_t'] 
                # , self.comb_col[:50]
                # , self.comb_col[50:100] 
                # , self.comb_col[100:150] 
                # , self.comb_col[150:200]
                # , self.comb_col[200:250]
                # , self.comb_col[250:300]
                # , self.comb_col[300:350]
                # , self.comb_col[350:400]
                # , self.comb_col[450:500] 
                # , self.index_col[:round(len(self.index_col)/2)] 
                # , self.index_col[round(len(self.index_col)/2):]  
                 ] 
        drop_corr=self.drop_corr
            
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        def agg_burst(data,cat_col,num_col,return_feature_name=True,exclude_outlier=True):
            df=data.copy()
            for cat in cat_col.copy():
                agg=[   
                    
                    df.loc[:,[cat]+num_col].groupby([cat]).mean().rename(columns=  dict(zip(num_col,[cat+'_'+i+'_mean' for i in num_col]))).reset_index()
                 ,df.loc[:,[cat]+num_col].groupby([cat]).median().rename(columns=dict(zip(num_col,[cat+'_'+i+'_med' for i in num_col]))).reset_index()
                 ,df.loc[:,[cat]+num_col].groupby([cat]).min().rename(columns=   dict(zip(num_col,[cat+'_'+i+'_min' for i in num_col]))).reset_index()
                 ,df.loc[:,[cat]+num_col].groupby([cat]).max().rename(columns=   dict(zip(num_col,[cat+'_'+i+'_max' for i in num_col]))).reset_index()
            
                ]
                for a in agg:
                    df=pd.merge(left=df,right=a,how='left',on=cat)  


            cat_col2=cat_col.copy()
            for cat1 in cat_col2:
                for cat2 in cat_col2:
                    if cat1!=cat2:                    
                        agg2=[   

                            df.loc[:,[cat1,cat2]+num_col].groupby([cat1,cat2]).mean().rename(columns=   dict(zip(num_col,[cat1+'_'+cat2+'_'+i+'_mean'for i in num_col]))).reset_index()
                            ,df.loc[:,[cat1,cat2]+num_col].groupby([cat1,cat2]).median().rename(columns=dict(zip(num_col,[cat1+'_'+cat2+'_'+i+'_med' for i in num_col]))).reset_index()
                            ,df.loc[:,[cat1,cat2]+num_col].groupby([cat1,cat2]).min().rename(columns=   dict(zip(num_col,[cat1+'_'+cat2+'_'+i+'_min'for i in num_col]))).reset_index()
                            ,df.loc[:,[cat1,cat2]+num_col].groupby([cat1,cat2]).max().rename(columns=   dict(zip(num_col,[cat1+'_'+cat2+'_'+i+'_max' for i in num_col]))).reset_index()
                        
                        ]
                
                        for a2 in agg2:
                            df=pd.merge(left=df,right=a2,how='left',on=[cat1,cat2])  
                    
                    else:   
                        cat_col2.remove(cat2)
                    
            if exclude_outlier:
                n=0
                for cat in cat_col.copy():
                    for num in num_col.copy():    
                        dist=pd.merge(left=df.loc[:,[cat,num]],right= int_train.loc[:,[cat,num]].groupby(cat).agg({num:'mean'}).rename(columns={num:num+'_mean'}).reset_index(),how='left',on=cat)
                        dist=pd.merge(left=dist,right= int_train.loc[:,[cat,num]].groupby(cat).agg({num:'std'}).rename(columns={num:num+'_std'}).reset_index(),how='left',on=cat)                  
                        cond=[(dist[num]<dist[num+'_mean']-1.5*dist[num+'_std']) , (dist[num]>dist[num+'_mean']+1.5*dist[num+'_std']) ]
                        val=[1,1]
                        dist[num+'_outlier']=np.select(cond,val)
                    
                        dist2=dist.loc[dist[num+'_outlier']==0,[cat,num]]
                        dist2=pd.merge(left=dist2,right= dist2.loc[:,[cat,num]].groupby(cat).agg({num:'max'}).rename(columns={num:num+'_max'}).reset_index(),how='left',on=cat)
                        dist2=pd.merge(left=dist2,right= dist2.loc[:,[cat,num]].groupby(cat).agg({num:'min'}).rename(columns={num:num+'_min'}).reset_index(),how='left',on=cat)
                        dist2[num+'_normalized']=(dist2[num]-dist2[num+'_min']) / (dist2[num+'_max']-dist2[num+'_min'])

                        dist3=dist2.loc[:,[cat,num+'_normalized']]
            
                        agg3=[   

                            dist3.groupby([cat]).mean().rename(columns= { num+'_normalized': cat+'_'+num+'_normalized_mean'}).reset_index()
                            ,dist3.groupby([cat]).median().rename(columns={ num+'_normalized': cat+'_'+num+'_normalized_med'}).reset_index()
                            ,dist3.groupby([cat]).min().rename(columns= { num+'_normalized': cat+'_'+num+'_normalized_min'}).reset_index()
                            ,dist3.groupby([cat]).max().rename(columns= { num+'_normalized': cat+'_'+num+'_normalized_max'}).reset_index()
                
                        ]
                    
                    
                        for a3 in agg3:
                            df=pd.merge(left=df,right=a3,how='left',on=cat)  
                    
                        n+=1    
                        print('created agg '+str(n)+'/'+str(len(cat_col)*len(num_col))+'!')
                        
                    
            if return_feature_name:
                return df,[i for i in df.columns if '_mean' in i or '_max' in i or '_min' in i or '_std' in i or '_sem' in i or '_var' in i or '_med' in i]
        
            else:
                return df
                
                
        for num in num_col:        
            int_train,new_name =agg_burst(int_train,cat_col,num_col=num,return_feature_name=True)
            ext_val  =agg_burst(ext_val,cat_col,num_col=num,return_feature_name=False)
            test     =agg_burst(test,cat_col,num_col=num,return_feature_name=False)
            union=list(set(int_train.columns) & set(ext_val.columns) & set(test.columns))
    
            if drop_corr:
                to_drop=drop_high_corr(int_train,new_name,threshold=0.99)
                fnl_to_drop=set(set(union) & set(to_drop))
    
                self.X_train=int_train.loc[:,union].drop(fnl_to_drop,axis=1,inplace=False)
                self.X_test=ext_val.loc[:,union].drop(fnl_to_drop,axis=1,inplace=False)
                self.test_x=test.loc[:,['respondent_id']+union].drop(fnl_to_drop,axis=1,inplace=False)
    
            else:
                self.X_train=int_train.loc[:,union]
                self.X_test=ext_val.loc[:,union]
                self.test_x=test.loc[:,['respondent_id']+union]
            print('number of features: {}'.format(str(len(self.X_train.columns))))
        
        print('aggregation is created!')
        

        return self



    def drop_unused(self,drop_col=['age_group','education','income_poverty']):
        if any(i in drop_col for i in self.X_train.columns):
            self.X_train.drop(drop_col,axis=1,inplace=True)
            self.X_test.drop(drop_col,axis=1,inplace=True)
            self.test_x.drop(drop_col,axis=1,inplace=True)
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
        
        return self


    def feature_selection(self):
    
        X=self.X_train.copy()
        y=self.y_train.copy()
        X_test=self.X_test.copy()
        test_x=self.test_x.copy()
        new_target=self.new_target
        nfold=self.nfold
        concervative=self.concervative
        drop_corr=self.drop_corr
        top_k=self.top_k
    
        #insert random variables
        def insert_random(X, N=10, seed=1, prefix='rand'):
            x=X.copy()
            n=1
            rand_name=[]
            while n<=N:    
                np.random.seed(seed*n)
                name='rand'+str(n)
                x[name]=np.random.uniform(0,1,len(x))
                rand_name.append(name)
                n+=1
        
            return x, N, [1.0] * N,rand_name
    
        def rand_imp(f_log,N,concervative):
            imp=[]
            for i in range(1,N+1):
                r_imp=f_log.loc[f_log.feature=='rand'+str(i)].reset_index().feature_importance[0]    
                imp.append(max(0,r_imp))
                
            if concervative:
                cutoff=sorted(imp,reverse=True)[1]
            else:
                cutoff=max(imp)
                        
            return cutoff
                
        X_train, n_rand, rand_factor,rand_name =insert_random(X.copy())
        y_train=y.copy().reset_index().drop(columns='respondent_id',axis=1)
    
        sc=[]
        k=1
    
        #cross validation
        folds=RepeatedKFold(n_splits=nfold,n_repeats=5,random_state=100)
        for n, (t,v) in enumerate(folds.split(X_train.values,y_train.values)):
             print("fold n°{}".format(n))
     
             try:
                    
                 gbm_fs=lgb.LGBMClassifier(
                        n_estimators=500 
                        ,objective='multiclass'
                        ,metric=['auc_mu']
                        ,early_stopping_rounds=50
                        ,boosting_type= 'gbdt'
                        ,learning_rates=0.01
                        ,num_leaves= 7
                        ,bagging_fraction= 0.5
                        ,bagging_freq= 1
                        ,feature_fraction= 1
                        ,min_data_in_bin=500
                        ,min_data_in_leaf= 50
                        ,cat_smooth=50
                        ,max_cat_to_onehot=4
                        ,extra_trees=True
                        #,cegb_penalty_feature_coupled=[ 1.0 for i in set(X_train.columns)]  
                        #,cegb_tradeoff=5 + np.log2(k+1)
                        ,is_unbalance=True
                        ,class_weight={'S':0.8,'H':1.2,'HS':1.1,'N':0.9}
                        ,feature_penalty=[0.80 if 'age_group' in i and 'census_msa' in i else (0.90 if 'census_msa' in i else (0.90 if 'age_group' in i else 1.0)) for i in set(list(X_train.columns)) ]
                        ).fit(
                               X_train.loc[t,list(set(X_train.columns))]
                              ,y_train.loc[t,'new_target']
                              ,eval_set=[
                                   (X_train.loc[v,list(set(X_train.columns))]
                                  ,y_train.loc[v,'new_target'])
                                  
                                  ]
                              ,eval_names=['int val']
                              ,verbose=False
                              )
       
                 feature_log=pd.DataFrame({'feature':gbm_fs.feature_name_,'feature_importance':gbm_fs.feature_importances_})
                 cutoff=rand_imp(feature_log,n_rand,concervative)
            
                 sc=sc.copy() + list(set(feature_log.loc[feature_log.feature_importance>cutoff]['feature'].tolist())-set(rand_name))        
                 print(len(sc))
                
             except:
                 print('Error occurs during this split; this split is skipped!')
             
             k+=1    
    
             print('feature selection completion '+str(round(k*100/nfold))+'% !')
            
    
        #vote    
        var=[]
        cnt=[]
        for i in set(sc):
            var.append(i)
            cnt.append(sc.count(i))
        
        if top_k> len(var):
            top_k=len(var)
            print('The number of features intend to select has existed the possible limit. Overriden by total number of variable!')
        
        sf=pd.DataFrame.from_dict({'var':var,'cnt':cnt}).sort_values(by='cnt',ascending=False).head(top_k)['var'].tolist()
        sf_report=pd.DataFrame.from_dict({'var':var,'cnt':cnt}).sort_values(by='cnt',ascending=False)
    
        if drop_corr:
            to_drop=drop_high_corr(X_train,sf,threshold=0.99)
            X_train.drop(columns=to_drop,axis=1,inplace=True)
            X_test.drop (columns=to_drop,axis=1,inplace=True)
            test_x.drop (columns=to_drop,axis=1,inplace=True)
    
            print(str(len(to_drop))+' features are dropped due to high correlation!')
        else:
            to_drop=[]
        
        fnl_feature=set(sf)-set(to_drop)
        print(str(len(fnl_feature)))

        self.fnl_feature=fnl_feature
        self.sf_report=sf_report

 
        return self


    def feature_selection_recursive(self):
        candidate= self.fnl_feature 
        log={}
        rm_log={}
        step=1
        log[0]=0
        rm_log[0]=[]

        def get_key(d,val):
            for k, v in d.items():
                if val == v:
                    return k

        while len(candidate)-len(rm_log) > self.n_select:
            estimator = lgb.LGBMClassifier(
             n_estimators=500 
            ,objective='multiclass'
            ,metric=['auc_mu']
            #,early_stopping_rounds=50
            ,boosting_type= 'gbdt'
            ,learning_rates=0.01
            ,num_leaves= 7
            ,bagging_fraction= 0.5
            ,bagging_freq= 1
            ,feature_fraction= 1
            ,min_data_in_bin=500
            ,min_data_in_leaf= 50
            ,cat_smooth=50
            ,max_cat_to_onehot=2
            ,extra_trees=True
            ,is_unbalance=True
            ,class_weight={'S':0.8,'H':1.2,'HS':1.1,'N':0.9}
            #,feature_penalty=[0.80 if 'age_group' in i and 'census_msa' in i else (0.90 if 'census_msa' in i else (0.90 if 'age_group' in i else 1.0)) for i in candidate]
       #     ,eval_set=[(self.X_test.loc[:,candidate],self.y_test.loc[:,self.new_target])]
            ).fit( self.X_train.loc[:,candidate]
              ,self.y_train.loc[:,'new_target']
              ,eval_set=[
                          (self.X_test.loc[:,candidate]
                          ,self.y_test.loc[:,'new_target'])
                                  
                           ]
              ,eval_names=['ext val']
                          ,verbose=False
                              )
       
            print(123)      
            feature_log=pd.DataFrame({'feature':estimator.feature_name_,'feature_importance':estimator.feature_importances_}).sort_values(ascending=False,by='feature_importance')
            print(234)
            lst_imp=feature_log.iloc[-1,0]
            print(345)
 
            v_pred=estimator.predict_proba(self.X_test.loc[:,candidate],raw_score=False)    
            print(456)
            vd=self.X_test.loc[:,candidate].copy()
            print(567)
            for i in range(0,len(v_pred)):
                vd.loc[i,'h1n1_vaccine_pred']    =v_pred[i,0]+v_pred[i,1]
                vd.loc[i,'seasonal_vaccine_pred']=v_pred[i,1]+v_pred[i,3]
            print(567)
            log[step]=(roc_auc_score(self.y_test.loc[:,'h1n1_vaccine'], vd.loc[:,'h1n1_vaccine_pred']) + roc_auc_score(self.y_test.loc[:,'seasonal_vaccine'], vd.loc[:,'seasonal_vaccine_pred'] ))/2
            print(678)
            if len(log.keys())<5:
                rm_log[step]=rm_log[step-1]+[lst_imp]
                print(789)
            elif log[step]==min(log[step],log[step-1],log[step-2],log[step-3],log[step-4]):
                rm_log[step]=rm_log[step-1]+[lst_imp]
                candidate.remove(lst_imp)
                print(8910)
            else:
                print(91011)
                fnl_step=get_key(log,min(log[step],log[step-1],log[step-2],log[step-3],log[step-4]))
                break
            
            step+=1
            
            print(rm_log)
        
        print(101112)    
        self.fnl_selected_feature=list(set(candidate)-set(rm_log[fnl_step]))
        self.X_train=self.X_train.loc[:,self.fnl_selected_feature]
        self.X_test=self.X_test.loc[:,self.fnl_selected_feature]
        self.test_x=self.test_x.loc[:,self.fnl_selected_feature]
        
    
        
        return self
    

#debug duplicate rm    
#{0: [], 1: ['behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1'], 2: ['behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1', 'behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1'], 3: ['behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1', 'behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1', 'behavioral_antiviral_meds_and_behavioral_wash_hands_or_doctor_recc_h1n1']}

    def model_train (self):
    
        X_train=self.X_train.copy()
        y_train=self.y_train.copy()
        X_test =self.X_test.copy()
        y_test =self.y_test.copy()
        test_x =self.test_x.copy()
        fnl_feature=self.fnl_feature
        target=self.new_target
        file_name=self.file_name
        submit=self.submit
        root=self.root
        show_n=self.show_n
    
        #train model with selected features    
        gbm=lgb.LGBMClassifier(
             n_estimators=500 
            ,objective='multiclass'
            ,metric=['auc_mu']
            ,early_stopping_rounds=50
            ,boosting_type= 'gbdt'
            ,learning_rates=0.01
            ,num_leaves= 7
            ,bagging_fraction= 0.5
            ,bagging_freq= 1
            ,feature_fraction= 1
            ,min_data_in_bin=500
            ,min_data_in_leaf= 50
            ,cat_smooth=50
            ,max_cat_to_onehot=4
            ,extra_trees=True
            ,is_unbalance=True
            ,class_weight= {'S':0.8,'H':1.2,'HS':1.1,'N':0.9}
            ,feature_penalty=[0.80 if 'age_group' in i and 'census_msa' in i else (0.90 if 'census_msa' in i else (0.90 if 'age_group' in i else 1.0)) for i in fnl_feature]
            
        ).fit(
                     X_train.loc[:,fnl_feature]
                     ,y_train.loc[:,'new_target']
                     ,eval_set=[    (X_train.loc[:,fnl_feature],y_train.loc[:,'new_target'])
                                    ,(X_test.loc[:,fnl_feature],y_test.loc[:,'new_target'])]
                     ,eval_names=['all train','ext val']
                     ,verbose=False
                     )
        
        lgb.plot_metric(gbm,metric='auc_mu')
        lgb.plot_importance(gbm,max_num_features=show_n)
    
        print('feature used: '+str(sum(gbm.feature_importances_>0)))
        
        t_pred=gbm.predict_proba(X_train.loc[:,fnl_feature],raw_score=False)
        td=X_train.loc[:,fnl_feature].copy()
        for i in range(0,len(t_pred)):
            td.loc[i,'h1n1_vaccine_pred']    =t_pred[i,0]+t_pred[i,1]
            td.loc[i,'seasonal_vaccine_pred']=t_pred[i,1]+t_pred[i,3]
        
        print('h1n1 training auc: '    + str(roc_auc_score(y_train.loc[:,'h1n1_vaccine'], td.loc[:,'h1n1_vaccine_pred'])))
        print('seasonal training auc: '+ str(roc_auc_score(y_train.loc[:,'seasonal_vaccine'], td.loc[:,'seasonal_vaccine_pred'])))
        print('avg training auc: ' + str( (roc_auc_score(y_train.loc[:,'h1n1_vaccine'], td.loc[:,'h1n1_vaccine_pred']) + roc_auc_score(y_train.loc[:,'seasonal_vaccine'], td.loc[:,'seasonal_vaccine_pred'] ))/2))
    
        v_pred=gbm.predict_proba(X_test.loc[:,fnl_feature],raw_score=False)    
        vd=X_test.loc[:,fnl_feature].copy()
        for i in range(0,len(v_pred)):
            vd.loc[i,'h1n1_vaccine_pred']    =v_pred[i,0]+v_pred[i,1]
            vd.loc[i,'seasonal_vaccine_pred']=v_pred[i,1]+v_pred[i,3]
        print('h1n1 ext val auc: '+ str(roc_auc_score(y_test.loc[:,'h1n1_vaccine'], vd.loc[:,'h1n1_vaccine_pred'])))
        print('seasonal ext val auc: '+ str(roc_auc_score(y_test.loc[:,'seasonal_vaccine'], vd.loc[:,'seasonal_vaccine_pred'])))
        print('avg ext val auc: ' + str( (roc_auc_score(y_test.loc[:,'h1n1_vaccine'], vd.loc[:,'h1n1_vaccine_pred']) + roc_auc_score(y_test.loc[:,'seasonal_vaccine'], vd.loc[:,'seasonal_vaccine_pred'] ))/2))
    
        fnl_pred=gbm.predict_proba(test_x.loc[:,fnl_feature],raw_score=False)    
        fd=test_x.loc[:,list(fnl_feature)+['respondent_id']].copy()
        for i in range(0,len(fnl_pred)):
            fd.loc[i,'h1n1_vaccine']    =fnl_pred[i,0]+fnl_pred[i,1]
            fd.loc[i,'seasonal_vaccine']=fnl_pred[i,1]+fnl_pred[i,3]
        if submit:
            fd.loc[:,['respondent_id','h1n1_vaccine','seasonal_vaccine']].to_csv(root+file_name+'.csv',index=False) 
            print('csv file is created for submission!')
    
 
        return self
        

#add plotting
#add train all (which may mean need two set of append for test_X)
