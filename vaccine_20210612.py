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



def drop_high_corr(df,col,threshold=0.95):
    num_f=[i for i in col if is_numeric_dtype(df[i])==True ]
    corr_matrix = df.loc[:,num_f].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = set([column for column in upper.columns if any(upper[column] > threshold)])

    print('dropped '+str(len(to_drop))+' highly correlated features out of '+str(len(col))+'!')

    return to_drop


class vaccine_classifier:
    
    '''
    A class designed specifically for this competition, starting from data load to create submission file.
    
    '''
    
    
    def __init__ (self
                  ,file_name
                  ,submit
                  ,top_k
                  ,run_from_beginning 
                  ,show_n=20
                  ,repeat=5
                  ,nfold=5
                  ,concervative=True
                  ,drop_corr=True
                  ,root='/Users/raymondlei/Downloads/project/'
                  ,debug=True
                      
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
        self.debug=debug
        

    def parameter_qc(self):
        '''
        A QC runs prior everything to make sure all parameters are defined as expected
        '''
        
        
        

    def checkpoint(self):
        
        with open(self.cur_check_name, 'wb') as f:
             pickle.dump(self, f)
             
        #double check to make sure saving sucessfully     
        os.rename(self.cur_check_name ,self.new_check_name)
        print('saved!')     
 
    
    def resume(self):
        self = pickle.load( open( self.new_check_name, "rb" ) )  
        print('resumed from {}'.format(self.step_log[self.log_step]))
        
        return self 
        
    def execute(self):

        '''
        
        Execute the pipeline in sequence with check point to save a state of each step
        
        '''        

        self.step_log={
            
         #1st step   
         0:'self.load(step=1)'
        ,1:'self.create_label(step=1)'
        ,2:'self.cast()'
        ,3:'self.split()'
        ,4:'self.create_combination()'
        ,5:'self.create_num_trans()'
        ,6:'self.create_index(step=1)'
        ,7:'self.create_agg()' 
        ,8:'self.drop_unused()'
        ,9:'self.feature_selection(step=1)'
        ,10:'self.model_train(step=1)'
        
        #2nd step
        ,11:'self.load(step=2)'
        ,12:'self.create_label(step=2)'
        ,13:'self.cast()'
        ,14:'self.split()' #it will be the same split as 1st step
        
        #following step will be identical to 1st step, however, the feature created will be different because of using different data
        ,15:'self.create_combination()'
        ,16:'self.create_num_trans()'
        ,17:'self.create_index(step=2)'
        ,18:'self.create_agg()' 
        ,19:'self.drop_unused()'
        
        #train model for 2nd step
        ,20:'self.feature_selection(step=2)'
        ,21:'self.model_train(step=2)'
        
        }        
        self.log_step=1    

        if self.run_from_beginning==False and os.path.exists(self.new_check_name):
            if self.log_step<21:
                self=self.resume()
                while self.log_step<max(self.step_log.keys()):
                    exec(self.step_log[self.log_step])
                    self.log_step+=1
                    self.checkpoint()
            else:
                print("Last run is a complete end-to-end run. Nothing to rerun.")
        else:
            if os.path.exists(self.new_check_name):
                os.remove(self.new_check_name)
            for k, v in self.step_log.items():  
                exec(v)
                if k<21:
                    self.log_step+=1
                    self.checkpoint()
                else:
                    self.checkpoint()
                    print("This is the end of process!")
        
 
        
    def load (self,step):
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
        
        if step==1:
            self.fail_col=[]

        if step==2:
            # sample for second step would just limit to those who took at least 1 type of vaccine
            self.train=self.train.loc[(self.train.h1n1_vaccine>0) | (self.train.seasonal_vaccine>0)]
        
        print(len(self.train))    
        #target ratios (assuming no duplicates)
        print('h1n1_vaccine_target_ratios: ' + str(self.train_y.h1n1_vaccine.sum()/len(self.train_y)))
        print('seasonal_Vaccine_target_ratios: ' + str(self.train_y.seasonal_vaccine.sum()/len(self.train_y)))
        print('number of features: {}'.format(str(len(self.train_x.columns))))

        return self


    def create_label(self,step,drop_orig=False):  
        '''
        
        Parameters
        ----------
        drop_orig : Bool, optional
            Drop original targets. The default is False.

        '''
    
        def f(row):
            v=''
            if row[self.raw_target[0]]==1 and row[self.raw_target[1]]==1:
                v='HS'
            elif row[self.raw_target[0]]==1 and row[self.raw_target[1]]==0:
                v='H'
            elif row[self.raw_target[0]]==0 and row[self.raw_target[1]]==1:
                v='S'
            else:
                v='N'
        
            return v
    
        #convert multi-label to multi-class
        self.train['new_target']=self.train.apply(f,axis=1)
        
        # create binary target for step 1
        if step==1:
            self.train.new_target=self.train.new_target.map(lambda x: 'N' if x=='N' else 'Y' )
        
        if drop_orig:
            self.train.drop([self.raw_target[0],self.raw_target[1]],axis=1,inplace=True)
            self.new_target=['new_target']
        else:
            self.new_target=[self.raw_target[0],self.raw_target[1],'new_target']
            
        
        return self
    
        
    def cast(self):
        for i in self.train.select_dtypes(exclude=['float64']).columns:
            if i not in ['seasonal_vaccine','h1n1_vaccine']:
                self.train[i]=self.train[i].astype('category')
            if i not in ['seasonal_vaccine','h1n1_vaccine','new_target']:
                self.test_x[i]=self.test_x[i].astype('category')
    
        return self
    
    def split(self):
        '''
        Split data into training and validation

        '''
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.train.drop(self.new_target,axis=1),self.train.loc[:,self.new_target],test_size=0.2,random_state=100)
        
        return self


    def create_combination(self):
        '''
        Create new predictor by combining binary attributes

        '''
    
    
        t=self.X_train.copy()
        v=self.X_test.copy()
        te=self.test_x.copy()
        columns=self.binary_col

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
                    

        self.X_train=t.copy()
        self.X_test=v.copy()
        self.test_x=te.copy()
        
        self.comb_col=[i for i in self.X_train.columns if '_comb' in i ]
        
        
        return self

    def create_num_trans(self):
        '''
        Convert categorical variable to ordinal
        '''
        
        def convert(df,num_col):
            df['age_group_t']=df['age_group'].map(lambda x: 1 if x=='18 - 34 Years' else (2 if x=='35 - 44 Years' else (3 if x=='45 - 54 Years' else (4 if x=='55 - 64 Years' else (5 if '65+ Years' else -1))))).map(lambda x: float(x)).astype('float64')    
            df['education_t']=df['education'].map(lambda x: 1 if x=='< 12 Years' else (2 if x=='12 Years' else (3 if x=='Some College' else (4 if x=='College Graduate' else -1)))).map(lambda x: float(x)).astype('float64')     
            df['income_poverty_t']=df['income_poverty'].map(lambda x: 1 if x=='Below Poverty' else (2 if x=='<= $75,000, Above Poverty' else (3 if x=='> $75,000' else -1))).map(lambda x: float(x)).astype('float64')  
            
            for col in num_col:
                df[col+'_log_t']=np.log(df[col]+2) 
            
            return df

        self.X_train=convert(self.X_train.copy(),self.num_col)
        self.X_test =convert(self.X_test.copy(),self.num_col)
        self.test_x =convert(self.test_x.copy(),self.num_col)
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
           
        return self
 

 
    
    def create_index(self,step,targets=['new_target'],drop_orig=False,keep_unweighted=False):
        '''
        Create indexes on weighted basis

        Parameters
        ----------
        targets : List, optional
            target used in modeling. The default is ['new_target'].
        drop_orig : Bool, optional
            Drop original target columns. The default is False.
        keep_unweighted : Bool, optional
            Do not remove the unweighted version. The default is False.


        '''
        
        
        columns=self.cat_col    
        train=pd.merge(left=self.X_train,right=self.y_train,how='inner',left_index=True,right_index=True)
        drop_corr=self.drop_corr
        
        int_train=pd.get_dummies(train.copy(),columns= list(set(targets)-set(['h1n1_vaccine','seasonal_vaccine'])),prefix='',prefix_sep='')
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
        
        if step==2:
            X_train_s1=self.X_train_s1.copy()
            X_test_s1=self.X_test_s1.copy()
            test_x_s1=self.test_x_s1.copy()
    
        def append(t,v,ev,grp,kc,c):
            t =pd.merge(left=t ,right=grp.loc[:,kc],how='left',on=c)
            v =pd.merge(left=v ,right=grp.loc[:,kc],how='left',on=c)
            ev=pd.merge(left=ev,right=grp.loc[:,kc],how='left',on=c)
            
            return t, v, ev
    
        new_name=[]
        for target in set(train.copy()[targets[0]]):
            int_train['no_'+target] = int_train[target].map(lambda x: 1-x)
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
            
                int_train,ext_val,test = append(t=int_train,v=ext_val,ev=test,grp=grp,kc=keep_col,c=col)
                
                if step==2:    
                    try:
                        X_train_s1,X_test_s1,test_x_s1 = append(t=X_train_s1,v=X_test_s1,ev=test_x_s1,grp=grp,kc=keep_col,c=col)
                    
                    except:
                        self.fail_col.append(col+'_'+target+'_ratios')
                
            columns_a=columns.copy()
            columns_b=columns.copy() + self.binary_col 
            
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
            
                        int_train,ext_val,test = append(t=int_train,v=ext_val,ev=test,grp=grp,kc=keep_col,c=[col1,col2])
                
                        if step==2:
                            try:
                                X_train_s1,X_test_s1,test_x_s1 = append(t=X_train_s1,v=X_test_s1,ev=test_x_s1,grp=grp,kc=keep_col,c=[col1,col2])
                      
                            except:
                                self.fail_col.append(col1+'_'+col2+'_'+target+'_ratios')
                    else:
                        columns_b.remove(col2)
                    

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
        
        if step==2:
            self.X_train_s1=X_train_s1.copy()
            self.X_test_s1=X_test_s1.copy()
            self.test_x_s1=test_x_s1.copy()
        
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
        
        self.index_col=[i for i in self.X_train.columns if '_ratios_weighted' in i]

        return self

 

    def create_agg(self):
        '''
        Create aggregated version
        '''
        
        cat_col=self.cat_col
        num_col=[   self.num_col 
                , ['age_group_t','education_t','income_poverty_t'] 
                ,self.comb_col [0:100]
                 ] 
        drop_corr=self.drop_corr
            
        int_train=self.X_train.copy()
        ext_val=self.X_test.copy()
        test=self.test_x.copy()
    
        def agg_burst(data,cat_col,num_col,return_feature_name=True,exclude_outlier=True):
            '''
            Burst out the combination by looking different aggregation methods

            Parameters
            ----------
            data : dataframe
                Sample used in the burst.
            cat_col : List
                Categorical columns.
            num_col : List
                Numeric columns.
            return_feature_name : Bool, optional
                Wether to return feature name. The default is True.
            exclude_outlier : Bool, optional
                Wether to remove outlier. The default is True.

            Returns
            -------
            dataframe/dataframe and a list of feature name
                DESCRIPTION.

            '''
            
            
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
        '''
        Drop unused columns

        Parameters
        ----------
        drop_col : List, optional
            Columns to drop. The default is ['age_group','education','income_poverty'].


        '''
        if any(i in drop_col for i in self.X_train.columns):
            self.X_train.drop(drop_col,axis=1,inplace=True)
            self.X_test.drop(drop_col,axis=1,inplace=True)
            self.test_x.drop(drop_col,axis=1,inplace=True)
        
        print('number of features: {}'.format(str(len(self.X_train.columns))))
        
        return self


    def feature_selection(self,step):
        
        '''
        Run preliminary selection on predictor ranked by importance
        
        '''
        
        if step==1:
            self.obj='binary'
            self.metric='auc'
        else:
            self.obj='multiclass'
            self.metric='auc_mu'
        
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
    
        def rand_imp(f_log,N):
            imp=[]
            for i in range(1,N+1):
                r_imp=f_log.loc[f_log.feature=='rand'+str(i)].reset_index().feature_importance[0]    
                imp.append(max(0,r_imp))
                
            if self.concervative:
                cutoff=sorted(imp,reverse=True)[1]
            else:
                cutoff=max(imp)
                        
            return cutoff
                
        X_train, n_rand, rand_factor,rand_name =insert_random(self.X_train.copy())
        y_train=self.y_train.copy().reset_index().drop(columns='respondent_id',axis=1)
    
        sc=[]
        k=1
    
        #cross validation
        folds=RepeatedKFold(n_splits=self.nfold,n_repeats=5,random_state=100)
        for n, (t,v) in enumerate(folds.split(X_train.values,y_train.values)):
             print("fold n°{}".format(n))
     
             try:
                 
                 gbm_fs=lgb.LGBMClassifier(
                        n_estimators=1000 
                        ,objective=self.obj
                        ,metric=[self.metric]
                        ,early_stopping_rounds=50
                        ,boosting_type= 'gbdt'
                        ,learning_rates=0.005
                        ,num_leaves= 10
                        ,bagging_fraction= 0.5
                        ,bagging_freq= 1
                        ,feature_fraction= 0.5
                        ,min_data_in_bin=5000
                        ,min_data_in_leaf= 20
                        ,cat_smooth=200
                        ,max_cat_to_onehot=3
                        ,extra_trees=True
                        #,cegb_penalty_feature_coupled=[ 1.0 for i in set(X_train.columns)]  
                        #,cegb_tradeoff=5 + np.log2(k+1)
                        ,is_unbalance=True
                        #,class_weight={'S':0.8,'H':1.2,'HS':1.1,'N':0.9}
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
                 cutoff=rand_imp(feature_log,n_rand)
            
                 sc=sc.copy() + list(set(feature_log.loc[feature_log.feature_importance>cutoff]['feature'].tolist())-set(rand_name))        
                 print(len(sc))
                
             except:
                 print('Error occurs during this split; this split is skipped!')
             
             k+=1    
    
             print(str(k))
            
    
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
    
        if self.drop_corr:
            #drop highly correlated columns to shrink down the size of data, may lose the best bet in term of predictability; but impact is minimum 
            to_drop=drop_high_corr(X_train,sf,threshold=0.99)
            X_train.drop(columns=to_drop,axis=1,inplace=True)
            self.X_test.drop (columns=to_drop,axis=1,inplace=True)
            self.test_x.drop (columns=to_drop,axis=1,inplace=True)
    
            print(str(len(to_drop))+' features are dropped due to high correlation!')
        else:
            to_drop=[]
        
        fnl_feature=set(sf)-set(to_drop)
        print(str(len(fnl_feature)))

        self.fnl_feature=fnl_feature
        self.sf_report=sf_report

 
        return self


    def model_train (self,step):
    
        if step==1:
            self.obj='binary'
            self.metric='auc'
        else:
            self.obj='multiclass'
            self.metric='auc_mu'    
    
        show_n=self.show_n
    
        #train model with selected features    
        gbm=lgb.LGBMClassifier(
             n_estimators=1000 
            ,objective=self.obj
            ,metric=[self.metric]
            ,early_stopping_rounds=50
            ,boosting_type= 'gbdt'
            ,learning_rates=0.005
            ,num_leaves= 10
            ,bagging_fraction= 0.5
            ,bagging_freq= 1
            ,feature_fraction= 0.5
            ,min_data_in_bin=5000
            ,min_data_in_leaf= 30
            ,cat_smooth=200
            ,max_cat_to_onehot=3
            ,extra_trees=True
            ,is_unbalance=True
            #,class_weight= {'S':0.8,'H':1.2,'HS':1.1,'N':0.9}
            ,feature_penalty=[0.80 if 'age_group' in i and 'census_msa' in i else (0.90 if 'census_msa' in i else (0.90 if 'age_group' in i else 1.0)) for i in self.fnl_feature]
            
        ).fit(
                     self.X_train.loc[:,self.fnl_feature]
                     ,self.y_train.loc[:,'new_target']
                     ,eval_set=[    (self.X_train.loc[:,self.fnl_feature],self.y_train.loc[:,'new_target'])
                                    ,(self.X_test.loc[:,self.fnl_feature],self.y_test.loc[:,'new_target'])]
                     ,eval_names=['all train','ext val']
                     ,verbose=False
                     )
        
        lgb.plot_metric(gbm,metric=self.metric)
        lgb.plot_importance(gbm,max_num_features=show_n)
    
        print('feature used: '+str(sum(gbm.feature_importances_>0)))
        
        
        def eval(X,y,pred,pred_name,prefix,step,target='new_target'):
            
            #initialize
            X[pred_name]=0
            
            if step==1:
                for i in range(0,len(pred)):
                    X.loc[i,pred_name]=pred[i,1]  
                try:    
                    print(prefix+' auc: '    + str(roc_auc_score(y.loc[:,target], X.loc[:,pred_name])))
                except:
                    print('final prediction is done!')
            
            else:
                for pred_name in ['H_pred','S_pred','HS_pred']:
                    for i in range(0,len(pred)):
                        X.loc[i,pred_name]=pred[i,1]  
                        
                    if pred_name == 'H_pred':
                        print(prefix+' auc: '    + str(roc_auc_score(y.loc[:,target].map(lambda x: 1 if x=='H' else 0), X.loc[:,pred_name])))
                    if pred_name == 'S_pred':
                        print(prefix+' auc: '    + str(roc_auc_score(y.loc[:,target].map(lambda x: 1 if x=='S' else 0), X.loc[:,pred_name])))
                    if pred_name == 'HS_pred':
                        print(prefix+' auc: '    + str(roc_auc_score(y.loc[:,target].map(lambda x: 1 if x=='HS' else 0), X.loc[:,pred_name])))
                    
            
            return X,pred_name
        
       
        def eval2(X,y,pred,step1_pred_name,prefix1,prefix2,pred_name1='h1n1_vaccine_pred',pred_name2='seasonal_vaccine_pred',target1='h1n1_vaccine',target2='seasonal_vaccine'):
 
             #initialize
             X[pred_name1]=0
             X[pred_name2]=0
            
             for i in range(0,len(pred)):
                 X.loc[i,pred_name1]=(pred[i,0]+pred[i,1]) * X.loc[i,step1_pred_name]  #TBC 0+1>0+1
                 X.loc[i,pred_name2]=(pred[i,1]+pred[i,2]) * X.loc[i,step1_pred_name]  #TBC 1+2>0+2
             
             try:    
                 print(prefix1+' auc: '+ str(roc_auc_score(y.loc[:,target1], X.loc[:,pred_name1])))
                 print(prefix2+' auc: '+ str(roc_auc_score(y.loc[:,target2], X.loc[:,pred_name2])))
                 print('avg auc: ' + str( (roc_auc_score(y.loc[:,target1], X.loc[:,pred_name1]) + roc_auc_score(y.loc[:,target2], X.loc[:,pred_name2] ))/2))
             except:
                 print('final prediction is done!')
            
             return X
 
        
        if step==1:
            
            self.X_train_s1=self.X_train.copy()
            self.X_test_s1=self.X_test.copy()
            self.test_x_s1=self.test_x.copy()
            self.y_train_s1=self.y_train.copy()
            self.y_test_s1=self.y_test.copy()
            
            #internal training
            t_pred=gbm.predict_proba(self.X_train.loc[:,self.fnl_feature],raw_score=False)
            self.X_train_s1,self.pred_name_s1=eval(step=step,X=self.X_train_s1,y=self.y_train_s1,pred=t_pred,pred_name='bin_pred',prefix='step1 training',target='new_target')           
            
            #external validation
            v_pred=gbm.predict_proba(self.X_test.loc[:,self.fnl_feature],raw_score=False)    
            self.X_test_s1,self.pred_name_s1=eval(step=step,X=self.X_test_s1,y=self.y_test_s1,pred=v_pred,pred_name='bin_pred',prefix='step1 ext val',target='new_target')           
  
            #test data
            fnl_pred=gbm.predict_proba(self.test_x.loc[:,self.fnl_feature],raw_score=False)    
            self.test_x_s1,self.pred_name_s1=eval(step=step,X=self.test_x_s1,y='',pred=fnl_pred,pred_name='bin_pred',prefix='',target='')           
  
        else:
            
            '''
            #internal training
            t_pred=gbm.predict_proba(self.X_train.loc[:,self.fnl_feature],raw_score=False)
            eval(step=step,X=self.X_train_s1,y=self.y_train_s1,pred=t_pred,pred_name='mult_pred',prefix='step2 training',target='new_target')           
            
            #external validation
            v_pred=gbm.predict_proba(self.X_test.loc[:,self.fnl_feature],raw_score=False)    
            eval(step=step,X=self.X_test_s1,y=self.y_test_s1,pred=v_pred,pred_name='mult_pred',prefix='step2 ext val',target='new_target')           
  
            #test data
            fnl_pred=gbm.predict_proba(self.test_x.loc[:,self.fnl_feature],raw_score=False)    
            eval(step=step,X=self.test_x_s1,y='',pred=fnl_pred,pred_name='mult_pred',prefix='',target='')           
            '''
            
            
            #internal training
            t_pred=gbm.predict_proba(self.X_train_s1.loc[:,self.fnl_feature],raw_score=False)
            self.X_train_s1=eval2(X=self.X_train_s1,y=self.y_train_s1,pred=t_pred,step1_pred_name=self.pred_name_s1,prefix1='h1n1 training ',prefix2='seasonal training ')
            
            #external validation
            v_pred=gbm.predict_proba(self.X_test_s1.loc[:,self.fnl_feature],raw_score=False)    
            self.X_test_s1=eval2(X=self.X_test_s1,y=self.y_test_s1,pred=v_pred,step1_pred_name=self.pred_name_s1,prefix1='h1n1 ext val  ',prefix2='seasonal ext val ')
           
            #test data
            fnl_pred=gbm.predict_proba(self.test_x_s1.loc[:,self.fnl_feature],raw_score=False) 
            self.test_x_s1=eval2(X=self.test_x_s1.loc[:,list(self.fnl_feature)+['respondent_id']+[self.pred_name_s1]],y='',pred_name1='h1n1_vaccine',pred_name2='seasonal_vaccine',pred=fnl_pred,step1_pred_name=self.pred_name_s1,prefix1='',prefix2='',target1='',target2='')
           
 
            if self.submit:
                self.test_x_s1.loc[:,['respondent_id','h1n1_vaccine','seasonal_vaccine']].to_csv(self.root+self.file_name+'.csv',index=False) 
                print('csv file is created for submission!')
       
 
        return self
        

 