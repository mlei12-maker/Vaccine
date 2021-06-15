#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:13:41 2021

@author: raymondlei
"""

import vaccine_20210612 as vc
print(vc._version)
 


vac=vc.vaccine_classifier(file_name='V37'
                                        ,submit=False
                                        ,top_k=500
                                        ,run_from_beginning=True 
                                        ,concervative=True
                                        ,drop_corr=True
                                        ,debug=True
                                        )
vac.execute()
 
 