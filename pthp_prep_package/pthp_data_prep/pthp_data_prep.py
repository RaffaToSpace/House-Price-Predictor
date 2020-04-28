#!/usr/bin/env python
# coding: utf-8

# # Predicting the housing price - Preprocessing and Prediction module prototype
# 
# List of functions and description.

# In[22]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle


# ### Data preparation
# 
# example:
# 
# _data= pd.read_csv('data/train.csv')  
# preprocessed_data = pthp_dp.p_h_p_data_handling(data)  
# be_feat_data =BE_feat_selection(preprocessed_data)_

# In[19]:


def p_h_p_data_handling(data):
    Neighborhood = pd.read_csv('Neighborhood_coordinates.csv')
    Neighborhood=Neighborhood.drop('Unnamed: 0',axis=1)
    
    MSZoning = {'A':1,'C':2,'C (all)':2,'FV':3,'I':4,'RH':5,'RL':6,'RP':7,'RM':8}
    Street = {'Grvl':1,'Pave':2,'unknown':0}
    LotShape= {'Reg':0,'IR1':1,'IR2':2,'IR3':3}
    LandContour={'Lvl':0,'Bnk':1,'HLS':2,'Low':3}
    Utilities = {'AllPub':0,'NoSewr':1,'NoSeWa':2,'ELO':3}
    LotConfig= {'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4}
    LandSlope ={'Gtl':0,'Mod':1,'Sev':2}
    Condition = {'PosN':1,'PosA':1,'Norm':0,'Artery':-1,'Feedr':-1,'RRNn':-1,'RRAn':-1,'RRNe':-1,'RRAe':-1}
    BldgType ={'Twnhs':0,'TwnhsI':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4}

    HouseStyle ={'SLvl':4,'1Story':6,'1.5Fin':3,'1.5Unf':1.5,'SFoyer':2,'2Story':7,'2.5Fin':8,'2.5Unf':7.5}
    RoofStyle={'Flat':4,'Shed':0,'Gable':2,'Hip':5,'Gambrel':1,'Mansard':3}
    RoofMatl = {'Membran':5,'Tar&Grv':1,'WdShake':4,'WdShngl':6,'Metal':2,'ClyTile':0,'CompShg':3,'Roll':-1}
    Exterior1st={'CBlock':0,'Other':0,'PreCast':0,'Stone':0,'BrkComm':1,'Brk Cmn':1,'AsphShn':2,'AsbShng':3,'ImStucc':4,'Stucco':4,'Wd Sdng':5,'MetalSd':6,'HdBoard':7,'Wd Shng':8,'WdShing':8,'Plywood':9,'BrkFace':10,'VinylSd':11,'CemntBd':12,'CmentBd':12}
    MasVnrType={'BrkCmn':1,'BrkFace':3,'CBlock':0,'None':2,'Stone':4}
    ExterQual={'Fa':2,'Gd':4,'Po':1,'TA':3,'Ex':5}
    Foundation={'BrkTil':2,'CBlock':4,'Slab':1,'Wood':3,'Stone':5,'PConc':6}
    BsmtQC={'Po':1,'TA':3,'NA':0,'Fa':2,'Gd':4,'Ex':5}
    BsmtExposure={'No':1,'Av':3,'NA':0,'Mn':2,'Gd':4}
    BsmtFinType1 ={'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
    Heating ={'Floor':0,'Grav':1,'Wall':2,'GasW':3,'GasA':4,'OthW':0}
    HeatingQC ={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    CentralAir={'Y':1,'N':0}
    Electrical={'Mix':1,'FuseP':2,'FuseF':3,'FuseA':4,'SBrkr':5,'NA':0}
    KitchenQual={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    Functional= {'Typ':0,'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6,'Sal':7}
    FireplaceQu= {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    GarageType = {'NA':0,'CarPort':1,'Detchd':2,'2Types':3,'Basment':4,'Attchd':5,'BuiltIn':6}
    GarageFinish = {'NA':0,'Unf':1,'RFn':2,'Fin':3}
    GarageQual= {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    PavedDrive={'N':0,'P':1,'Y':2}
    PoolQC= {'NA':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
    Fence= {'NA':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}

    data['MSZoning']=data['MSZoning'].apply(lambda x: MSZoning[x])
    data.LotFrontage=data.LotFrontage.fillna(0)
    data['Street']=data['Street'].apply(lambda x: Street[x])
    data.Alley=data.Alley.fillna('unknown')
    data['Alley']=data['Alley'].apply(lambda x: Street[x])
    data['LotShape']=data['LotShape'].apply(lambda x: LotShape[x])
    data['LandContour']=data['LandContour'].apply(lambda x: LandContour[x])
    data['Utilities']=data['Utilities'].apply(lambda x: Utilities[x])
    data['LotConfig']=data['LotConfig'].apply(lambda x: LotConfig[x])
    data['LandSlope']=data['LandSlope'].apply(lambda x: LandSlope[x])
    for ii in range(len(data['Neighborhood'])):
        data.loc[ii,'Distance_from_centre']=float(Neighborhood[Neighborhood.Code==data.loc[ii]['Neighborhood']]['Distance_from_centre'])
    data['Condition1']=data['Condition1'].apply(lambda x: Condition[x])
    data['Condition2']=data['Condition2'].apply(lambda x: Condition[x])
    data['BldgType']=data['BldgType'].apply(lambda x: BldgType[x])
    data['HouseStyle']=data['HouseStyle'].apply(lambda x: HouseStyle[x])
    data['YearBuilt']=data['YearBuilt'].apply(lambda x: 2020-x)
    data['YearRemodAdd']=data['YearRemodAdd'].apply(lambda x: 2020-x)
    data['RoofStyle']=data['RoofStyle'].apply(lambda x: RoofStyle[x])
    data['RoofMatl']=data['RoofMatl'].apply(lambda x: RoofMatl[x])
    data['Exterior1st']=data['Exterior1st'].apply(lambda x: Exterior1st[x])
    data['Exterior2nd']=data['Exterior2nd'].apply(lambda x: Exterior1st[x])
    data.MasVnrType=data.MasVnrType.fillna('None')
    data['MasVnrType']=data['MasVnrType'].apply(lambda x: MasVnrType[x])
    data['ExterQual']=data['ExterQual'].apply(lambda x: ExterQual[x])
    data['ExterCond']=data['ExterCond'].apply(lambda x: ExterQual[x])
    data['Foundation']=data['Foundation'].apply(lambda x: Foundation[x])
    data.BsmtQual=data.BsmtQual.fillna('NA')
    data['BsmtQual']=data['BsmtQual'].apply(lambda x: BsmtQC[x])
    data.BsmtCond=data.BsmtCond.fillna('NA')
    data['BsmtCond']=data['BsmtCond'].apply(lambda x: BsmtQC[x])
    data.BsmtExposure=data.BsmtExposure.fillna('NA')
    data['BsmtExposure']=data['BsmtExposure'].apply(lambda x: BsmtExposure[x])
    data.BsmtFinType1=data.BsmtFinType1.fillna('NA')
    data['BsmtFinType1']=data['BsmtFinType1'].apply(lambda x: BsmtFinType1[x])
    data.BsmtFinType2=data.BsmtFinType2.fillna('NA')
    data['BsmtFinType2']=data['BsmtFinType2'].apply(lambda x: BsmtFinType1[x])
    data['Heating']=data['Heating'].apply(lambda x: Heating[x])
    data['HeatingQC']=data['HeatingQC'].apply(lambda x: HeatingQC[x])
    data['CentralAir']=data['CentralAir'].apply(lambda x: CentralAir[x])
    data.Electrical=data.Electrical.fillna('NA')
    data['Electrical']=data['Electrical'].apply(lambda x: Electrical[x])
    data['KitchenQual']=data['KitchenQual'].apply(lambda x: KitchenQual[x])
    data['Functional']=data['Functional'].apply(lambda x: Functional[x])
    data.FireplaceQu=data.FireplaceQu.fillna('NA')
    data['FireplaceQu']=data['FireplaceQu'].apply(lambda x: FireplaceQu[x])
    data.GarageType=data.GarageType.fillna('NA')
    data['GarageType']=data['GarageType'].apply(lambda x: GarageType[x])
    data['GarageYrBlt']=data['GarageYrBlt'].apply(lambda x: 2020-x)
    data.GarageFinish=data.GarageFinish.fillna('NA')
    data['GarageFinish']=data['GarageFinish'].apply(lambda x: GarageFinish[x])
    data.GarageQual=data.GarageQual.fillna('NA')
    data['GarageQual']=data['GarageQual'].apply(lambda x: GarageQual[x])
    data.GarageCond=data.GarageCond.fillna('NA')
    data['GarageCond']=data['GarageCond'].apply(lambda x: GarageQual[x])
    data['PavedDrive']=data['PavedDrive'].apply(lambda x: PavedDrive[x])
    data.PoolQC=data.PoolQC.fillna('NA')
    data['PoolQC']=data['PoolQC'].apply(lambda x: PoolQC[x])
    data.Fence=data.Fence.fillna('NA')
    data['Fence']=data['Fence'].apply(lambda x: Fence[x])
    MiscFeature = pd.get_dummies(data['MiscFeature'])
    data = pd.concat([data,MiscFeature],axis=1)
    data=data.drop('MiscFeature',axis=1)
    data=data.drop('MoSold',axis=1)
    data['YrSold']=data['YrSold'].apply(lambda x: 2020-x)
    #data['New']=data['SaleType'].apply(lambda x: if x=='New': 1 else: 0)
    data.loc[data['SaleType']=='New','New']=1
    data.New=data.New.fillna(0)
    data=data.drop('SaleType',axis=1)
    SaleCondition= pd.get_dummies(data['SaleCondition'])
    data = pd.concat([data,SaleCondition],axis=1)
    data=data.drop('SaleCondition',axis=1)
    
    data=data.drop('Neighborhood',axis=1)

    return data


# ## Feature engineering
# ### Selecting variables correlated with target

# In[4]:


def pearson_corr_skimming(preprocessed_ds):
    relevant_features=['OverallQual','YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath','KitchenQual','TotRmsAbvGrd','FireplaceQu','GarageFinish','GarageCars','GarageArea','SalePrice']
    skimmed_ds = preprocessed_ds[relevant_features]
    return skimmed_ds


# ### Removing redundant variables (i.e. highly correlated to others)

# In[5]:


def redundant_var_remove(preprocessed_ds):
    toss = ['GarageYrBlt', 'Exterior2nd', '1stFlrSF', 'TotRmsAbvGrd', 'FireplaceQu', 'GarageArea', 'GarageCond', 'Gar2', 'Partial']
    redundancy_removed_ds = preprocessed_ds.drop(toss, axis=1)
    return redundancy_removed_ds


# ### Dimensionality reduction through PCA

# In[49]:


def pca_reduction(preprocessed_ds):
    
    X_from_feat_eng_col=set(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'YrSold',
       'Distance_from_centre', 'Gar2', 'Othr', 'Shed', 'New', 'Abnorml',
       'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial'])
    
    preproc_col = set(preprocessed_ds.columns.tolist())
    
    missing_cols=list(X_from_feat_eng_col.difference(preproc_col))
    extra_cols=list(preproc_col.difference(X_from_feat_eng_col))
    
    preprocessed_ds = preprocessed_ds.drop(extra_cols,axis=1)
    for col in missing_cols:
        preprocessed_ds[col] = 0
    
    #reorder columns
    preprocessed_ds =preprocessed_ds[X_from_feat_eng_col]
    
    scaler = MinMaxScaler()
    X_rescaled = scaler.fit_transform(preprocessed_ds)
    X_rescaled = np.nan_to_num(X_rescaled, nan=0.0, posinf=None, neginf=None)
    # load the PCA model
    filename = 'models/pthp_pca.sav'
    pthp_pca = pickle.load(open(filename, 'rb'))
    
    pca_reduced_ds=pthp_pca.transform(X_rescaled)
    
    return pca_reduced_ds


# ### Feature selection through backward elimination

# In[7]:


def BE_feat_selection(preprocessed_ds):
    selected_features_BE = ['MSSubClass', 'LotFrontage', 'LotArea', 'Condition2', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtUnfSF', 'TotalBsmtSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageArea', 'ScreenPorch', 'PoolQC', 'New', 'Abnorml','SalePrice']
    be_feat_selection = preprocessed_ds[selected_features_BE]
    return be_feat_selection


# ### Feature selection through recursive feature elimination

# In[8]:


def RFE_feat_selection(preprocessed_ds):
    selected_features_rfe = ['LotArea', 'Condition2', 'OverallQual', 'RoofMatl', 'ExterQual',
       'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'GarageArea', 'PoolQC','SalePrice']
    rfe_feat_selection = preprocessed_ds[selected_features_rfe]
    return rfe_feat_selection


# ### Feature selection through LassoCV

# In[9]:


def LassoCV_feat_selection(preprocessed_ds):
    selected_features_lassocv = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                                 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'MiscVal','SalePrice']
    lassocv_feat_selection=preprocessed_ds[selected_features_lassocv]
    return lassocv_feat_selection

