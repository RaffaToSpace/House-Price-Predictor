# Predicting the housing price - Preprocessing and Prediction module prototype

List of functions and description.

### Data preparation

example:

_data= pd.read_csv('data/train.csv')  
preprocessed_data = pthp_dp.p_h_p_data_handling(data)  
be_feat_data =BE_feat_selection(preprocessed_data)_

*p_h_p_data_handling*: takes raw data in the format of the project dataset, and returns a preprocessed dataset, i.e. with categorical and non-numerical variables dealt with.

Returns a preprocessed dataset.

## Feature engineering
### Selecting variables correlated with target
_pearson_corr_skimming(preprocessed_ds)_

Returns dataset with reduced dimensionality through feature selection based on correlation with target.

### Removing redundant variables (i.e. highly correlated to others)
_def redundant_var_remove(preprocessed_ds)_

Returns dataset with reduced dimensionality through feature selection based on correlation between variables.

### Dimensionality reduction through PCA
_pca_reduction(preprocessed_ds)_

Returns dataset with reduced dimensionality through principal component analysis.

### Feature selection through backward elimination
_BE_feat_selection(preprocessed_ds)_

Returns dataset  with reduced dimensionality through backward elimination, a process that iteratively removed the least important features as long as the model perfors with certain levels of metrics.

### Feature selection through LassoCV
_LassoCV_feat_selection(preprocessed_ds)_

Returns dataset  with reduced dimensionality through LassoCV method.

