# I have found all the features that are currently outside the PCA are less important than random noise.
# Just there is an interesting feature that can be leveraged: 'nan_per_sample'
bow_features = ['industry', 'function', 'company_profile', 'title', 'description']  # long categorical features
onehot_features = ['required_experience', 'required_education', 'employment_type']  # shorter categorical features
binnan_features = ['country', 'state', 'city', 'department', 'salary_range', 'requirements', 'benefits']
# binnan_features = ['location', 'department', 'salary_range', 'requirements', 'benefits']  # binary encoded: '{col_name}_nan' & '{col_name}_notnan'
yesno_features = ['telecommuting', 'has_company_logo', 'has_questions']  # True or False features (of type 'category' however)
numerical_features = ['required_doughnuts_comsumption']  # actually speaking the only one of type 'float' instead of 'category'
synthetic_features = ['nan_per_sample']
