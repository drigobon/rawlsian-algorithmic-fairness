import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc

from tensorflow import square
from tensorflow.math import exp, log, reduce_sum

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import RandomNormal, Zeros

from tensorflow.keras.callbacks import EarlyStopping




#######################################
#           Loading Data              #
#######################################


def load_communities():

    cols = [
        'state',
        'county',
        'community',
        'communityname',
        'fold',
        'population',
        'householdsize',
        'racepctblack',
        'racePctWhite',
        'racePctAsian',
        'racePctHisp',
        'agePct12t21',
        'agePct12t29',
        'agePct16t24',
        'agePct65up',
        'numbUrban',
        'pctUrban',
        'medIncome',
        'pctWWage',
        'pctWFarmSelf',
        'pctWInvInc',
        'pctWSocSec',
        'pctWPubAsst',
        'pctWRetire',
        'medFamInc',
        'perCapInc',
        'whitePerCap',
        'blackPerCap',
        'indianPerCap',
        'AsianPerCap',
        'OtherPerCap',
        'HispPerCap',
        'NumUnderPov',
        'PctPopUnderPov',
        'PctLess9thGrade',
        'PctNotHSGrad',
        'PctBSorMore',
        'PctUnemployed',
        'PctEmploy',
        'PctEmplManu',
        'PctEmplProfServ',
        'PctOccupManu',
        'PctOccupMgmtProf',
        'MalePctDivorce',
        'MalePctNevMarr',
        'FemalePctDiv',
        'TotalPctDiv',
        'PersPerFam',
        'PctFam2Par',
        'PctKids2Par',
        'PctYoungKids2Par',
        'PctTeen2Par',
        'PctWorkMomYoungKids',
        'PctWorkMom',
        'NumIlleg',
        'PctIlleg',
        'NumImmig',
        'PctImmigRecent',
        'PctImmigRec5',
        'PctImmigRec8',
        'PctImmigRec10',
        'PctRecentImmig',
        'PctRecImmig5',
        'PctRecImmig8',
        'PctRecImmig10',
        'PctSpeakEnglOnly',
        'PctNotSpeakEnglWell',
        'PctLargHouseFam',
        'PctLargHouseOccup',
        'PersPerOccupHous',
        'PersPerOwnOccHous',
        'PersPerRentOccHous',
        'PctPersOwnOccup',
        'PctPersDenseHous',
        'PctHousLess3BR',
        'MedNumBR',
        'HousVacant',
        'PctHousOccup',
        'PctHousOwnOcc',
        'PctVacantBoarded',
        'PctVacMore6Mos',
        'MedYrHousBuilt',
        'PctHousNoPhone',
        'PctWOFullPlumb',
        'OwnOccLowQuart',
        'OwnOccMedVal',
        'OwnOccHiQuart',
        'RentLowQ',
        'RentMedian',
        'RentHighQ',
        'MedRent',
        'MedRentPctHousInc',
        'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg',
        'NumInShelters',
        'NumStreet',
        'PctForeignBorn',
        'PctBornSameState',
        'PctSameHouse85',
        'PctSameCity85',
        'PctSameState85',
        'LemasSwornFT',
        'LemasSwFTPerPop',
        'LemasSwFTFieldOps',
        'LemasSwFTFieldPerPop',
        'LemasTotalReq',
        'LemasTotReqPerPop',
        'PolicReqPerOffic',
        'PolicPerPop',
        'RacialMatchCommPol',
        'PctPolicWhite',
        'PctPolicBlack',
        'PctPolicHisp',
        'PctPolicAsian',
        'PctPolicMinor',
        'OfficAssgnDrugUnits',
        'NumKindsDrugsSeiz',
        'PolicAveOTWorked',
        'LandArea',
        'PopDens',
        'PctUsePubTrans',
        'PolicCars',
        'PolicOperBudg',
        'LemasPctPolicOnPatr',
        'LemasGangUnitDeploy',
        'LemasPctOfficDrugUn',
        'PolicBudgPerPop',
        'ViolentCrimesPerPop'
    ]

    df = pd.read_csv('./Datasets/CommunitiesAndCrime/communities.data', header = None, skipinitialspace = True)
    df.columns = cols
    df = df.replace({'?': np.nan}).dropna(axis = 1)

    df_out = df.drop(['state', 'communityname', 'fold'], axis = 1)

    df_out.loc[:,'PovertyPercent_Q1'] = df_out.PctPopUnderPov < 0.25
    df_out.loc[:,'PovertyPercent_Q2'] = (df_out.PctPopUnderPov > 0.25) & (df_out.PctPopUnderPov < 0.5)
    df_out.loc[:,'PovertyPercent_Q3'] = (df_out.PctPopUnderPov > 0.5) & (df_out.PctPopUnderPov < 0.75)
    df_out.loc[:,'PovertyPercent_Q4'] = df_out.PctPopUnderPov > 0.75
    

    y_name = 'ViolentCrimesPerPop'

    return (df_out.drop(y_name, axis = 1).astype(float), df_out.loc[:,y_name].astype(float))




def load_default():
    df = pd.read_csv('./Datasets/DefaultOfCreditCard/default.csv', skiprows = 1, index_col = 0)

    old_colname = df.columns[-1]
    df.loc[:,'default_next_month'] = df.loc[:,old_colname]
    df.drop(old_colname, axis = 1, inplace = True)


    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

    enc = OneHotEncoder()

    df_cat = pd.DataFrame(enc.fit_transform(df.loc[:,cat_cols]).todense(), columns = enc.get_feature_names_out(cat_cols))
    df_cts = df.drop(cat_cols, axis = 1)

    df_out = pd.concat([df_cts, df_cat], axis = 1)
    df_out = df_out.dropna()

    y_name = 'default_next_month'

    return (df_out.drop(y_name, axis = 1).astype(float), df_out.loc[:,y_name].astype(float))



def load_compas():
    df = pd.read_csv('./Datasets/COMPAS/compas-scores-two-years.csv', index_col = 0)

    cols = ['age',
            'c_charge_degree',
            'race', 
            'age_cat', 
            #'score_text', # not needed for trying to build a predictive model
            'sex',
            'priors_count',
            'days_b_screening_arrest',
            'decile_score',
            #'is_recid', # not needed
            'two_year_recid', 
            #'c_jail_in',
            #'c_jail_out',
           ]

    bools = ((-30 <= df.days_b_screening_arrest) & 
             (df.days_b_screening_arrest <= 30) & 
             #(df.is_recid != -1) & 
             (df.c_charge_degree != 'O') & 
             (df.score_text != "N/A")
            )

    df = df.loc[bools, cols]


    cat_cols = ['age_cat', 'race', 'sex', 'c_charge_degree']

    enc = OneHotEncoder()

    df_cat = pd.DataFrame(enc.fit_transform(df.loc[:,cat_cols]).todense(), columns = enc.get_feature_names_out(cat_cols))
    df_cts = df.drop(cat_cols, axis = 1)

    df_out = pd.concat([df_cts, df_cat], axis = 1)
    df_out = df_out.dropna()

    y_name = 'two_year_recid'

    return (df_out.drop(y_name, axis = 1).astype(float), df_out.loc[:,y_name].astype(float))



def load_bank():
    df = pd.read_csv('./Datasets/BankMarketing/bank-full.csv', sep = ';')

    df.drop(['day','month'], axis = 1, inplace = True)

    bin_cols = ['default', 'housing', 'loan', 'y']

    for col in bin_cols:
        df.loc[:, col] = df.loc[:, col]=='yes'


    cat_cols = ['job',
                'marital',
                'education',
                'contact',
                'poutcome',
               ]

    enc = OneHotEncoder()

    df_cat = pd.DataFrame(enc.fit_transform(df.loc[:,cat_cols]).todense(), columns = enc.get_feature_names_out(cat_cols))
    df_cts = df.drop(cat_cols, axis = 1)

    df_out = pd.concat([df_cts, df_cat], axis = 1)
    df_out = df_out.dropna()

    y_name = 'y'

    return (df_out.drop(y_name, axis = 1).astype(float), df_out.loc[:,y_name].astype(float))



def load_adult():
    cols = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'income',
           ]


    df = pd.read_csv('./Datasets/AdultCensusIncome/adult.data', header = None, skipinitialspace = True)
    df.columns = cols

    df.drop('fnlwgt', axis = 1, inplace = True)
    df = df.replace({'?': np.nan}).dropna()
    
    df.loc[:, 'income_over_50k'] = df.income == '>50K'
    df.drop('income', axis = 1, inplace = True)

    cat_cols = ['workclass',
                'education',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'native-country',
               ]
    
    enc = OneHotEncoder()

    df_cat = pd.DataFrame(enc.fit_transform(df.loc[:,cat_cols]).todense(), columns = enc.get_feature_names_out(cat_cols))
    df_cts = df.drop(cat_cols, axis = 1)

    df_out = pd.concat([df_cts, df_cat], axis = 1)
    df_out = df_out.dropna()

    y_name = 'income_over_50k'

    return (df_out.drop(y_name, axis = 1).astype(float), df_out.loc[:,y_name].astype(float))

    



#######################################
#           Misc. Utilities           #
#######################################

def get_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)






#######################################
#       Custom Loss & Metrics         #
#######################################


def exp_loss(l):
    def loss(y_true, y_pred):
        return exp(l * square(y_true - y_pred)) - 1
    
    return loss

def logsumexp_metric(l):
    def logsumexp(y_true, y_pred):
        return log(reduce_sum(exp(l * square(y_true - y_pred)) - 1, axis = -1)) / l
    
    return logsumexp






#######################################
#        Training Procedures          #
#######################################


def train_one_model(X, y, l, dataset,
                    prev_model = None, n_layers = 1,
                    stddev = 0.01, seed = 0,
                    full_batch = True
                    ):

    n, m = X.shape



    model = Sequential([Input(shape = (m,))])
    for i in range(n_layers):
        model.add(Dense(m, activation="relu", kernel_initializer=RandomNormal(stddev=stddev, seed = seed), bias_initializer=Zeros()))

    model.add(Dense(1, activation="sigmoid", kernel_initializer=RandomNormal(stddev=stddev, seed = seed), bias_initializer=Zeros()))


    if l > 0:
        model.compile(loss = exp_loss(l=l), optimizer = 'adam')
        callback = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 5, restore_best_weights = True)

    else:
        model.compile(loss = 'mse', optimizer = 'adam')
        callback = EarlyStopping(monitor = 'loss', patience = 10, restore_best_weights = True)


    if prev_model is not None:
        model.set_weights(prev_model.get_weights())


    if full_batch == True:
        batch_size = n
    else:
        batch_size = 128


    hist = model.fit(X, y, epochs = 500, batch_size = batch_size, callbacks = [callback], verbose = 0)


    y_pred = model.predict(X, verbose = 0).ravel()

    avg_err = np.mean((y_pred - y)**2)
    max_err = np.max((y_pred - y)**2)

    if dataset == 'COMMUNITIES':
        auc_val = np.nan
    else:
        auc_val = get_auc(y, y_pred)

    obj_val = model.evaluate(X, y, verbose = 0)

    return (model, y_pred, avg_err, max_err, auc_val, obj_val)




def sweep_ls(X, y, ls, 
             n_layers = 1,
             dataset = None, 
             seed = 0,
             stddev = 0.01,
             full_batch = True,
            ):

    cols = ['seed', 'avg_error', 'max_error', 'auc', 'avg_grp_error', 'max_grp_error', 'obj_val']

    df_stats = pd.DataFrame(columns = cols)


    prev_model = None

    for l in ls:

        prev_model, y_pred, avg_err, max_err, auc_val, obj_val = train_one_model(X.drop('errs', axis = 1, errors = 'ignore'), 
                                                                                 y, l, dataset,
                                                                                 prev_model = prev_model, n_layers = n_layers, 
                                                                                 stddev = stddev, seed = seed,
                                                                                 full_batch = full_batch,
                                                                                 )

        # Evaluate group error rates
        X.loc[:,'errs'] = (y - y_pred)**2
        
        if dataset == 'COMPAS':
            protected_col = 'race'
        
        elif dataset == 'BANK':
            protected_col = 'marital'

        elif dataset == 'ADULT':
            protected_col = 'race'

        elif dataset == 'DEFAULT':
            protected_col = 'MARRIAGE'

        elif dataset == 'COMMUNITIES':
            protected_col = 'PovertyPercent'

        else:
            raise ValueError('Unspecified Dataset.')


        grp_cols = [name for name in X.columns.values if protected_col in name]
        grp_errs = [X.loc[X.loc[:,col] == 1, 'errs'].mean() for col in grp_cols]

    
        stats_tmp = pd.DataFrame({'l': l, 'avg_error': avg_err, 'max_error': max_err, 'auc': auc_val,
                               'avg_grp_error': np.mean(grp_errs), 'max_grp_error': np.max(grp_errs),
                               'obj_val': obj_val}, index = [0])

        df_stats = pd.concat([df_stats, stats_tmp], axis = 0, ignore_index = True)

    return df_stats




def get_frontier(X, y, l_min, l_max, n_l, 
                 n_models = 20, 
                 n_layers = 1,
                 dataset = None,
                 stddev = 0.01,
                 include_min_mse = True,
                 full_batch = True,
                 ):
    
    cols = ['l', 'seed', 'avg_error', 'max_error', 'auc', 'avg_grp_error', 'max_grp_error', 'obj_val']
    
    df_out = pd.DataFrame(columns = cols)
    

    if include_min_mse:
        ls = list([0])
        ls.extend(np.logspace(l_min, l_max, n_l))
    else:
        ls = np.logspace(l_min, l_max, n_l)    


    for i in range(n_models):
        
        if i% 10 == 0:
            sys.stdout.write('\ni = '+str(i))
            sys.stdout.flush()

        df_stats = sweep_ls(X, y, ls,
                           n_layers = n_layers,
                           dataset = dataset,
                           seed = i,
                           stddev = stddev,
                           full_batch = full_batch,
                           )
        
        df_stats.loc[:,['seed']] = i
        out_tmp = df_stats.loc[:, cols]

        df_out = pd.concat([df_out, out_tmp], axis = 0, ignore_index = True)
    
    return df_out


