import numpy as np

def manualOrdinal(input_val,map_dictionary):
    return map_dictionary[input_val]


def frequencyEncoding(X):
    tov=X.groupby('type_of_vehicle').size()/len(X)
    aao=X.groupby('area_accident_occured').size()/len(X)
    toc=X.groupby('type_of_collision').size()/len(X)

    X.loc[:,'type_of_vehicle']=X['type_of_vehicle'].map(tov)
    X.loc[:,'area_accident_occured']=X['area_accident_occured'].map(aao)
    X.loc[:,'type_of_collision']=X['type_of_collision'].map(toc)
    
    return X
def ordinal(input_val,feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value

def get_prediction(data,model):
    pred=model.predict(data)
    return pred