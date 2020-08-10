#!/usr/bin/env python

'''load.py: Script to load and test the final models; calls supplement.py'''

__author__     = 'Camilo A. Fernandes Salvador'
__email__      = 'csalvador@usp.br'
__copyright__  = 'Copyright © 2020, University of Sao Paulo'

# This code is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 of 29 June 2007.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ATTENTION: please use sklearn.RandomForestRegressor ver. 0.21.2 
# ATTENTION: this script requires local files (supplement.py)


import pandas as pd
import numpy as np
import keras
import matminer
import pickle

from keras.models import load_model
from matminer.utils.io import load_dataframe_from_json
from sklearn.preprocessing import MinMaxScaler
from supplement import searchNN, trainNN, evaluateModel, createRF #stores custom methods/functions

fdf = load_dataframe_from_json('metisdb.json')
print("The starting dataset has {}".format(fdf.shape))
print (fdf.head())

'''
Block 1 - Random Forest, K
'''
print ('\n---\n')
print ('Results for K using RF')

excluded = ['material_id', 'spacegroup', 'structure',
            'elastic_anisotropy', 'G_VRH', 'poisson_ratio', 'elasticity',
            'formula', 'composition', 'composition_oxid',
            'HOMO_character', 'HOMO_element',
            'LUMO_character', 'LUMO_element']

# A few additional adjustments
fdf = fdf.drop(excluded, axis=1)
fdf = fdf.fillna(0)

not_ionic = fdf['compound possible'] == 0
fdf = fdf[not_ionic]

# Defining the target variable
target = 'K_VRH'
y = fdf[target].values
X = fdf.drop([target, 'compound possible'], axis=1)
print ('The target variable was stored in y: {}'.format(y))

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# How many descriptors were generated? 
print('There are {} possible descriptors:\n{}'.format(X.shape[1], X.columns.values))

model = pickle.load(open('RF_K.pkl', 'rb')) 

evaluateModel(model, X, y)
print ('Done 1')
print ('\n---\n')

'''
Block 2 - Random Forest, G
'''
print ('\n---\n')
print ('Results for G using RF')

fdf = load_dataframe_from_json('metisdb.json')
excluded = ['material_id', 'spacegroup', 'structure',
            'elastic_anisotropy', 'K_VRH', 'elasticity',
            'formula', 'composition', 'composition_oxid',
            'HOMO_character', 'HOMO_element',
            'LUMO_character', 'LUMO_element']

# A few additional adjustments
fdf = fdf.drop(excluded, axis=1)
fdf = fdf.fillna(0)

not_ionic = fdf['compound possible'] == 0
fdf = fdf[not_ionic]

# Defining the target variable
target = 'G_VRH'
y = fdf[target].values
X = fdf.drop([target, 'compound possible'], axis=1)
print ('The target variable was stored in y: {}'.format(y))

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# How many descriptors were generated? 
print('There are {} possible descriptors:\n{}'.format(X.shape[1], X.columns.values))

model = pickle.load(open('RF_G.pkl', 'rb')) 

evaluateModel(model, X, y)
print ('Done 2')
print ('\n---\n')

'''
Block 3 - Neural network, K
'''
print ('\n---\n')
print ('Results for K using NN')

fdf = load_dataframe_from_json('metisdb.json')
excluded = ['material_id', 'spacegroup', 'structure',
            'elastic_anisotropy', 'G_VRH', 'poisson_ratio', 'elasticity',
            'formula', 'composition', 'composition_oxid',
            'HOMO_character', 'HOMO_element',
            'LUMO_character', 'LUMO_element']

# A few additional adjustments
fdf = fdf.drop(excluded, axis=1)
fdf = fdf.fillna(0)

not_ionic = fdf['compound possible'] == 0
fdf = fdf[not_ionic]

# The selected variables
selected = ['mean GSvolume_pa', 'transition metal fraction', 'cohesive energy',
			'band center', 'mean MeltingT', 'mode GSvolume_pa', 'mean Electronegativity',
			'maximum GSvolume_pa', 'mean CovalentRadius', 'minimum MendeleevNumber',
			'minimum Column', 'minimum GSvolume_pa', 'avg_dev MeltingT', 'range MeltingT',
			'avg_dev CovalentRadius', 'minimum Electronegativity', # top 20 RF +
			'mean NfUnfilled', 'maximum CovalentRadius', 'avg_dev MendeleevNumber',
			'avg_dev NfUnfilled', 'maximum SpaceGroupNumber', 'maximum NfUnfilled',
			'range NfUnfilled', 'mean NsValence', 'avg s valence electrons', #additions from SVD
			'mode MendeleevNumber', 'avg_dev Column', 'maximum Electronegativity',
			'mode Number', 'mode AtomicWeight', 'maximum MeltingT', 'mean NdValence',
			'mode Column', 'mean Column', 'avg_dev NdValence', 'avg d valence electrons',
			'maximum AtomicWeight'] #additions from MID

# Defining the target variable
target = 'K_VRH'
y = fdf[target].values
X = fdf[selected]
print ('The target variable was stored in y: {}'.format(y))

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# How many descriptors were generated? 
print('There are {} possible descriptors:\n{}'.format(X.shape[1], X.columns.values))

model = load_model('NN_K.h5')

evaluateModel(model, X, y)
print ('Done 3')
print ('\n---\n')

'''
Block 4 - Neural network, G
'''
print ('\n---\n')
print ('Results for G using NN')

fdf = load_dataframe_from_json('metisdb.json')
excluded = ['material_id', 'spacegroup', 'structure',
            'elastic_anisotropy', 'K_VRH', 'elasticity',
            'formula', 'composition', 'composition_oxid',
            'HOMO_character', 'HOMO_element',
            'LUMO_character', 'LUMO_element']

# A few additional adjustments
fdf = fdf.drop(excluded, axis=1)
fdf = fdf.fillna(0)

not_ionic = fdf['compound possible'] == 0
fdf = fdf[not_ionic]

# Defining the target variable
target = 'G_VRH'
y = fdf[target].values
X = fdf.drop([target, 'compound possible'], axis=1)
print ('The target variable was stored in y: {}'.format(y))

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# How many descriptors were generated? 
print('There are {} possible descriptors:\n{}'.format(X.shape[1], X.columns.values))

model = load_model('NN_G.h5')

evaluateModel(model, X, y)
print ('Done 4')
print ('\n---\n')

#
# --- end
