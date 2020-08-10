# Static repository for the following paper


## Title: "Discovery of low-modulus Ti-Nb-Zr alloys based on machine learning and first-principles calculations"
### Authors: Salvador, Camilo; Zornio, Bruno; Miranda, Caetano
### For the full version, please visit: [doi](doi)

### Contents

- **supplement.py**: Functions to train RF or NN (3 layers) models
- **descriptors.csv**: List of matminer descriptors needed for each model
- **models.tar.xz**: Final models used to make the predictions
  - **RF_K.pkl**: RandomForestRegressor (best estimator), sklearn ver. 0.21
  - **RF_G.pkl**: RandomForestRegressor (best estimator), sklearn ver. 0.21
  - **NN_K.h5**: Keras/tensorflow neural network model
  - **NN_G.h5**: Keras/tensorflow neural network model

**Additional material** 
- **descriptors.xlsl**: The same as descriptors.csv, but in spreadsheet format
- **load.py**: Raw script to load and test the models
- **metisdb.json**: An featurized instance of the MaterialsProject database (Jul 2019)
