import numpy as np
import pandas as pd
import os
from preprocessor import DataPreprocessor

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print('========== CHARGEMENT ==========')
train = pd.read_csv('data/bank_data_train.csv')
test  = pd.read_csv('data/bank_data_test.csv')

print(f'Train : {train.shape[0]} lignes, {train.shape[1]} colonnes')
print(f'Test  : {test.shape[0]} lignes,  {test.shape[1]} colonnes')

# ============================================================
# 2. PREPROCESSING COMPLET
# ============================================================
preprocessor = DataPreprocessor(test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val, X_test = preprocessor.fit_transform(train, test)

# # ============================================================
# # 3. SAUVEGARDE (optionnel)
# # ============================================================
# os.makedirs('data', exist_ok=True)
# np.save('data/X_train.npy', X_train)
# np.save('data/X_val.npy',   X_val)
# np.save('data/y_train.npy', y_train)
# np.save('data/y_val.npy',   y_val)
# np.save('data/X_test.npy',  X_test)

print('\n========== SAUVEGARDE ==========')
print(' data/X_train.npy')
print(' data/X_val.npy')
print(' data/y_train.npy')
print(' data/y_val.npy')
print(' data/X_test.npy')
print('\n Prêt pour les modèles !')