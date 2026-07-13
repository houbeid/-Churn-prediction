import numpy as np
import pandas as pd
import os
from preprocessor import DataPreprocessor
from models.random_forest import RandomForestModel
from models.mlp_sklearn import MLPSklearnModel
from models.mlp_keras import MLPKerasModel
from models.mlp_tensorflow import MLPTensorFlowModel
from models.mlp_numpy import MLPNumpyModel
from sklearn.metrics import roc_auc_score, accuracy_score

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print('========== CHARGEMENT ==========')
train    = pd.read_csv('data/bank_data_train.csv')
test     = pd.read_csv('data/bank_data_test.csv')
test_ids = test['ID'].values

print(f'Train : {train.shape[0]} lignes, {train.shape[1]} colonnes')
print(f'Test  : {test.shape[0]} lignes,  {test.shape[1]} colonnes')

# ============================================================
# 2. PREPROCESSING COMPLET
# ============================================================
preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, X_test = preprocessor.fit_transform(train, test)

print(f'\nVerification NaN :')
print(f'  X_train : {np.isnan(X_train).sum()}')
print(f'  X_val   : {np.isnan(X_val).sum()}')
print(f'  X_test  : {np.isnan(X_test).sum()}')

# ============================================================
# 3. BASELINE NAÏVE
# ============================================================
print('\n========== BASELINE NAIVE ==========')
majority_class   = int(np.bincount(y_train.astype(int)).argmax())
y_baseline       = np.full(len(y_val), majority_class)
y_baseline_proba = np.full(len(y_val), float(majority_class))

baseline_acc = accuracy_score(y_val, y_baseline)
baseline_auc = roc_auc_score(y_val, y_baseline_proba)
print(f'  Classe majoritaire : {majority_class}')
print(f'  Accuracy           : {baseline_acc:.4f}')
print(f'  AUC                : {baseline_auc:.4f}')

# ============================================================
# 4. RANDOM FOREST
# ============================================================
rf_model = RandomForestModel(random_state=42)
rf_model.search(X_train, y_train, n_iter=20, cv=3)
rf_model.train(X_train, y_train)
rf_acc, rf_auc = rf_model.evaluate(X_val, y_val)

os.makedirs('predictions', exist_ok=True)
pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': rf_model.predict_proba(X_test)
}).to_csv('predictions/random_forest.csv', index=False)
print('  Predictions → predictions/random_forest.csv')

# ============================================================
# 5. MLP SCIKIT-LEARN
# ============================================================
mlp_sklearn = MLPSklearnModel(random_state=42)
mlp_sklearn.search(X_train, y_train, n_iter=10, cv=3)
mlp_sklearn.train(X_train, y_train)
mlp_acc, mlp_auc = mlp_sklearn.evaluate(X_val, y_val)

pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': mlp_sklearn.predict_proba(X_test)
}).to_csv('predictions/mlp_sklearn.csv', index=False)
print('  Predictions → predictions/mlp_sklearn.csv')

# ============================================================
# 6. MLP KERAS
# ============================================================
mlp_keras = MLPKerasModel(
    input_dim    = X_train.shape[1],
    random_state = 42
)
mlp_keras.train(X_train, y_train, X_val, y_val)
keras_acc, keras_auc = mlp_keras.evaluate(X_val, y_val)

pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': mlp_keras.predict_proba(X_test)
}).to_csv('predictions/mlp_keras.csv', index=False)
print('  Predictions → predictions/mlp_keras.csv')

# ============================================================
# 7. MLP TENSORFLOW
# ============================================================
mlp_tf = MLPTensorFlowModel(
    input_dim    = X_train.shape[1],
    random_state = 42
)
mlp_tf.train(X_train, y_train, X_val, y_val)
tf_acc, tf_auc = mlp_tf.evaluate(X_val, y_val)

pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': mlp_tf.predict_proba(X_test)
}).to_csv('predictions/mlp_tensorflow.csv', index=False)
print('  Predictions → predictions/mlp_tensorflow.csv')

# ============================================================
# 8. MLP NUMPY (from scratch)
# ============================================================
mlp_numpy = MLPNumpyModel(
    input_dim    = X_train.shape[1],
    random_state = 42
)
mlp_numpy.train(X_train, y_train, X_val, y_val)
numpy_acc, numpy_auc = mlp_numpy.evaluate(X_val, y_val)

pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': mlp_numpy.predict_proba(X_test)
}).to_csv('predictions/mlp_numpy.csv', index=False)
print('  Predictions → predictions/mlp_numpy.csv')

# ============================================================
# 9. TABLEAU RÉCAPITULATIF COMPLET
# ============================================================

# Construire les strings d hyperparamètres
rf_params_str = (
    f"n_est={rf_model.best_params.get('n_estimators')} "
    f"depth={rf_model.best_params.get('max_depth')} "
    f"split={rf_model.best_params.get('min_samples_split')} "
    f"leaf={rf_model.best_params.get('min_samples_leaf')} "
    f"feat={rf_model.best_params.get('max_features')} "
    f"cw=balanced"
)

mlp_params_str = (
    f"layers={mlp_sklearn.best_params.get('hidden_layer_sizes')} "
    f"act={mlp_sklearn.best_params.get('activation')} "
    f"alpha={mlp_sklearn.best_params.get('alpha')} "
    f"lr={mlp_sklearn.best_params.get('learning_rate_init')}"
)

keras_params_str = (
    f"layers=[128,64] dropout=0.3 "
    f"lr=0.001 batch=512 "
    f"optimizer=Adam epochs=50"
)

tf_params_str = (
    f"layers=[128,64] dropout=0.3 "
    f"lr=0.001 batch=512 "
    f"optimizer=Adam epochs=50"
)

numpy_params_str = (
    f"layers=[128,64] dropout=0.3 "
    f"lr=0.001 batch=512 "
    f"optimizer=Adam epochs=100"
)

baseline_params_str = f"classe majoritaire = {majority_class}"

# Affichage du tableau
W = 110
print('\n' + '=' * W)
print(f'{"Modele":<18} {"Bibliotheque":<14} {"Hyperparametres":<55} {"Accuracy":<10} {"AUC":<8}')
print('-' * W)
print(f'{"Baseline":<18} {"Python":<14} {baseline_params_str:<55} {baseline_acc:<10.4f} {baseline_auc:<8.4f}')
print(f'{"Random Forest":<18} {"Scikit-learn":<14} {rf_params_str:<55} {rf_acc:<10.4f} {rf_auc:<8.4f}')
print(f'{"MLP":<18} {"Scikit-learn":<14} {mlp_params_str:<55} {mlp_acc:<10.4f} {mlp_auc:<8.4f}')
print(f'{"MLP":<18} {"Keras":<14} {keras_params_str:<55} {keras_acc:<10.4f} {keras_auc:<8.4f}')
print(f'{"MLP":<18} {"TensorFlow":<14} {tf_params_str:<55} {tf_acc:<10.4f} {tf_auc:<8.4f}')
print(f'{"MLP (from scratch)":<18} {"NumPy":<14} {numpy_params_str:<55} {numpy_acc:<10.4f} {numpy_auc:<8.4f}')
print('=' * W)

# ============================================================
# 10. MEILLEUR MODÈLE — FICHIER DE SOUMISSION FINAL
# ============================================================
results = {
    'Random Forest': (rf_auc,    rf_model.predict_proba(X_test)),
    'MLP Sklearn'  : (mlp_auc,   mlp_sklearn.predict_proba(X_test)),
    'MLP Keras'    : (keras_auc, mlp_keras.predict_proba(X_test)),
    'MLP TF'       : (tf_auc,    mlp_tf.predict_proba(X_test)),
    'MLP NumPy'    : (numpy_auc, mlp_numpy.predict_proba(X_test)),
}

best_name  = max(results, key=lambda k: results[k][0])
best_proba = results[best_name][1]

print(f'\nMeilleur modele : {best_name}')
print(f'Meilleure AUC   : {results[best_name][0]:.4f}')

# Fichier final — ordre des IDs identique au fichier test original
submission = pd.DataFrame({
    'ID'    : test_ids,
    'TARGET': best_proba
})
submission.to_csv('predictions/predictions.csv', index=False)
print('\n predictions/predictions.csv sauvegarde !')
print(f'   {len(submission)} lignes | ordre IDs : identique au fichier test')
print(submission.head(5))