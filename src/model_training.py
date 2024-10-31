from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model
 
