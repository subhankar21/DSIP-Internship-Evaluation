import numpy as np
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def date_time(columns_names, df):
    '''
    Prepare the data types into datetime objects.

    Args:
    columns_names: Names of the columns to convert into datetime objects
    df: The pandas dataframe

    Returns:
    df: Pandas dataframe
    '''
    for col in columns_names:
        df[col] = pd.to_datetime(df[col])
    return df

def frequency_encoding (df, column_names):
    '''
    Prepare the encoded dataframe using frequency encoding.

    Args:
    df: The pandas dataframe
    columns_names: Names of the columns to convert into datetime objects

    Returns:
    df_encoded: Pandas dataframe
    '''
    for col in column_names:
        freq_encoding = df.groupby(col).size() / len(df)
        df[col + '_freq'] = df[col].map(freq_encoding)
        df_encoded = df.drop(columns=column_names)
    return df_encoded

def train_test_prep(X, y, scale):
    '''
    Prepares the train and test datasets and ptionally scales the features if scaling is required.

    Args:
    X: Features dataset
    y: Target dataset
    scale: Boolean indicating whether to scale the features using StandardScaler

    Returns:
    X_train, X_test, y_train, y_test: splitted train and test data with or without scaled
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if scale == True:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test


def random_forest_cv(X, y, n_estimators=100, max_depth=None, min_samples_split=2, 
                     min_samples_leaf=1, cv=5, scoring='accuracy'):
    """
    Perform cross-validation with Random Forest.
    
    Args:
    X: The input samples.
    y: The target values.
    n_estimators: The number of trees in the forest.
    max_depth: The maximum depth of the tree.
    min_samples_split: The minimum number of samples required to split an internal node.
    min_samples_leaf: The minimum number of samples required to be at a leaf node.
    cv: Number of cross-validation folds.
    scoring: Scoring metric to use for cross-validation.
    
    Returns:
    float: Mean cross-validated score.
    rf: Random forest Model
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring)
    rf.fit(X, y)
    return np.mean(scores), rf


def gradient_boosting_cv(X, y, n_estimators=100, max_depth=3, learning_rate=0.1, cv=5, scoring='accuracy'):
    """
    Perform cross-validation with Gradient Boosting.
    
    Args:
    X: The input samples.
    y: The target values.
    n_estimators: The number of boosting stages to perform.
    max_depth: Maximum depth of the individual regression estimators.
    learning_rate: Learning rate shrinks the contribution of each tree by learning_rate.
    cv: Number of cross-validation folds.
    scoring: Scoring metric to use for cross-validation.
    
    Returns:
    tuple: (mean cross-validated score, trained model)
    gb: Gradient Boosting Model
    """
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    learning_rate=learning_rate, random_state=42)
    scores = cross_val_score(gb, X, y, cv=cv, scoring=scoring)
    gb.fit(X, y)
    return np.mean(scores), gb

def save_model(model, filename):
    """
    Save the model to a file.
    
    Args:
    model: The model to save
    filename: The name of the file to save the model to
    """
    dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load a model from a file.
    
    Args:
    filename: The name of the file to load the model from
    
    Returns:
    object: The loaded model
    """
    return load(filename)


def plot_feature_importance(model, features, model_name, top_n=10):
    """
    Plots the top N most important features from a trained model.

    Args:
    model: Trained model object with a `feature_importances_` attribute (e.g., RandomForest, GradientBoosting).
    features: List of feature names used in training the model.
    top_n: Number of top features to display (default is 10).

    Returns:
    A bar plot of the top N most important features.
    """
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})    
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    # Plotting the top N important features
    plt.figure(figsize=(10, 7))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(f'figs/{model_name}_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()    
    print(f"figs/Feature Importance for {model_name} saved as '{model_name}_importance.png'")