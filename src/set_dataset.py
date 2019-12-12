import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error, classification_report, confusion_matrix
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
from scipy import interp
from sklearn.preprocessing import MinMaxScaler, Normalizer
from yellowbrick.model_selection.importances import FeatureImportances
import scikitplot as skplt
import warnings
from ngboost.ngboost import NGBoost
from ngboost.distns import Bernoulli
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

warnings.filterwarnings("ignore")
# %matplotlib inline
# import importlib
# import set_dataset as k
# #importlib.reload(k)

# Dataframe
df = pd.read_csv('cleaned_df.csv')


# Get X and Y labels
def x_and_y(label_threshold):
    '''
    :param label_threshold: Threshold that separates most popular/successful games
    :return: X and y labels
    '''
    # Total Ratings
    df['total_ratings'] = df.positive_ratings + df.negative_ratings
    # Finding percentage by total rating
    df['weight_total_quantile']= pd.qcut(df['total_ratings'], [0,.25,.5,.70,.85,.99,1],
                                         labels=[0.20, 0.45, .63,.80, .92, 1]).astype(float)
    # Getting ratio of positives over total
    division = (df.positive_ratings / df.total_ratings)
    ## Weighting ratio by distribution of total ratings
    ratio_weighted = division * df.weight_total_quantile.astype(float)
    # Y == rating ratio weighted  > Threshold
    y = (ratio_weighted > label_threshold).astype(int)

    # Drop unnecessary columns on X
    x = df.drop(['appid', 'name', 'days_since', 'positive_ratings', 'negative_ratings', 'estimated_revenue', 'weight_total_quantile', 'total_ratings'], axis=1)
    x = x.drop(['release_year', 'average_playtime', 'median_playtime', 'release_years_ago'], axis=1)
    x = x.drop(["release_month_cos", "release_month_sin", 'price'], axis=1)
    # Log Reg coef close to 0
    #x = x.drop(['single-player', 'action', 'achievements', 'survival', 'release_may', 'fantasy'], axis=1)
    return x, y

# Balances dataset
def balance(x_train, y_train, method):
    '''
    Balance X and y training set into any method: Oversample, ADASYN, or SMOTE
    :param x_train: training set parameters
    :param y_train: training set label
    :param method: Either ADASYN() or SMOTE()
    :return: balanced x and y training set
    '''
    x_balance, y_balance = method.fit_sample(x_train, y_train)
    x_balance = pd.DataFrame(x_balance, columns=x_train.columns)
    return x_balance, y_balance

# Feature Engineering by LogReg logloss
def top_logreg_coef_l1(x, y, method, number):
    '''
    :param x: Full DB X shape(27k, 425)
    :param y: labels
    :param method: balance
    :return: filtered X DataFrame with less columns
    '''
    x_balance, y_balance = balance(x, y, method)
    # Train model
    model = LogisticRegression(penalty='l1')
    scaler = StandardScaler()
    x_balance = scaler.fit_transform(x_balance)
    fig, ax = plt.subplots(figsize=(30, 30))
    ax = FeatureImportances(model, relative=False)
    ax.fit(x_balance, y_balance)

    coefficients = pd.concat(
        [pd.DataFrame(x.columns, columns=["feature"]), pd.DataFrame(np.transpose(model.coef_), columns=["coef"])],
        axis=1)
    # Absolute values of coefficients and sort them in descending order
    log_coefficients = coefficients.copy()
    coefficients.coef = abs(coefficients.coef)
    coefficients = coefficients.sort_values(by="coef", ascending=False)
    pd.set_option('display.max_rows', 250)
    print("Displaying top 245 variables", coefficients)
    # Get mask of top coefficients, and apply back to df X
    lists = list(coefficients.head(number).feature)
    x_top_coefficients = x[x.columns[x.columns.isin(lists)]]
    return x_top_coefficients, y, coefficients

# Feature Engineering by LogReg logloss
def top_logreg_coef(x, y, method, number):
    '''
    :param x: Full DB X shape(27k, 425)
    :param y: labels
    :param method: balance
    :return: filtered X DataFrame with less columns
    '''
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)
    # Balance X and y train
    x_balance, y_balance = balance(x, y, method)
    # Train model
    model = LogisticRegression()
    fig, ax = plt.subplots(figsize=(30, 30))
    ax = FeatureImportances(model, relative=False)
    ax.fit(x_balance, y_balance)
    # plot feature importance
    #score = model.score(x_test, y_test)
    #print("Score: ", score)
    # Get coefficients of log reg
    coefficients = pd.concat(
        [pd.DataFrame(x.columns, columns=["feature"]), pd.DataFrame(np.transpose(model.coef_), columns=["coef"])],
        axis=1)
    # Absolute values of coefficients and sort them in descending order
    log_coefficients = coefficients.copy()
    coefficients.coef = abs(coefficients.coef)
    coefficients = coefficients.sort_values(by="coef", ascending=False)
    pd.set_option('display.max_rows', 250)
    print("Displaying top 245 variables", coefficients)
    # Get mask of top coefficients, and apply back to df X
    lists = list(coefficients.head(number).feature)
    x_top_coefficients = x[x.columns[x.columns.isin(lists)]]
    return x_top_coefficients, y, log_coefficients

# Feature Engineering by XGB Feature Importances
def top_xgb_importance(x, y, method, n_features):
    '''
    :param x: Reduced DB
    :param y: labels
    :param method: balance
    :param n_features: number of features to return
    :return: Filtered X db and labels
    '''
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
    # Balance X and y train
    x_balance, y_balance = balance(x, y, method)
    # Train model
    model = XGBClassifier()
    model.fit(x_balance, y_balance)
    # plot feature importance
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(8, 12)
    pyplot.show()
    #score = model.score(x_test, y_test)
    #print("XGB Score: ", score)

    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(x.columns, model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

    pd.set_option('display.max_rows', 100)
    # Get top Feature importance
    top_features_df = importance.sort_values(by='Gini-importance', ascending=False)
    top_features_df_tableau = top_features_df.copy().head(n_features)
    print(top_features_df.head(100))
    top_features = top_features_df.head(n_features).index
    x_top = x[top_features]
    return x_top, y, top_features_df_tableau

# Fits balanced db and returns prediction dictionary y_test, y_pred, y_prob
models = {'XGBoost': XGBClassifier(),
          'Light GBM': lgbm.LGBMClassifier(),
          'Logistic Regression': LogisticRegression(solver='lbfgs'),
          'Random Forest': RandomForestClassifier(),
          'Gaussian Naive Bayes': GaussianNB()}

############ KFOLD CV ############

# 10 Fold CV returns 1. Dict: Stratified K-fold labels (val, pred, prob), 2. List: Test set, 3. Dict: Models trained on K-fold
def fit_predict_KFold(x, y, models, method, threshold=0.5):
    '''
    :param x: all db parameters
    :param y: all db labels
    :param models: dictionary with models
    :param method: ADASYN() or SMOTE()
    :return:    1. model_scores: Dictionary with models and K-fold y labels (val, pred, prob)
                2. model_test: List with test set X and y. 20% of total db
                3. models_fitted: Dictionary with models and models trained on last Strat K-fold
    '''
    # Dictionary with Key = Models, Value = List of K-fold labels (val, pred, prob)
    model_scores = {} ###
    # Dict with Key = models, Values = models trained on last K-Fold
    models_fitted = {} ###
    # Save scale instance
    scaled_instance = Normalizer()
    scaled_instance.fit(x, y)

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # Iterate through dictionary with [models] = model-instance
    for model_name, model in models.items():
        # List of (val, pred, prob) K-fold
        cv_scores = []
        # Iterate through K - Folds indexes
        for train_index, val_index in kf.split(x, y):
            # Get Train and Validation set: X and y
            x_trained, x_val = x.iloc[train_index], x.iloc[val_index]
            y_trained, y_val = y.iloc[train_index].values, y.iloc[val_index].values
            # Balance Train set: X and y USING balance FUNCTION
            x_balance, y_balance = balance(x_trained, y_trained, method)
            if model_name in ['Stochastic Gradient Descent', 'K-Nearest Neighbors']:
                scale = Normalizer()
                x_balance[x_balance.columns] = scale.fit_transform(x_balance[x_balance.columns])
                x_val[x_val.columns] = scale.transform(x_val[x_val.columns])
                # Calibrate model
                model = CalibratedClassifierCV(model)
            model.fit(x_balance, y_balance)
            # Get Prediction and Probabilities of Validation Set
            #y_pred = model.predict(x_val)
            if model_name in ['Natural Gradient Boosting']:
                y_prob = model.pred_dist(x_val).prob
                y_pred = (y_prob > threshold).astype(int)
            else:
                y_prob = model.predict_proba(x_val)
                y_pred = (y_prob[:,1] > threshold).astype(int)
            # Append y Validation, Prediction, and Probabilities
            cv_scores.append([y_val, y_pred, y_prob])
            # Save last fold model into models_fitted
            models_fitted[model_name] = model
        # Save list of K-fold (val, pred, prob) values into dictionary
        model_scores[model_name] = cv_scores
    return model_scores, models_fitted, scaled_instance

# Get Dataframe with metrics of each model using K-fold Labels (val, pred, prob)
def metrics_KF(model_scores):
    '''
    :param model_scores: Dictionary with models and K-fold y labels (val, pred, prob)
    :return: DataFrame with metrics on val
    '''
    # Empty list with model, metrics
    metrics = {}
    # Iterate through models, K-fold labels
    for model, values in model_scores.items():
        # Lists of metrics from each K-fold
        precision_list = []
        recall_list = []
        fscore_list = []
        # Lists of FP and TP from K-folds
        false_positive_rate_list = []
        true_positive_rate_list = []
        # Iterate through each K-fold label (val, pred, prob)
        for fold in values:
            y_val, y_pred, y_prob = fold
            # Get Precision, Recall, F-score
            precision, recall, f_score, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(f_score)
            # Get ROC Curve
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_pred)
            false_positive_rate_list.append(false_positive_rate)
            true_positive_rate_list.append(true_positive_rate)
        # Get ROC AUC from mean of K-fold (FP and TP)
        roc_auc = auc(np.mean(false_positive_rate_list, axis=0), np.mean(true_positive_rate_list, axis=0))
        # Get Dict with Key = Model, Value = List of metrics
        metrics[model] = [np.mean(precision_list), np.mean(recall_list), np.mean(fscore_list), roc_auc]
    # Transform metrics dict into DataFrame
    metrics_t = round(pd.DataFrame(metrics).T, 2)
    metrics_t.columns = ['Precision', 'Recall', 'F-Score', 'ROC-AUC']
    print(metrics_t)
    return

# Get ROC and Precision-Recall Curves on Kfold
def curves_KF(model_scores, models_fitted):
    '''
      :param model_test: List with Test Set: X, y
      :param models_fitted: Dict with models and models fitted
      :return: Print: 1. DataFrame with model metrics on Test set
                      2. Plot Precision-Recall Curve
                      3. Plot ROC Curve
      '''
    plt.figure(dpi=100)
    # Iterate through models, K-fold labels
    for model, values in model_scores.items():
        # Precision - Recall
        tprs = []

        base_fpr = np.linspace(0,1,100)
        for fold in values:
            y_val, y_pred, y_prob = fold
            if model in ['Natural Gradient Boosting']:
                ## Get metrics from Validation set
                # Get Precision, Recall curves
                precision_fold, recall_fold, thresh = precision_recall_curve(y_val, y_prob)
                # reverse order of results
                precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
                tpr = interp(base_fpr, recall_fold, precision_fold)
                # tpr[0] = 1
                tprs.append(tpr)
            else:
                ## Get metrics from Validation set
                # Get Precision, Recall curves
                precision_fold, recall_fold, thresh = precision_recall_curve(y_val, y_prob[:, 1])
                # reverse order of results
                precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
                tpr = interp(base_fpr, recall_fold, precision_fold)
                #tpr[0] = 1
                tprs.append(tpr)
        tprs = np.array(tprs)
        # Mean Precision Recall curve
        mean_tprs = tprs.mean(axis=0)
        # Std
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        plt.plot(base_fpr, mean_tprs)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(models_fitted.keys())
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision - Recall Curve")
    plt.plot([1, 0], [0, 1], c='violet', ls='--')

    plt.figure(dpi=100)
    # Iterate through models, K-fold labels
    for model, values in model_scores.items():
        # Precision - Recall
        tprs = []
        base_fpr = np.linspace(0, 1, 100)
        for fold in values:
            y_val, y_pred, y_prob = fold
            if model in ['Natural Gradient Boosting']:
                ## Get metrics from Validation set
                # Get Precision, Recall curves
                false_positive_rate, true_positive_rate, _ = roc_curve(y_val, y_prob)
                tpr = interp(base_fpr, false_positive_rate, true_positive_rate)
                tpr[0] = 0.0
                tprs.append(tpr)
            else:
                ## Get metrics from Validation set
                # Get Precision, Recall curves
                false_positive_rate, true_positive_rate, _ = roc_curve(y_val, y_prob[:, 1])
                tpr = interp(base_fpr, false_positive_rate, true_positive_rate)
                tpr[0] = 0.0
                tprs.append(tpr)
        tprs = np.array(tprs)
        # Mean Precision Recall curve
        mean_tprs = tprs.mean(axis=0)
        # Std
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        plt.plot(base_fpr, mean_tprs)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(models_fitted.keys())
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve');
    plt.plot([0, 1], [0, 1], c='violet', ls='--')
    return


# CV Using only one model
def KF(x, y, model, method):
    '''
    :param x: all db parameters
    :param y: all db labels
    :param model: Model with set HyperParameters
    :param method: ADASYN(), SMOTE(), or RandomOverSampler()
    :return: list - cv_scores: [y_val, y_pred, y_prob]
    '''
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # List of (val, pred, prob) K-fold
    cv_scores = []
    # Iterate through K - Folds indexes
    for train_index, val_index in kf.split(x, y):
        # Get Train and Validation set: X and y
        x_trained, x_val = x.iloc[train_index], x.iloc[val_index]
        y_trained, y_val = y.iloc[train_index].values, y.iloc[val_index].values
        # Balance Train set: X and y USING balance FUNCTION
        x_balance, y_balance = balance(x_trained, y_trained, method)
        model.fit(x_balance, y_balance)
        # Get Prediction and Probabilities of Validation Set
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)
        # Append y Validation, Prediction, and Probabilities
        cv_scores.append([y_val, y_pred, y_prob])
    return cv_scores

############ Train Test Split ############

# 10 Fold CV returns 1. Dict: Stratified K-fold labels (val, pred, prob), 2. List: Test set, 3. Dict: Models trained on K-fold
def fit_predict_train_test(x, y, models, method):
    '''
    :param x: all db parameters
    :param y: all db labels
    :param models: dictionary with models
    :param method: ADASYN() or SMOTE()
    :return:    1. model_scores: Dictionary with models and K-fold y labels (val, pred, prob)
                2. model_test: List with test set X and y. 20% of total db
                3. models_fitted: Dictionary with models and models trained on last Strat K-fold
    '''
    # Dictionary with Key = Models, Value = List of K-fold labels (val, pred, prob)
    model_scores = {} ###
    # Dict with Key = models, Values = models trained on last K-Fold
    models_fitted = {} ###

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
    # Iterate through dictionary with [models] = model-instance
    for model_name, model in models.items():
        # List of (val, pred, prob) K-fold
        # Balance Train set: X and y USING balance FUNCTION
        x_balance, y_balance = balance(x_train, y_train, method)
        # if model_name in ['Stochastic Gradient Descent', 'K-Nearest Neighbors']:
        #     scale = Normalizer()
        #     x_balance[x_balance.columns] = scale.fit_transform(x_balance[x_balance.columns])
        #     x_val[x_val.columns] = scale.transform(x_val[x_val.columns])
        #     # Calibrate model
        #     model = CalibratedClassifierCV(model)
        model.fit(x_balance, y_balance)
        # Get Prediction and Probabilities of Validation Set
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)
        # Append y Validation, Prediction, and Probabilities
        cv_score = [y_test, y_pred, y_prob]
        # Save model into models_fitted
        models_fitted[model_name] = model
        # Save (val, pred, prob) values into dictionary
        model_scores[model_name] = cv_score
    return model_scores, models_fitted

def metrics_train_test(model_scores):
    '''
    :param model_scores: Dictionary with models and K-fold y labels (val, pred, prob)
    :return: DataFrame with metrics on val
    '''
    # Empty list with model, metrics
    metrics = {}
    # Iterate through models, K-fold labels
    for model, values in model_scores.items():
        # Lists of metrics from each K-fold
        precision_list = []
        recall_list = []
        fscore_list = []
        # Lists of FP and TP from K-folds
        false_positive_rate_list = []
        true_positive_rate_list = []
        # Iterate through each K-fold label (val, pred, prob)

        y_val, y_pred, y_prob = values
        # Get Precision, Recall, F-score
        precision, recall, f_score, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)
        # Get ROC Curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_pred)
        false_positive_rate_list.append(false_positive_rate)
        true_positive_rate_list.append(true_positive_rate)
        # Get ROC AUC from mean of K-fold (FP and TP)
        roc_auc = auc(np.mean(false_positive_rate_list, axis=0), np.mean(true_positive_rate_list, axis=0))
        # Get Dict with Key = Model, Value = List of metrics
        metrics[model] = [np.mean(precision_list), np.mean(recall_list), np.mean(fscore_list), roc_auc]
    # Transform metrics dict into DataFrame
    metrics_t = round(pd.DataFrame(metrics).T, 2)
    metrics_t.columns = ['Precision', 'Recall', 'F-Score', 'ROC-AUC']
    print(metrics_t)
    return

def curves_train_test(model_scores, models_fitted):
    '''
      :param model_test: List with Test Set: X, y
      :param models_fitted: Dict with models and models fitted
      :return: Print: 1. DataFrame with model metrics on Test set
                      2. Plot Precision-Recall Curve
                      3. Plot ROC Curve
      '''
    plt.figure(dpi=100)

    for model, values in model_scores.items():
        # tprs = []
        # base_fpr = np.linspace(0, 1, 100)
        # y_test, y_pred, y_prob = values
        # precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, y_prob[:, 1])
        # # reverse order of results
        # precision_fold, recall_fold, thresh = precision_curve[::-1], recall_curve[::-1], threshold_curve[::-1]
        # tpr = interp(base_fpr, recall_fold, precision_fold)
        #
        # tpr[-1] = 0
        # tprs.append(tpr)
        # tprs = np.array(tprs)
        # # Mean Precision Recall curve
        # mean_tprs = tprs.mean(axis=0)
        # std_tprs = tprs.std(axis=0)
        # tprs_upper = np.minimum(base_fpr + std_tprs, 1)
        # tprs_lower = base_fpr - std_tprs
        # plt.plot(base_fpr, mean_tprs)
        # # plt.fill_between(mean_tprs, tprs_lower, tprs_upper, alpha=0.3)
        y_test, y_pred, y_prob = values
        precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, y_prob[:, 1])
        plt.plot(recall_curve[1:], precision_curve[1:], label='precision')
        plt.legend(models_fitted.keys())
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
    plt.plot([1, 0], [0, 1], c='violet', ls='--')

    plt.figure(dpi=100)
    # Iterate through models, K-fold labels
    for model, values in model_scores.items():
        # Precision - Recall
        tprs = []
        base_fpr = np.linspace(0, 1, 100)
        y_val, y_pred, y_prob = values
        # Get Precision, Recall curves
        false_positive_rate, true_positive_rate, _ = roc_curve(y_val, y_prob[:, 1])
        tpr = interp(base_fpr, false_positive_rate, true_positive_rate)
        tpr[0] = 0.0
        tprs.append(tpr)
        tprs = np.array(tprs)
        # Mean Precision Recall curve
        mean_tprs = tprs.mean(axis=0)
        # Std
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        plt.plot(base_fpr, mean_tprs)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(models_fitted.keys())
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve');
    plt.plot([0, 1], [0, 1], c='violet', ls='--')
    return

def confusion_train_test(model_scores):
    for model, values in model_scores.items():
        y_test, y_pred, y_prob = values
        matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax = plt.subplot()
        ax = sns.heatmap(matrix, annot=True, annot_kws={"size": 16}, ax=ax, cmap="Blues", fmt="d")
        # labels, title and ticks
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.tick_params(labelsize=15)
        ax.set_xlabel("Predicted values", fontsize=20)
        ax.set_ylabel("True values", fontsize=20)
        plt.title('{} Confusion Matrix'.format(model), fontsize=20)
        ax.xaxis.set_ticklabels(['Average', 'Success']);
        ax.yaxis.set_ticklabels(['Average', 'Success'])
    return








############ Unused Functions ############

# Plot AUC
def metrics_df(model_test, models_fitted, scaled_instance):
    '''
    :param model_test: List with Test Set: X, y
    :param models_fitted: Dict with models and models fitted
    :return: DataFrame with model metrics on Test set
    '''
    x_test, y_test = model_test

    metrics = {}
    precision_list = []
    recall_list = []
    f_score_list = []
    false_positive_rate_list = []
    true_positive_rate_list = []
    # Iterate through each fitted model
    for model_name, model_fitted in models_fitted.items():
        # Standardize for scale sensitive models
        if model_name in ['Stochastic Gradient Descent', 'K-Nearest Neighbors']:
            # Transform x_test
            x_test[x_test.columns] = scaled_instance.transform(x_test[x_test.columns])
        y_pred = model_fitted.predict(x_test)
        # Precision Recall scores
        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        precision_list.append(precision)
        recall_list.append(recall)
        f_score_list.append(f_score)
        # ROC CURVE
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        false_positive_rate_list.append(false_positive_rate)
        true_positive_rate_list.append(true_positive_rate)
        # Get ROC AUC from mean of K-fold (FP and TP)
        roc_auc = auc(np.mean(false_positive_rate_list, axis=0), np.mean(true_positive_rate_list, axis=0))
        # Get Dict with Key = Model, Value = List of metrics
        metrics[model_name] = [np.mean(precision), np.mean(recall), np.mean(f_score), roc_auc]
    metrics_t = round(pd.DataFrame(metrics).T, 2)
    metrics_t.columns = ['Precision', 'Recall', 'F-Score', 'ROC-AUC']
    return metrics_t

# LOT PR CURVES ON TEST SET
def curves_test(model_test, models_fitted, scaled_instance):
    '''
    :param model_test: List with Test Set: X, y
    :param models_fitted: Dict with models and models fitted
    :return: Print: 1. DataFrame with model metrics on Test set
                    2. Plot Precision-Recall Curve
                    3. Plot ROC Curve
    '''

    # Get test x and y
    x_test, y_test = model_test
    # Plot Metrics DF
    print(metrics_df(model_test, models_fitted, scaled_instance))

    # Plot Precision Recall Curve on Test set
    plt.figure(dpi=100)
    for model_name, model_fitted in models_fitted.items():
        if model_name in ['Stochastic Gradient Descent', 'K-Nearest Neighbors']:
            # Transform x_test
             x_test[x_test.columns] = scaled_instance.transform(x_test[x_test.columns])
        y_prob = model_fitted.predict_proba(x_test)
        precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, y_prob[:, 1])
        plt.plot(recall_curve[1:], precision_curve[1:], label='precision')
        plt.legend(models_fitted.keys())
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
    # Plot ROC Curve on Test Set
    plt.figure(dpi=100)
    for model_name, model_fitted in models_fitted.items():
        if model_name in ['Stochastic Gradient Descent', 'K-Nearest Neighbors']:
            # Transform x_test
            x_test[x_test.columns] = scaled_instance.transform(x_test[x_test.columns])
        y_prob = model_fitted.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        plt.plot(fpr, tpr, lw=2)
        plt.legend(models_fitted.keys())
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve');
    plt.plot([0, 1], [0, 1], c='violet', ls='--')
    return

def confusion(model_output):
    for model, values in model_output.items():
        y_test, y_pred, y_prob = values
        matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax = plt.subplot()
        ax = sns.heatmap(matrix, annot=True, annot_kws={"size": 16}, ax=ax, cmap="Blues", fmt="d")
        # labels, title and ticks
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.tick_params(labelsize=15)
        ax.set_xlabel("Predicted values", fontsize=20)
        ax.set_ylabel("True values", fontsize=20)
        plt.title('{} Confusion Matrix'.format(model), fontsize=20)
        ax.xaxis.set_ticklabels(['Average', 'Success']);
        ax.yaxis.set_ticklabels(['Average', 'Success'])
    return

def metrics(y_test, y_pred, y_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("ROC_AUC: ", roc_auc)
    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('TN, FP, FN, TP: ', tn, fp, fn, tp)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax= plt.subplot()
    ax = sns.heatmap(matrix, annot=True, annot_kws={"size": 16}, ax=ax, cmap="Blues", fmt="d")
    # labels, title and ticks
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.tick_params(labelsize=15)
    ax.set_xlabel("Predicted values", fontsize=20)
    ax.set_ylabel("True values", fontsize=20)
    plt.title('Confusion Matrix', fontsize=20)
    ax.xaxis.set_ticklabels(['Average', 'Success']); ax.yaxis.set_ticklabels(['Average', 'Success'])
    ### REPORT
    report = classification_report(y_test, y_pred, target_names=['Average', 'Success'], output_dict=True)
    print(pd.DataFrame(report).transpose())
    ### ROC CURVE
    # plt.title("ROC Curve", fontsize=15)
    skplt.metrics.plot_roc(y_test, y_prob, figsize=(7,7), title_fontsize='large', text_fontsize='large')
    plt.show()
    ### PRECISION RECALL CURVE WITH PROBS
    skplt.metrics.plot_precision_recall(y_test, y_prob, figsize=(7,7), title_fontsize='large', text_fontsize='large')
    plt.show()
    return

#def metrics_df(model_output):
    metrics = {}
    for model, values in model_output.items():
        y_test, y_pred, y_prob = values
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        #AUC
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        metrics[model] = [precision, recall, fscore, roc_auc]
    metrics_t = round(pd.DataFrame(metrics).T,2)
    metrics_t.columns = ['Precision', 'Recall', 'Fscore', 'ROC-AUC']
    print(metrics_t)
    ## PRECISION / RECALL
    plt.figure(dpi=100)
    for model, values in model_output.items():
        y_test, y_pred, y_prob = values
        precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, y_prob[:, 1])
        plt.plot(recall_curve[1:], precision_curve[1:], label='precision')
        plt.legend(model_output.keys())
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
    plt.figure(dpi=100)
    for model, values in model_output.items():
        y_test, y_pred, y_prob = values
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        plt.plot(fpr, tpr, lw=2)
        plt.legend(model_output.keys())
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve');
    plt.plot([0, 1], [0, 1], c='violet', ls='--')
    return

def xgb(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)
    model = XGBClassifier()
    model.fit(x_train, y_train)
    # plot feature importance
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(8, 12)
    pyplot.show()
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    print("CV Score: ", np.mean(cross_val_score(model, x_test, y_test, cv=10)))
    print("XGB Score: ", score)
    print("MSE: ", mean_squared_error(y_test, y_pred))
    return y_test, y_pred, y_prob

def xgb_smote(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_smoted, y_smoted = SMOTE().fit_sample(x_train, y_train)
    x_smoted = pd.DataFrame(x_smoted, columns=x_train.columns)
    model = XGBClassifier()
    model.fit(x_smoted, y_smoted)
    # plot feature importance
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(8, 12)
    pyplot.show()
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    print("CV Score: ", np.mean(cross_val_score(model, x_test, y_test, cv=10)))
    print("XGB Score: ", score)
    print("MSE: ", mean_squared_error(y_test, y_pred))
    return y_test, y_pred, y_prob

def xgb_adasyn(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_adasyn, y_adasyn = ADASYN().fit_sample(x_train, y_train)
    x_adasyn = pd.DataFrame(x_adasyn, columns=x_train.columns)
    model = XGBClassifier()
    model.fit(x_adasyn, y_adasyn)
    # plot feature importance
    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(8, 12)
    pyplot.show()
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    print("CV Score: ", np.mean(cross_val_score(model, x_test, y_test, cv=10)))
    print("XGB Score: ", score)
    print("MSE: ", mean_squared_error(y_test, y_pred))
    return y_test, y_pred, y_prob

def logreg_adasyn(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    x_adasyn, y_adasyn = ADASYN().fit_sample(x, y)
    x_adasyn = pd.DataFrame(x_adasyn, columns=x_train.columns)
    model = LogisticRegression()
    fig, ax = plt.subplots(figsize=(80, 80))
    ax = FeatureImportances(model, relative=False)
    ax.fit(x_adasyn, y_adasyn)
    # plot feature importance
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    #score = model.score(x_test, y_test)
    return y_test, y_pred, y_prob

def logreg_smote(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    x_adasyn, y_adasyn = SMOTE().fit_sample(x, y)
    x_adasyn = pd.DataFrame(x_adasyn, columns=x_train.columns)
    model = LogisticRegression()
    fig, ax = plt.subplots(figsize=(80, 80))
    ax = FeatureImportances(model, relative=False)
    ax.fit(x_adasyn, y_adasyn)
    # plot feature importance
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    #score = model.score(x_test, y_test)
    return y_test, y_pred, y_prob