

import pandas as pd
from lifetimes import BetaGeoFitter
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


class MLRepurchase:
    def __init__(self, n_days: int) -> None:

        self.n_days = n_days

        self.clf = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=22,
        )

        self.clf_params = {
            "learning_rate": [0.01, 0.1],
        }


    def _fit(self, train_df: pd.DataFrame, target: str) -> None:

        self.X = train_df.drop(target, axis=1).copy()
        y = train_df[target].copy()
        skfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=22)

        self.clf_model = GridSearchCV(
            estimator=self.clf,
            param_grid=self.clf_params,
            scoring="roc_auc",
            refit=True,
            cv=skfold,
            verbose=4,
            n_jobs=-1,
        )

        self.clf_model.fit(self.X, y)


    def _predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        return self.clf_model.predict_proba(test_df[self.X.columns])


    def _get_training_evaluation(self) -> float:
        """Evaluation 採用 roc_auc 計算,範圍在 0 - 1,越接近 1 越好."""
        return self.clf_model.cv_results_["std_test_score"].mean()


    def _get_feature_importance(self) -> pd.DataFrame:
        imp_proba_df = pd.DataFrame(
            data = {
                "feature": list(self.clf_model.best_estimator_.feature_names_in_),
                "value": list(self.clf_model.best_estimator_.feature_importances_),
            },
        )
        return imp_proba_df.sort_values("value", ascending=False)


class BGPurchase:
    def __init__(self, n_days: int) -> None:
        self.n_days = n_days
        self.bg_model = BetaGeoFitter(penalizer_coef=10)


    def _fit(self, train_df: pd.DataFrame) -> None:
        self.bg_model.fit(train_df["frequency"], train_df["recency"], train_df["T"])


    def _predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        predictions = self.bg_model.conditional_expected_number_of_purchases_up_to_time(
            self.n_days, test_df["frequency"], test_df["recency"], test_df["T"],
        ).apply(
            lambda x: 1 if x>1 else x,
        )
        return predictions


class HLHRepurchase:
    """
    HLHRepurchase 類別用於進行客戶回購預測,包括訓練和預測兩個階段。可選擇使用機器學習模型或貝塔地幾何模型進行預測。
    屬性:
        n_days (int): 預測的天數
        repurchase_model (object): 客戶回購模型
    方法:
        fit(train_df: pd.DataFrame) -> None:
            訓練客戶回購模型,使用 train_df 訓練模型。
    repurchase_predict(test_df: pd.DataFrame) -> pd.DataFrame:
        預測客戶回購概率,使用 test_df 預測每位客戶在 n_days 內回購的概率。
        返回一個包含預測概率的 DataFrame。.
    """

    def _get_classifier(self, method: str) -> object:
        """
        根據輸入的方法選擇使用的客戶回購模型。.

        參數:
            method (str): 可選擇使用 "ml" 或 "bg" 方法

        返回:
            object: 客戶回購模型對象
        """
        if method == "ml":
            return MLRepurchase(n_days=self.n_days)
        else:
            return BGPurchase(n_days=self.n_days)


    def __init__(self, n_days: int, method: str) -> None:
        """
        初始化 HLHRepurchase 類別。.

        參數:
            n_days (int): 預測的天數
            method (str): 可選擇使用 "ml" 或 "bg" 方法
        """
        self.n_days = n_days
        self.repurchase_model = self._get_classifier(method=method)


    def fit(self, train_df: pd.DataFrame, target: str | None = None) -> None:
        """
        訓練客戶回購模型。.

        參數:
            train_df (pd.DataFrame): 訓練數據,包含客戶特徵和標籤
        """
        if target:
            self.repurchase_model._fit(train_df=train_df, target=target)
        else:
            self.repurchase_model._fit(train_df=train_df)


    def repurchase_predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        預測客戶回購概率。.

        參數:
            test_df (pd.DataFrame): 測試數據,包含客戶特徵

        返回:
            pd.DataFrame: 一個包含預測概率的 DataFrame。
        """
        purchase_predictions = self.repurchase_model._predict(test_df=test_df)
        return purchase_predictions
