import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class CovariateController(BaseEstimator, TransformerMixin):
    """
    Remove (control for) the effect of covariates

    CovariateController controls for the effect of covariates (commonly called
    confounding variables) using linear regression. More specifically, it is
    hypothesized that if there is an effect, it can removed by subtracting the
    predictions made by the linear model trained as X = f(c), i.e. the features
    without the effect of covariates can be expressed as the residuals of such
    model.

    Implemented as a Scikit-learn transformer. Can be fitted on 1D or xD data
    as there can be multiple covariates having an effect on the variables in
    X (features). It assumes that data pre-processing such as feature scaling
    or normalization is done prior using CovariateController.

    The procedure of transforming a feature can be defined as follows:

      [1.]      [2.]              [3.]                            [4.]
    feature = feature - regressor.fit(covariates, feature).predict(covariates)

    [1.] updated feature (residuals of the model)
    [2.] original feature
    [4.] predictions
    [3.] trained linear model feature = f(covariates)

    Parameters
    ----------

    inline : bool, optional, default False
        Parameter that specifies if the transformation should be performed
        inline, i.e. if it is feasible to alter the original DataFrame

    Attributes
    ----------

    regressors : dict, len(models) == len(X.columns) (after fitted)
        Dictionary with the fitted linear regression models for each column
        in X (each column represents a feature). After __init__(), the dict
        is empty. After fit(): {"f1": model, "f2": model, "f3": model, ...}

    covariates : numpy array
        Numpy array storing the covariates used to fit the linear regression
        models. After __init__(), the covariates are unknown and therefore
        the variable is None, After fit(), covariates.shape corresponds to
        the number of observations (rows) and features (cols) used to fit
        the models (model per input feature vector).
    """

    def __init__(self, inline=False):
        # Parse input arguments
        self.inline = inline

        # Prepare instance attributes
        self.regressors = {}
        self.covariates = None

    def fit(self, X, y, **params):
        """
        Fit the transformer

        This method uses the covariates to fit the linear regression models according
        to: features = f(covariates) (model per feature; column of X), i.e. it fits
        the models to capture the (linear) relationship that the covariates and each
        of the features have. The regressor can be modified using input **params.

        Parameters
        ----------

        X : numpy array
            1D or 2D array of features (rows=observations, cols=features)

        y : numpy array
            1D or 2D array of covariates (rows=observations, cols=covariates)
        """

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert X.shape[0] == y.shape[0]

        # Fit the linear regressors (feature = f(covariates))
        self.regressors = {c: LinearRegression(**params).fit(y.values, X[c].values) for c in X.columns}

        # Store the covariates
        self.covariates = y

        return self

    def transform(self, X):
        """
        Transform the data

        This method uses the fitted linear regression models to predict the values of
        the features (columns of X) and replace each of the features by the residuals
        of such model, i.e. feature = feature - feature_hat. With this approach, the
        new feature values are transformed in a way it removes the (linear) effect
        of the covariates on the features.

        Parameters
        ----------

        X : numpy array
            1D or 2D array of features (rows=observations, cols=features)

        Returns
        -------

        Feature matrix without the effect of covariates (numpy array)
        """

        assert isinstance(X, pd.DataFrame)

        # Get the DataFrame to work with
        x = X.copy() if not self.inline else X

        # Remove the effect of covariates
        for c in x.columns:
            x[c] = x[c].values - self.regressors.get(c).predict(self.covariates.values)

        return x
