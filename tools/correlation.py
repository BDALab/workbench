import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from common.exports import save_to_excel_multi


def compute_correlation(x, y, corr_type="spearman", supported_types=("pearson", "spearman", "kendall")):
    """
    Compute correlation between <x> and <y>

    This function computes the correlation between <x> and <y>. The correlation function
    is chosen using the mapping of the <supported_types> and <corr_type>. The function
    supports three types of correlation: ("pearson", "spearman", "kendall").

    Parameters
    ----------

    x : numpy ndarray
        numpy ndarray with the data

    y : numpy ndarray
        numpy ndarray with the data

    corr_type : str, optional, default "spearman"
        str with the correlation type

    supported_types : iterable, optional, default ("pearson", "spearman", "kendall")
        iterable with the supported correlation types

    Returns
    -------

    Correlation results: r, p
    """

    # Prepare the mapping
    mapping = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}

    # Compute the correlation
    if corr_type not in supported_types:
        raise ValueError(f"Corr. type {corr_type} is unsupported. Supported ones: {supported_types}")
    return mapping[corr_type.lower()](x, y)


def assess_correlation(df_feat, df_clin, computation_settings, save_as="tmp.xlsx"):
    """
    Assess the correlation between features and clinical rating scales according to the settings.

    This function computes the correlation between all features from <df_feat> and selected
    clinical rating scales from <df_clin> defined in <computation_settings>. The function
    supports three types of correlation: ("pearson", "spearman", "kendall"). The results
    can be stored to excel sheet, see: <save_as>. The following structure of the settings
    is anticipated:

        [
            {
                "scale": "class",
                "field_name": "class",
                "correlation": ("pearson", "spearman", "kendall")
            }
        ]

    Parameters
    ----------

    df_feat : pandas DataFrame with the features
        pandas DataFrame (rows=observations, cols=features)

    df_clin : pandas DataFrame with the clinical data
        pandas DataFrame (rows=observations, cols=features)

    computation_settings : list with the computation settings
        list of dicts with the computation settings

    save_as : str, optional, default "tmp.xlsx"
        str with the full-path to store the tables with the correlations
    """

    # Prepare the tables and sheets (each table and sheet pair stores the correlation for one scale)
    tables = []
    sheets = []

    # Compute the correlation between the features and clinical rating scales defined in the settings
    for setting in computation_settings:

        # Prepare the results table
        results = []

        for feat in df_feat:

            # Select the feature and clinical rating scale to analyze
            clin_data = df_clin[setting.get("field_name")]
            feat_data = df_feat[feat]

            # Select only observations with no missing values in both features and clinical scale
            clin_nans = clin_data.isnull()
            feat_nans = feat_data.isnull()
            with_nans = clin_nans | feat_nans

            clin_data = clin_data.values[np.logical_not(with_nans.values)]
            feat_data = feat_data.values[np.logical_not(with_nans.values)]

            # Prepare the dictionary of correlations
            correlations = {"feature": feat}

            # Compute the correlations and update the results
            for corr in setting.get("correlation"):
                r, p = compute_correlation(feat_data, clin_data, corr_type=corr)
                correlations.update({f"r ({corr})": round(float(r), 4)})
                correlations.update({f"p ({corr})": round(float(p), 4)})

            # Append to correlation the results
            results.append(correlations)

        # Update the resulting correlation tables and sheets
        tables.append(pd.DataFrame(results))
        sheets.append(setting.get("scale"))

    # Save the tables to the excel sheets
    save_to_excel_multi(tables, output_path=save_as, sheet_names=sheets, index=False)

    # Return the correlations
    return tables
