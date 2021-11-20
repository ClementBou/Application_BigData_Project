import pandas as pd
import shap
import matplotlib.pyplot as pl
from matplotlib import rcParams


def get_interpretation(model, X_test, interpretations):
    shap.initjs()

    prepare_model = prepare_model_shap(model)

    # Get explainer
    explainer = shap.TreeExplainer(prepare_model)

    # Get shap values
    shap_values = explainer.shap_values(X_test)
    X_test_min = X_test.sample(n=1000, random_state=1);
    shap_values_min = explainer.shap_values(X_test_min)

    # Get all features from x_test
    all_features = list(X_test.columns.values)
    rcParams.update({'figure.autolayout': True})

    for interpretation_name in interpretations:
        if interpretation_name == "shap_values_all":
            plot_shapley_values_all = shap.force_plot(explainer.expected_value, shap_values_min, features=X_test, feature_names=all_features)
            shap.save_html("plot_shapley_values_all.html", plot_shapley_values_all)
        elif interpretation_name == "shap_values_one":
            i = 0
            plot_shapley_values = shap.force_plot(explainer.expected_value, shap_values[i], features=X_test.iloc[i], feature_names=all_features,
                                                  show=False, matplotlib=True)
            plot_shapley_values.savefig('plot_shapley_values_i.png', format="png", dpi=150, bbox_inches='tight')

        elif interpretation_name == "summary_plot":
            shap.summary_plot(shap_values_min, features=X_test_min, show=False, max_display=X_test.shape[1])
            pl.savefig('summary_plot.png')
        else:
            raise ValueError("This interpretation isn't available. Try \"shap_values_all\" or \"shap_values_one\" or \"summary_plot\"")

def prepare_model_shap(model):
    # Get the underlying xgboost Booster of this model
    mybooster = model.get_booster()

    # Convert to good format
    model_bytearray = mybooster.save_raw()[4:]

    def convert_bytearray(self=None):
        return model_bytearray

    mybooster.save_raw = convert_bytearray

    return mybooster
