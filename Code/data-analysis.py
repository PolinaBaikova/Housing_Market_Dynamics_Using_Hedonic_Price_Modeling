import os 
import pandas as pd
import sqlite3
from scipy.stats._distn_infrastructure import rv_frozen
import scipy.stats as stats
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from functools import reduce
from operator  import add
from pygam   import LinearGAM, s, l
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent   # …/project
FIG_DIR  = ROOT_DIR / "Figures"                    # all PNGs here
TBL_DIR  = ROOT_DIR / "Tables"                     # all .tex tables here

# create them once
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
    
    
    
# Connect to SQLite database
conn = sqlite3.connect("orlando_housing.db")
cursor = conn.cursor()

# Define ZIP code groupings
zip_groups = {
    'orlando_central':    ['32801', '32804', '32809', '32839', '32806', '32807', '32822', '32812'],
    'orlando_east':       ['32817', '32820', '32826', '32828']
}

# Load data from each group and concatenate into datasets
for region, zips in zip_groups.items():
    dfs = []
    for zip_code in zips:
        table_name = f"orlando_{zip_code}"
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        dfs.append(df)
    globals()[region] = pd.concat(dfs, ignore_index=True)

# Close the database connection
conn.close()











def model_pipeline(df, predictors, region_name,
                   test_size=0.30, random_state=42,
                   out_dir=FIG_DIR, name=None):

    if name is None:
        name = region_name.replace(" ", "_").lower()

    # FIG_DIR already exists, but this is harmless
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()

    # Feature engineering
    df['ln_sale_price']    = np.log(df['sale_price'])
    df['sqft_per_bedroom'] = np.log(df['sqft_per_bedroom'])
    df['age']              = np.sqrt(df['age'])

    X = df[predictors].copy()
    y = df['ln_sale_price']

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    alphas = np.linspace(0.01, 0.003, num=50)

    scaler = StandardScaler().fit(X_trainval)
    X_trainval_scaled = scaler.transform(X_trainval)

    lasso = LassoCV(alphas=alphas, cv=5, random_state=random_state)
    lasso.fit(X_trainval_scaled, y_trainval)

    best_mu = lasso.alpha_
    selected_features = X.columns[lasso.coef_ != 0]

    # Refit OLS with robust SEs
    X_ols = sm.add_constant(X_trainval[selected_features])
    ols_model = sm.OLS(y_trainval, X_ols).fit(cov_type="HC3")

    # Test‑set metrics
    X_test_ols = sm.add_constant(X_test[selected_features])
    y_pred_test = ols_model.predict(X_test_ols)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2   = r2_score(y_test, y_pred_test)

    # Q–Q plot of *this* model’s residuals
    stats.distributions          = stats
    stats.distributions.rv_frozen = rv_frozen
    plt.figure(figsize=(5, 5))
    sm.qqplot(ols_model.resid, line="45", fit=True)   # ← fixed
    qq_path = FIG_DIR / f"qqplot_{name}.png"
    plt.tight_layout()
    plt.savefig(qq_path, dpi=300)
    plt.close()

    return ols_model, selected_features, best_mu, rmse, r2



predictors = [ 'home_type', 'age',
              'bedrooms', 'bathrooms', 'levels', 
              'has_garage', 'private_pool', 'hoa', 
              'sqft_per_bedroom', 'gated', 'recreational_facilities',
              'park', 'greenbelt', 'above_flood_plain', 'city_lot', 
              "historic_district", 'avg_schools_rating', 'water_view', 
              'playground', 'min_distance_highway'
]



# ---------------------------------------------------------------------------
#  Run for each region
# ---------------------------------------------------------------------------
model_central, selected_features_central, best_mu_central, rmse_central, \
r2_central = model_pipeline(         # ← name fixed here
    df=orlando_central,
    predictors=predictors,
    region_name="Orlando Central",
    test_size=0.3,
    random_state=42,
    out_dir=FIG_DIR,
    name="orlando_central"
)

model_east, selected_features_east, best_mu_east, rmse_east, \
r2_east = model_pipeline(
    df=orlando_east,
    predictors=predictors,
    region_name="Orlando East",
    test_size=0.3,
    random_state=42,
    out_dir=FIG_DIR,
    name="orlando_east"
)


print('Least Squares Regression Model Pipeline is Completed')


# List of Lasso input variables
lasso_inputs = [
    "home_type", "age", "bedrooms", "bathrooms", "sqft_per_bedroom",
    "levels", "parking_spaces", "has_garage", "private_pool", "hoa",
    "gated", "recreational_facilities", "park", "playground", "greenbelt",
    "above_flood_plain", "city_lot", "historic_district", "water_view",
    "avg_distance_to_schools", "avg_schools_rating", "min_distance_highway"
]

pretty_name = {
    "home_type":           "Home Type",
    "has_garage":          "Garage",
    "private_pool":        "Private Pool",
    "hoa":                 "HOA",
    "levels":              "Number of Stories",
    "parking_spaces":"Number of Parking Spaces",
    "gated":               "Gated Community",
    "park":                "Nearby Park",
    "above_flood_plain":   "Above Flood Plain",
    "city_lot":            "City Lot",
    "playground":          "Playground Nearby",
    "water_view":          "Water View",
    "bedrooms":            "Bedrooms",
    "bathrooms":           "Bathrooms",
    "age":                 "Age of Property",
    "sqft_per_bedroom": "Sq. Feet per Bedroom",
    "avg_schools_rating":  "Average School Rating",
    "min_distance_highway":"Min. Distance to Highway",
    "recreational_facilities": "Recreational Facilities",
    "greenbelt": "Greenbelt",
    "historic_district":"Historic District",
    "avg_distance_to_schools":"Average Distance to Schools"
}

# Sets of LASSO-selected features
selected_central = set(selected_features_central)
selected_east    = set(selected_features_east)

# Build the table 
df = pd.DataFrame({
    "Feature": [pretty_name[v] for v in lasso_inputs],
    "Selected: Orlando Central": [
        pretty_name[v] if v in selected_central else "" for v in lasso_inputs
    ],
    "Selected: Orlando East": [
        pretty_name[v] if v in selected_east else "" for v in lasso_inputs
    ],
})

latex_code = df.to_latex(
    index=False,
    caption="LASSO Feature Selection Results by Region",
    label="tab:lasso_selection_results",
    column_format="lll",
    escape=False 
)

out_path = TBL_DIR / "lasso_selection_results.tex"  
out_path.write_text(latex_code, encoding="utf-8") 


print('Table /"lasso_selection_results.tex/" is Created')


def latex_escape(text: str) -> str:
    """Escape underscores so LaTeX does not treat them as math subscripts."""
    return text.replace('_', r'\_')

# ------------------------------------------------------------------
#  Export a coefficient table (coef, SE, 95 % CI) with pretty labels
# ------------------------------------------------------------------
def export_coef_table_to_latex(model, filename, caption, label):
    # 1) Base table (coef & SE)
    coef_tbl = model.summary2().tables[1].loc[:, ['Coef.', 'Std.Err.']].copy()
    coef_tbl.columns = ['Coefficient', 'Std. Error']

    # 2) Add 95 % confidence intervals
    ci = model.conf_int()
    ci.columns = ['CI Lower', 'CI Upper']
    coef_tbl = coef_tbl.join(ci)

    # 3) Pretty row labels
    coef_tbl.index = [
        latex_escape(pretty_name.get(var, var)) for var in coef_tbl.index
    ]

    # 4) Round & sort
    coef_tbl = coef_tbl.round(4).sort_values('Coefficient', ascending=False)

    # 5) Write LaTeX
    coef_tbl.to_latex(
        filename,
        index=True,
        escape=True,        
        float_format="%.4f",
        column_format="lrrrr", 
        caption=caption,
        label=label
    )

# ------------------------------------------------------------------
#  Generate tables for both regions
# ------------------------------------------------------------------
export_coef_table_to_latex(
    model=model_central,
    filename=TBL_DIR / "coef_table_central.tex",   # ← use TBL_DIR
    caption="Least‑Squares Estimates with 95 percent CI (Central Orlando)",
    label="tab:ols_coeffs_c"
)

export_coef_table_to_latex(
    model=model_east,
    filename=TBL_DIR / "coef_table_east.tex",      # ← use TBL_DIR
    caption="Least‑Squares Estimates with 95 percent CI (East Orlando)",
    label="tab:ols_coeffs_e"
)




print('Table /"coef_table_central.tex/" is Created')

print('Table /"coef_table_east.tex/" is Created')





def derivative_at(gam, baseline_scaled, col_idx, delta=1e-2):
    """Central finite‑difference slope of ŷ wrt variable `col_idx`."""
    lo, hi = baseline_scaled.copy(), baseline_scaled.copy()
    lo[0, col_idx] -= delta
    hi[0, col_idx] += delta
    return (gam.predict(hi)[0] - gam.predict(lo)[0]) / (2 * delta)


def run_gam_pipeline(
        data: pd.DataFrame,
        continuous_vars: list,
        binary_vars: list,
        target_col: str = "ln_sale_price",
        name: str = "dataset",
        quantiles: tuple = (0.10, 0.25, 0.50, 0.75, 0.90),
        lam_grid=np.logspace(-3, 2, 30),
        n_splines: int = 4,
        n_splits: int = 5,
        n_repeats: int = 3,
        rand_seed: int = 0,
        delta: float = 1e-2,
        out_dir: Path = TBL_DIR,
        dpi: int = 130
    ):
    """
    Fit a LinearGAM, report metrics, save LaTeX tables, and plot smooth effects.
    `continuous_vars` get spline terms, `binary_vars` get linear terms.
    """
    # ───────────────────────────────────────────────────────────── prep data
    df = data.copy()
    df['ln_sale_price'] = np.log(df['sale_price']) 
    if target_col not in df:
        raise ValueError(f"{target_col} not found in dataframe.")
    feature_cols = continuous_vars + binary_vars

    y  = df[target_col].to_numpy(float)
    X_df = df[feature_cols].copy()

    scaler = StandardScaler().fit(X_df[continuous_vars])
    X_df[continuous_vars] = scaler.transform(X_df[continuous_vars])
    X_s = X_df.to_numpy()

    # ───────────────────────────────────────────────────── build GAM formula
    s_terms = [s(i, n_splines=n_splines) for i in range(len(continuous_vars))]
    l_terms = [l(len(continuous_vars)+j)  for j in range(len(binary_vars))]
    gam_form = reduce(add, s_terms + l_terms)

    # ─────────────────────────────────────────── cross‑validated performance
    rkf = RepeatedKFold(n_splits=n_splits,
                        n_repeats=n_repeats,
                        random_state=rand_seed)

    def _metrics(y_true, y_pred):
        return (r2_score(y_true, y_pred),
                np.sqrt(mean_squared_error(y_true, y_pred)),
                mean_absolute_error(y_true, y_pred))

    r2s, rmses, maes = [], [], []
    for tr, te in rkf.split(X_s):
        gam = LinearGAM(gam_form).gridsearch(X_s[tr], y[tr],
                                             lam=lam_grid, progress=False)
        y_hat = gam.predict(X_s[te])
        r2, rmse, mae = _metrics(y[te], y_hat)
        r2s.append(r2); rmses.append(rmse); maes.append(mae)

    # final fit + hold‑out
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.25, random_state=rand_seed)
    gam_final = LinearGAM(gam_form).gridsearch(X_tr, y_tr,
                                               lam=lam_grid, progress=False)
    r2_h, rmse_h, mae_h = _metrics(y_te, gam_final.predict(X_te))

    # marginal‑effect table
    def latex_escape(text: str) -> str:
        """Escape underscores so LaTeX doesn’t treat them as math subscripts."""
        return text.replace('_', r'\_')

    # ─── Marginal‑effect table ──────────────────────────────────────────────
    idx_labels, rows = [], []
    for q in quantiles:
        baseline = df[feature_cols].quantile(q).to_numpy().reshape(1, -1)
        baseline_scaled = baseline.copy()
        baseline_scaled[0, :len(continuous_vars)] = (
            scaler.transform(baseline_scaled[:, :len(continuous_vars)])
        )
        slopes = {
            v: derivative_at(gam_final, baseline_scaled,
                             feature_cols.index(v), delta)
            for v in continuous_vars
        }
        rows.append(slopes)
        idx_labels.append(f"{int(q*100)}%")          # plain labels (with %)

    marg_df = (
        pd.DataFrame(rows, index=idx_labels)         # rows = quantiles
          .T.round(4)                                # transpose → cols = q‑tiles
    )

    # prettify: row names & header
    marg_df.index  = [latex_escape(pretty_name.get(v, v)) for v in marg_df.index]
    marg_df.index.name = None

    # escape % so LaTeX doesn’t eat the line
    marg_df.columns = [lbl.replace('%', r'\%') for lbl in marg_df.columns]

    caption_title  = name.replace("_", " ").title()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{name}_marginal.tex").write_text(
        "\\begin{table}[H]\n\\centering\n"
        + marg_df.to_latex(
              index   = True,
              escape  = False,
              float_format = "%.4f",
              column_format = "l" + "r"*len(marg_df.columns)  # auto‑width
          )
        + f"\\caption{{{caption_title} – Marginal Effects}}\n"
          f"\\label{{tab:{name}_marginal}}\n"
        + "\\end{table}\n",
        encoding="utf-8"
    )


    # ─── Binary‑coefficient table (unchanged) ───────────────────────────────
    lin_coefs = {
        var: gam_final.coef_[gam_final.terms.get_coef_indices(
            len(continuous_vars) + j)][0]
        for j, var in enumerate(binary_vars)
    }
    coef_df = pd.DataFrame(lin_coefs, index=["Coefficients"]).T.round(4)
    coef_df.index = [latex_escape(pretty_name.get(v, v)) for v in coef_df.index]

    (out_dir / f"{name}_binary_coef.tex").write_text(
        "\\begin{table}[H]\n\\centering\n"
        + coef_df.to_latex(
            index=True, escape=False, float_format="%.4f",
            column_format="lr"
        )
        + "\\end{table}\n",
        encoding="utf-8"
    )


    # smooth‑effect plots
    plot_dir = FIG_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)


    for i, var in enumerate(continuous_vars):
        xs_scaled = gam_final.generate_X_grid(term=i, n=200)[:, i]
        xs_raw    = xs_scaled * scaler.scale_[i] + scaler.mean_[i]
        pd_curve  = gam_final.partial_dependence(term=i,
                      X=gam_final.generate_X_grid(term=i, n=200)).ravel()

        pd_points = gam_final.partial_dependence(term=i, X=X_s).ravel()
        raw_obs   = (X_df[var] * scaler.scale_[i] + scaler.mean_[i]).values

        plt.figure(figsize=(5.5, 4), dpi=dpi)
        plt.scatter(raw_obs, pd_points, s=10, alpha=.18, color="grey",
                    label="Data points")
        plt.plot(xs_raw, pd_curve, color="royalblue", lw=2,
                 label="Smoothed curve")
        plt.xlabel(var)
        plt.ylabel(f"Effect on {target_col}")
        plt.legend()
        plt.tight_layout()
        fig_path = plot_dir / f"{name}_{var}_smooth.png"
        plt.savefig(fig_path, dpi=dpi)
        print(f'Figure "{fig_path.name}" is Created')
        plt.close()

    return gam_final, marg_df, coef_df


print('GAM Pipeline is Completed')


gam_model_central, marg_table_central, binary_table_central = run_gam_pipeline(
    data=orlando_central,                   
    continuous_vars=[
        'age', 'bedrooms', 'bathrooms',
        'sqft_per_bedroom', 'avg_schools_rating', 'min_distance_highway'
    ],
    binary_vars=[
        'home_type', 'has_garage', 'private_pool', 'hoa',
        'park', 'above_flood_plain', 'city_lot', 'playground', 'water_view'
    ],
    name="orlando_central"
)



data = orlando_central
continuous_vars = [
    'age', 'bedrooms', 'bathrooms',
    'sqft_per_bedroom', 'avg_schools_rating', 'min_distance_highway'
]


# Calculate selected quantiles
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_df = data[continuous_vars].quantile(quantiles).T
quantile_df.columns = ['10%', '25%', '50%', '75%', '90%']
quantile_df = quantile_df.round(2)



latex_path = TBL_DIR / "quantiles_central.tex"     
quantile_df.to_latex(
    latex_path,
    escape=True,
    caption="Quantiles of Continuous Variables in Orlando Central Dataset",
    label="tab:quantiles_orlando_c",
    float_format="%.2f",
    column_format="lccccc"
)



gam_model_east, marg_table, binary_table = run_gam_pipeline(
    data=orlando_east,                    # your DataFrame
    continuous_vars = [
        'age', 'bedrooms', 'bathrooms',
        'sqft_per_bedroom', 'avg_schools_rating', 'min_distance_highway'
    ],

    binary_vars = [
        'home_type', 'has_garage', 'private_pool', 'hoa', 'levels', 'gated',
        'above_flood_plain',  'playground', 'water_view'
    ],
    name="orlando_east"
)

print('Table /"quantiles_central.tex/" is Created')

data = orlando_east
continuous_vars = [
    'age', 'bedrooms', 'bathrooms',
    'sqft_per_bedroom', 'avg_schools_rating', 'min_distance_highway'
]

# Calculate selected quantiles
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_df = data[continuous_vars].quantile(quantiles).T
quantile_df.columns = ['10%', '25%', '50%', '75%', '90%']
quantile_df = quantile_df.round(2)



latex_path = TBL_DIR / "quantiles_east.tex"       
quantile_df.to_latex(
    latex_path,
    escape=True,
    caption="Quantiles of Continuous Variables in Orlando East Dataset",
    label="tab:quantiles_orlando_e",
    float_format="%.2f",
    column_format="lccccc"
)



print('Table /"quantiles_east.tex/" is Created')
