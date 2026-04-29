"""
NBA Salary Prediction Dashboard
CIS 2450: Big Data Analytics — Final Project
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, callback, dcc, html
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "processed" / "nba_player_games_clean.parquet"

TIER_COLORS = {
    "budget": "#636EFA",
    "mid_low": "#00CC96",
    "mid_high": "#FFA15A",
    "max": "#EF553B",
}
TIER_ORDER = ["budget", "mid_low", "mid_high", "max"]
TIER_LABELS = {
    "budget": "Budget (Bottom 25%)",
    "mid_low": "Mid-Low (25–50%)",
    "mid_high": "Mid-High (50–75%)",
    "max": "Max (Top 25%)",
}

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

NUMERIC_FEATURES = [
    "PTS_PER36", "REB_PER36", "AST_PER36", "STL_PER36", "BLK_PER36",
    "TOV_PER36", "USAGE_PROXY", "YEARS_EXP", "AGE_AT_GAME", "WIN",
    "REDDIT_SENTIMENT_AVG", "REDDIT_POST_COUNT",
]
CAT_FEATURES = ["POSITION_SIMPLE"]
ALL_FEATURE_COLS = NUMERIC_FEATURES + ["POS_C", "POS_PF", "POS_PG", "POS_SF", "POS_SG"]

# Hardcoded model scores from Notebook 4
MODEL_SCORES = {
    "regression": [
        {"Model": "OLS Linear Regression", "Task": "Regression", "R²": 0.4178, "RMSE": 1.0408, "MAE": 0.7268},
        {"Model": "Ridge Regression", "Task": "Regression", "R²": 0.4184, "RMSE": 1.0402, "MAE": 0.7263},
        {"Model": "Lasso (L1)", "Task": "Regression", "R²": -0.0009, "RMSE": 1.3646, "MAE": 1.0447},
        {"Model": "ElasticNet", "Task": "Regression", "R²": 0.2171, "RMSE": 1.2069, "MAE": 0.8950},
    ],
    "classification": [
        {"Model": "Decision Tree", "Task": "Classification", "Accuracy": 0.5581},
        {"Model": "Logistic Regression (PCA)", "Task": "Classification", "Accuracy": 0.5694},
        {"Model": "Random Forest", "Task": "Classification", "Accuracy": 0.6204},
    ],
}

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

print("Loading data...")
raw = pl.read_parquet(DATA_PATH)

# Full dataset as pandas for EDA
df_full = raw.to_pandas()

# Player-season aggregate (drop rows missing bio data)
player_season_pl = (
    raw.drop_nulls(subset=["POSITION_SIMPLE", "AGE_AT_GAME"])
    .group_by(["PLAYER_NAME", "SEASON", "POSITION_SIMPLE"])
    .agg([
        pl.col("PTS_PER36").mean(),
        pl.col("REB_PER36").mean(),
        pl.col("AST_PER36").mean(),
        pl.col("STL_PER36").mean(),
        pl.col("BLK_PER36").mean(),
        pl.col("TOV_PER36").mean(),
        pl.col("USAGE_PROXY").mean(),
        pl.col("YEARS_EXP").mean(),
        pl.col("AGE_AT_GAME").mean(),
        pl.col("WIN").mean(),
        pl.col("REDDIT_SENTIMENT_AVG").mean(),
        pl.col("REDDIT_POST_COUNT").mean(),
        pl.col("SEASON_SALARY").first(),
        pl.col("LOG_SALARY").first(),
        pl.col("SALARY_TIER").first(),
        pl.col("SALARY_TIER_NUM").first(),
    ])
)
ps = player_season_pl.to_pandas()

# One-hot encode position
pos_dummies = pd.get_dummies(ps["POSITION_SIMPLE"], prefix="POS")
for col in ["POS_C", "POS_PF", "POS_PG", "POS_SF", "POS_SG"]:
    if col not in pos_dummies.columns:
        pos_dummies[col] = 0
ps_encoded = pd.concat([ps, pos_dummies], axis=1)

# ---------------------------------------------------------------------------
# Model training (startup)
# ---------------------------------------------------------------------------

print("Training models...")
X = ps_encoded[ALL_FEATURE_COLS].values.astype(float)
y_reg = ps_encoded["LOG_SALARY"].values
y_cls = ps_encoded["SALARY_TIER_NUM"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linreg = LinearRegression()
linreg.fit(X_scaled, y_reg)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y_reg)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y_cls)

TIER_NUM_TO_NAME = {1: "budget", 2: "mid_low", 3: "mid_high", 4: "max"}

# ---------------------------------------------------------------------------
# Precomputed EDA dataframes
# ---------------------------------------------------------------------------

# Salary distribution (player-season level)
salary_dist = ps[["PLAYER_NAME", "SEASON", "SEASON_SALARY", "LOG_SALARY", "SALARY_TIER"]].copy()

# Salary by position
salary_by_pos = ps.dropna(subset=["POSITION_SIMPLE"])

# Age vs salary
age_salary = ps.dropna(subset=["AGE_AT_GAME"]).copy()
age_salary["AGE_AT_GAME"] = age_salary["AGE_AT_GAME"].round(0).astype(int)

# Season trends
season_trends = (
    ps.groupby("SEASON")["SEASON_SALARY"]
    .agg(["mean", "median"])
    .reset_index()
    .rename(columns={"mean": "Mean Salary", "median": "Median Salary"})
    .sort_values("SEASON")
)

# Games per season (raw game logs)
games_per_season = (
    raw.group_by("SEASON").len()
    .sort("SEASON")
    .to_pandas()
    .rename(columns={"len": "Game Logs"})
)

# Correlation matrix
CORR_COLS = [
    "PTS_PER36", "REB_PER36", "AST_PER36", "STL_PER36", "BLK_PER36",
    "TOV_PER36", "USAGE_PROXY", "YEARS_EXP", "AGE_AT_GAME",
    "REDDIT_SENTIMENT_AVG", "REDDIT_POST_COUNT", "SEASON_SALARY",
]
corr_df = ps[CORR_COLS].corr()

# Linear regression coefficients
linreg_coefs = pd.DataFrame({
    "Feature": ALL_FEATURE_COLS,
    "Coefficient": linreg.coef_,
}).sort_values("Coefficient", key=abs, ascending=False)

ridge_coefs = pd.DataFrame({
    "Feature": ALL_FEATURE_COLS,
    "Coefficient": ridge.coef_,
}).sort_values("Coefficient", key=abs, ascending=False)

# RF feature importances
rf_importances = pd.DataFrame({
    "Feature": ALL_FEATURE_COLS,
    "Importance": rf.feature_importances_,
}).sort_values("Importance", ascending=False)

# Reddit sentiment
reddit_df = ps[ps["REDDIT_POST_COUNT"] > 0].copy()

print("Startup complete. Launching app...")

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "NBA Salary Prediction"

# --- Shared styles ---
CARD_STYLE = {"marginBottom": "16px"}
TAB_LABEL_STYLE = {"fontWeight": "600"}

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

total_rows = len(df_full)
unique_players = df_full["PLAYER_NAME"].nunique()
salary_min = f"${df_full['SEASON_SALARY'].min() / 1e6:.2f}M"
salary_max = f"${df_full['SEASON_SALARY'].max() / 1e6:.1f}M"

tab1 = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("NBA Player Salary Prediction", className="mt-3 mb-1"),
            html.P(
                "Predicting NBA player salaries from 5 seasons of game-log performance stats, "
                "player bio data, and Reddit sentiment.",
                className="text-muted mb-1",
            ),
            html.P(
                html.I(
                    "Key finding: Scoring (PTS/36) is the strongest salary driver, "
                    "but performance stats explain only ~42% of variance — "
                    "contracts also reflect experience, market timing, and team context."
                ),
                className="text-muted small mb-3",
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Game Logs", className="card-title text-center"),
            html.H3(f"{total_rows:,}", className="text-center text-primary"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Unique Players", className="card-title text-center"),
            html.H3(f"{unique_players:,}", className="text-center text-success"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Seasons", className="card-title text-center"),
            html.H3("5", className="text-center text-warning"),
            html.P("2020–21 → 2024–25", className="text-center text-muted mb-0 small"),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Salary Range", className="card-title text-center"),
            html.H3(salary_max, className="text-center text-danger"),
            html.P(f"Min: {salary_min}", className="text-center text-muted mb-0 small"),
        ])), width=3),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Game Logs Per Season"),
                dcc.Graph(
                    figure=px.bar(
                        games_per_season,
                        x="SEASON", y="Game Logs",
                        color="SEASON",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        labels={"SEASON": "Season", "Game Logs": "# Game Logs"},
                    ).update_layout(showlegend=False, margin=dict(t=20)),
                    config={"displayModeBar": False},
                ),
            ])),
        ], width=7),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Data Sources"),
                html.Ul([
                    html.Li([html.B("nba_api: "), "Player game logs (5 seasons, ~119k rows)"]),
                    html.Li([html.B("Basketball-Reference: "), "Season salaries + player bios (scraped)"]),
                    html.Li([html.B("Reddit (PRAW): "), "Post counts & avg sentiment per player"]),
                ]),
                html.H5("Salary Tiers", className="mt-3"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th("Tier"), html.Th("Percentile"), html.Th("Color")])),
                    html.Tbody([
                        html.Tr([html.Td("Budget"), html.Td("Bottom 25%"),
                                 html.Td(html.Span("■", style={"color": TIER_COLORS["budget"], "fontSize": "20px"}))]),
                        html.Tr([html.Td("Mid-Low"), html.Td("25–50%"),
                                 html.Td(html.Span("■", style={"color": TIER_COLORS["mid_low"], "fontSize": "20px"}))]),
                        html.Tr([html.Td("Mid-High"), html.Td("50–75%"),
                                 html.Td(html.Span("■", style={"color": TIER_COLORS["mid_high"], "fontSize": "20px"}))]),
                        html.Tr([html.Td("Max"), html.Td("Top 25%"),
                                 html.Td(html.Span("■", style={"color": TIER_COLORS["max"], "fontSize": "20px"}))]),
                    ]),
                ], bordered=True, size="sm"),
            ])),
        ], width=5),
    ]),
], fluid=True)

# ---------------------------------------------------------------------------
# Tab 2 — EDA Explorer
# ---------------------------------------------------------------------------

EDA_OPTIONS = [
    {"label": "Salary Distribution", "value": "salary_dist"},
    {"label": "Salary by Position", "value": "salary_pos"},
    {"label": "Age vs. Salary", "value": "age_salary"},
    {"label": "Performance vs. Salary", "value": "perf_salary"},
    {"label": "Season Salary Trends", "value": "season_trends"},
    {"label": "Correlation Heatmap", "value": "corr"},
    {"label": "Reddit Sentiment vs. Salary", "value": "reddit"},
]

EDA_FINDINGS = {
    "salary_dist": "NBA salaries are highly right-skewed — a small number of max-contract players earn dramatically more than the median. Log-transforming salary reveals a roughly normal distribution, which is why LOG_SALARY is used as the regression target.",
    "salary_pos": "Centers and Power Forwards tend to earn higher median salaries, reflecting the premium placed on frontcourt size. Point Guards show the widest spread, from minimum-wage rookies to max-contract stars.",
    "age_salary": "Salary peaks around ages 27–30, consistent with players reaching their prime. Rookies cluster near the league minimum, and salaries generally decline after age 32 as production drops off.",
    "perf_salary": "Points per 36 minutes shows the strongest positive correlation with salary (r ≈ 0.58). Rebounds and assists are also positively correlated but weaker — scoring drives contracts more than other stats.",
    "season_trends": "Both mean and median NBA salaries have risen steadily from 2020–21 to 2024–25, reflecting the growing salary cap driven by TV deal revenues. The gap between mean and median widens each year, indicating increasing inequality.",
    "corr": "USAGE_PROXY and PTS_PER36 are the most correlated features with SEASON_SALARY. TOV_PER36 is negatively correlated once usage is controlled. Many per-36 stats are highly intercorrelated, motivating PCA for dimensionality reduction.",
    "reddit": "Players with more Reddit post mentions tend to earn higher salaries — high-profile stars attract both attention and big contracts. Sentiment alone is a weak predictor, and adding Reddit features to the model actually slightly hurts OLS regression (ΔR² = −0.015) while providing negligible benefit to the Random Forest (+0.003 accuracy). Reddit volume correlates with salary, but the signal is largely redundant with performance stats.",
}

tab2 = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("EDA Explorer", className="mt-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select EDA View:"),
                    dcc.Dropdown(
                        id="eda-dropdown",
                        options=EDA_OPTIONS,
                        value="salary_dist",
                        clearable=False,
                    ),
                ], width=5),
                dbc.Col([
                    # All controls always in layout; shown/hidden via callback
                    html.Div(
                        [html.Label("Scale:"),
                         dbc.RadioItems(
                             id="eda-scale",
                             options=[{"label": "Raw", "value": "raw"}, {"label": "Log", "value": "log"}],
                             value="raw", inline=True,
                         )],
                        id="ctrl-scale", style={"display": "none"},
                    ),
                    html.Div(
                        [html.Label("Season:"),
                         dcc.Dropdown(
                             id="eda-season-filter",
                             options=[{"label": "All Seasons", "value": "all"}]
                                     + [{"label": s, "value": s} for s in SEASONS],
                             value="all", clearable=False, style={"width": "180px"},
                         )],
                        id="ctrl-season", style={"display": "none"},
                    ),
                    html.Div(
                        [html.Label("Stat:"),
                         dcc.Dropdown(
                             id="eda-perf-stat",
                             options=[
                                 {"label": "Points per 36", "value": "PTS_PER36"},
                                 {"label": "Rebounds per 36", "value": "REB_PER36"},
                                 {"label": "Assists per 36", "value": "AST_PER36"},
                             ],
                             value="PTS_PER36", clearable=False, style={"width": "200px"},
                         )],
                        id="ctrl-perf", style={"display": "none"},
                    ),
                ], width=7),
            ], className="mb-3"),
            dcc.Graph(id="eda-chart", style={"height": "450px"}),
            html.P(id="eda-finding", className="text-muted small mt-2 px-1"),
        ])
    ])
], fluid=True)

# ---------------------------------------------------------------------------
# Tab 3 — Modeling Results
# ---------------------------------------------------------------------------

MODEL_DROPDOWN_OPTIONS = [
    {"label": "OLS Linear Regression — coefficients", "value": "ols"},
    {"label": "Ridge Regression — coefficients", "value": "ridge"},
    {"label": "Decision Tree — confusion matrix", "value": "dt"},
    {"label": "Logistic Regression (PCA) — confusion matrix", "value": "logreg"},
    {"label": "Random Forest — feature importances", "value": "rf"},
]

# Pre-built confusion matrices from notebook 4 — rows/cols in tier order
DT_CM = np.array([[71, 11, 10, 4], [16, 27, 26, 12], [18, 32, 35, 7], [7, 5, 8, 64]])
LOGREG_CM = np.array([[60, 15, 16, 5], [14, 38, 23, 6], [18, 34, 35, 5], [4, 3, 9, 68]])
RF_CM = np.array([[66, 13, 13, 4], [17, 34, 24, 6], [14, 18, 51, 9], [5, 3, 8, 68]])
CM_LABELS = ["budget", "mid_low", "mid_high", "max"]

MODEL_INTERPRETATIONS = {
    "ols": "OLS coefficients show the marginal effect of each standardized feature on log-salary. "
           "PTS_PER36 is the strongest positive driver — scoring is the primary determinant of NBA contracts. "
           "YEARS_EXP also has a large positive effect: veterans command higher salaries regardless of output. "
           "USAGE_PROXY has a negative coefficient after controlling for raw stats, penalizing high-usage but inefficient players.",
    "ridge": "Ridge regression adds L2 regularization (alpha=1.0) to shrink coefficients and reduce variance from correlated per-36 stats. "
             "Compared to OLS, Ridge slightly improves test R2 (0.4184 vs 0.4178) by preventing overfitting. "
             "The coefficient pattern is similar to OLS — scoring and experience dominate — but magnitudes are modestly shrunk across all features.",
    "dt": "The Decision Tree (max_depth=4, tuned via validation curve) achieves 55.8% accuracy on 4-class tier prediction. "
          "It performs best on Budget and Max players — the clearest statistical extremes — and struggles most with Mid-Low vs Mid-High, "
          "where the percentile boundary is inherently noisy.",
    "logreg": "Logistic Regression with PCA preprocessing (5 components) reaches 56.9% accuracy. "
              "PCA reduces the 17-feature space while preserving most variance, helping avoid overfitting in the multinomial setting. "
              "Max-tier players are predicted most reliably (precision 0.69), consistent with their statistically distinct profiles.",
    "rf": "Random Forest (200 trees, max_depth=10) is the best classifier at 62.0% accuracy — +6 points over Decision Tree. "
          "The ensemble averages many trees to reduce variance. PTS_PER36, USAGE_PROXY, and YEARS_EXP are the top three features, "
          "mirroring regression findings and confirming scoring volume and experience are the primary salary tier drivers.",
}

tab3 = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Modeling Results", className="mt-3 mb-0"),
            html.P(
                "Performance stats explain ~42% of salary variance (R²=0.42, held-out test set). "
                "Random Forest achieves 62% accuracy on 4-class tier prediction. "
                "Scores from notebook models (24 features, 80/20 split). Best scores highlighted in green.",
                className="text-muted small mb-3",
            ),
            dbc.Row([
                dbc.Col([
                    html.H6("Regression — target: log(salary)"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Model"), html.Th("R²"), html.Th("RMSE"), html.Th("MAE"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(r["Model"]),
                                html.Td(html.B(f"{r['R²']:.4f}",
                                    style={"color": "#28a745"} if r["R²"] == max(x["R²"] for x in MODEL_SCORES["regression"]) else {})),
                                html.Td(f"{r['RMSE']:.4f}"),
                                html.Td(f"{r['MAE']:.4f}"),
                            ]) for r in MODEL_SCORES["regression"]
                        ]),
                    ], bordered=True, hover=True, size="sm"),
                    html.P("Lasso collapses to predicting the mean (over-regularized). Ridge marginally beats OLS.",
                           className="text-muted small mt-1"),
                ], width=6),
                dbc.Col([
                    html.H6("Classification — target: salary tier"),
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Model"), html.Th("Accuracy")])),
                        html.Tbody([
                            html.Tr([
                                html.Td(r["Model"]),
                                html.Td(html.B(f"{r['Accuracy']:.4f}",
                                    style={"color": "#28a745"} if r["Accuracy"] == max(x["Accuracy"] for x in MODEL_SCORES["classification"]) else {})),
                            ]) for r in MODEL_SCORES["classification"]
                        ]),
                    ], bordered=True, hover=True, size="sm"),
                    html.P("Baseline (random guess) = 25%. Random Forest gains +6 pts over Decision Tree via ensemble averaging.",
                           className="text-muted small mt-1"),
                ], width=6),
            ], className="mb-4"),
            html.H6("Model Deep Dive"),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=MODEL_DROPDOWN_OPTIONS,
                        value="rf",
                        clearable=False,
                    ),
                    width=7,
                ),
            ], className="mb-2"),
            dcc.Graph(id="model-chart", style={"height": "380px"}),
            html.P(id="model-interpretation", className="text-muted small mt-1 mb-4"),
            html.H6("Unsupervised Learning — PCA & K-Means"),
            html.P(
                "PCA projects 17 features onto 2 principal components. "
                "K-Means (K=5) identifies natural player archetypes used as engineered features.",
                className="text-muted small mb-2",
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id="pca-scatter", style={"height": "360px"}), width=6),
                dbc.Col(dcc.Graph(id="kmeans-scatter", style={"height": "360px"}), width=6),
            ]),
        ])
    ])
], fluid=True)

# ---------------------------------------------------------------------------
# Tab 4 — Salary Predictor
# ---------------------------------------------------------------------------

def make_slider(id_, label, min_, max_, step, value, marks=None):
    return dbc.Row([
        dbc.Col(html.Label(label, style={"fontSize": "13px"}), width=4),
        dbc.Col(dcc.Slider(
            id=id_, min=min_, max=max_, step=step, value=value,
            marks=marks or {min_: str(min_), max_: str(max_)},
            tooltip={"placement": "bottom", "always_visible": True},
        ), width=8),
    ], className="mb-2")

tab4 = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Salary Predictor", className="mt-3 mb-1"),
            html.P("Adjust player stats to predict their salary and tier.", className="text-muted mb-3"),
            dbc.Card(dbc.CardBody([
                html.H6("Performance (per 36 minutes)"),
                make_slider("sl-pts", "Points", 0, 40, 0.5, 15),
                make_slider("sl-reb", "Rebounds", 0, 20, 0.5, 5),
                make_slider("sl-ast", "Assists", 0, 15, 0.5, 4),
                make_slider("sl-stl", "Steals", 0, 5, 0.1, 1),
                make_slider("sl-blk", "Blocks", 0, 5, 0.1, 0.5),
                make_slider("sl-tov", "Turnovers", 0, 10, 0.5, 2),
                html.Hr(),
                html.H6("Player Profile"),
                make_slider("sl-usage", "Usage Proxy", 0.0, 1.0, 0.01, 0.3),
                make_slider("sl-exp", "Years of Experience", 0, 20, 1, 5),
                make_slider("sl-age", "Age", 19, 42, 1, 26),
                make_slider("sl-win", "Win Rate", 0.0, 1.0, 0.01, 0.5),
                html.Hr(),
                html.H6("Reddit Presence"),
                make_slider("sl-sent", "Avg Sentiment (−1 to 1)", -1.0, 1.0, 0.05, 0.0),
                html.Hr(),
                html.Label("Position:"),
                dcc.Dropdown(
                    id="sl-pos",
                    options=[{"label": p, "value": p} for p in ["PG", "SG", "SF", "PF", "C"]],
                    value="SG",
                    clearable=False,
                    style={"marginBottom": "12px"},
                ),
            ])),
        ], width=5),
        dbc.Col([
            html.H4("Prediction", className="mt-3 mb-1"),
            html.P("Updates live as you adjust sliders.", className="text-muted mb-3"),
            dbc.Card(dbc.CardBody([
                html.H6("Predicted Salary (Linear Regression)"),
                html.H2(id="pred-salary", className="text-primary mb-3"),
                html.H6("Predicted Salary Tier (Random Forest)"),
                html.Div(id="pred-tier-badge", className="mb-3"),
                html.H6("Tier Probabilities"),
                dcc.Graph(id="pred-proba-chart", style={"height": "220px"},
                          config={"displayModeBar": False}),
            ]), className="mb-3"),
            html.P([
                "Salary estimated via log-salary regression. ",
                "Tier probabilities from Random Forest. ",
                "Note: the predictor's models are trained on all data (no holdout); "
                "the R²=0.42 / 62% accuracy figures in the Modeling tab are from "
                "notebook models using a held-out test set and a larger feature set.",
            ], className="text-muted small mt-2"),
        ], width=7),
    ])
], fluid=True)

# ---------------------------------------------------------------------------
# App layout — tabs
# ---------------------------------------------------------------------------

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="🏀 NBA Salary Prediction",
        brand_href="#",
        color="dark",
        dark=True,
        className="mb-0",
    ),
    dbc.Tabs([
        dbc.Tab(tab1, label="Overview", tab_style=TAB_LABEL_STYLE),
        dbc.Tab(tab2, label="EDA Explorer", tab_style=TAB_LABEL_STYLE),
        dbc.Tab(tab3, label="Modeling Results", tab_style=TAB_LABEL_STYLE),
        dbc.Tab(tab4, label="Salary Predictor", tab_style=TAB_LABEL_STYLE),
    ], className="mt-0"),
], fluid=True, style={"padding": "0"})

# ---------------------------------------------------------------------------
# Callbacks — Tab 2: EDA
# ---------------------------------------------------------------------------

@callback(
    Output("ctrl-scale", "style"),
    Output("ctrl-season", "style"),
    Output("ctrl-perf", "style"),
    Input("eda-dropdown", "value"),
)
def eda_controls(view):
    show = {"display": "block"}
    hide = {"display": "none"}
    if view == "salary_dist":
        return show, hide, hide
    if view in ("salary_pos", "age_salary", "reddit"):
        return hide, show, hide
    if view == "perf_salary":
        return hide, hide, show
    return hide, hide, hide


@callback(
    Output("eda-chart", "figure"),
    Output("eda-finding", "children"),
    Input("eda-dropdown", "value"),
    Input("eda-season-filter", "value"),
    Input("eda-perf-stat", "value"),
    Input("eda-scale", "value"),
)
def update_eda(view, season, perf_stat, scale):
    # Season filter helper
    def filter_season(df, col="SEASON"):
        if season and season != "all":
            return df[df[col] == season]
        return df

    if view == "salary_dist":
        data = salary_dist.copy()
        col = "LOG_SALARY" if scale == "log" else "SEASON_SALARY"
        xlabel = "Log Salary" if scale == "log" else "Season Salary ($)"
        fig = px.histogram(
            data, x=col, color="SALARY_TIER",
            color_discrete_map=TIER_COLORS,
            category_orders={"SALARY_TIER": TIER_ORDER},
            nbins=50, opacity=0.8,
            labels={col: xlabel, "SALARY_TIER": "Tier"},
            title="NBA Salary Distribution by Tier",
        )

    elif view == "salary_pos":
        data = filter_season(salary_by_pos)
        fig = px.box(
            data, x="POSITION_SIMPLE", y="SEASON_SALARY",
            color="POSITION_SIMPLE",
            category_orders={"POSITION_SIMPLE": ["PG", "SG", "SF", "PF", "C"]},
            labels={"POSITION_SIMPLE": "Position", "SEASON_SALARY": "Season Salary ($)"},
            title="Salary Distribution by Position",
        )

    elif view == "age_salary":
        data = filter_season(age_salary)
        fig = px.scatter(
            data, x="AGE_AT_GAME", y="SEASON_SALARY",
            color="SALARY_TIER", color_discrete_map=TIER_COLORS,
            category_orders={"SALARY_TIER": TIER_ORDER},
            trendline="lowess",
            labels={"AGE_AT_GAME": "Age", "SEASON_SALARY": "Season Salary ($)", "SALARY_TIER": "Tier"},
            title="Age vs. Salary",
            opacity=0.5,
        )

    elif view == "perf_salary":
        stat_label = {"PTS_PER36": "Points/36", "REB_PER36": "Rebounds/36", "AST_PER36": "Assists/36"}
        fig = px.scatter(
            ps, x=perf_stat, y="SEASON_SALARY",
            color="SALARY_TIER", color_discrete_map=TIER_COLORS,
            category_orders={"SALARY_TIER": TIER_ORDER},
            trendline="ols",
            labels={perf_stat: stat_label.get(perf_stat, perf_stat), "SEASON_SALARY": "Season Salary ($)", "SALARY_TIER": "Tier"},
            title=f"{stat_label.get(perf_stat, perf_stat)} vs. Salary",
            opacity=0.6,
            hover_data=["PLAYER_NAME", "SEASON"],
        )

    elif view == "season_trends":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=season_trends["SEASON"], y=season_trends["Mean Salary"],
            mode="lines+markers", name="Mean Salary",
            line=dict(color="#EF553B", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=season_trends["SEASON"], y=season_trends["Median Salary"],
            mode="lines+markers", name="Median Salary",
            line=dict(color="#636EFA", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="NBA Salary Trends Across Seasons",
            xaxis_title="Season", yaxis_title="Salary ($)",
        )

    elif view == "corr":
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Heatmap",
            text_auto=".2f",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    elif view == "reddit":
        data = filter_season(reddit_df)
        fig = px.scatter(
            data, x="REDDIT_SENTIMENT_AVG", y="SEASON_SALARY",
            color="SALARY_TIER", color_discrete_map=TIER_COLORS,
            size="REDDIT_POST_COUNT", size_max=20,
            category_orders={"SALARY_TIER": TIER_ORDER},
            trendline="ols",
            labels={
                "REDDIT_SENTIMENT_AVG": "Avg Reddit Sentiment",
                "SEASON_SALARY": "Season Salary ($)",
                "SALARY_TIER": "Tier",
                "REDDIT_POST_COUNT": "Post Count",
            },
            title="Reddit Sentiment vs. Salary",
            hover_data=["PLAYER_NAME", "SEASON"],
            opacity=0.7,
        )

    else:
        fig = go.Figure()

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, EDA_FINDINGS.get(view, "")


# ---------------------------------------------------------------------------
# Callbacks — Tab 3: Modeling
# ---------------------------------------------------------------------------

@callback(
    Output("model-chart", "figure"),
    Output("model-interpretation", "children"),
    Input("model-dropdown", "value"),
)
def update_model_chart(model):
    if model == "ols":
        fig = px.bar(
            linreg_coefs.head(15),
            x="Coefficient", y="Feature", orientation="h",
            color="Coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="OLS Linear Regression — Feature Coefficients (top 15, standardized)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

    elif model == "ridge":
        fig = px.bar(
            ridge_coefs.head(15),
            x="Coefficient", y="Feature", orientation="h",
            color="Coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Ridge Regression (alpha=1.0) — Feature Coefficients (top 15, standardized)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

    elif model == "rf":
        fig = px.bar(
            rf_importances.head(12),
            x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Random Forest — Feature Importances (top 12)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})

    elif model in ("dt", "logreg", "rf_cm"):
        cm_map = {"dt": (DT_CM, "Decision Tree"), "logreg": (LOGREG_CM, "Logistic Regression (PCA)")}
        cm, label = cm_map.get(model, (RF_CM, "Random Forest"))
        fig = px.imshow(
            cm,
            x=CM_LABELS, y=CM_LABELS,
            color_continuous_scale="Blues",
            title=f"{label} — Confusion Matrix (Actual vs Predicted)",
            text_auto=True,
            labels=dict(x="Predicted Tier", y="Actual Tier"),
        )
        fig.update_layout(margin=dict(t=50))

    else:
        fig = go.Figure()

    fig.update_layout(template="plotly_white", margin=dict(t=50, b=30, l=80, r=30))
    interpretation = MODEL_INTERPRETATIONS.get(model, "")
    return fig, interpretation


@callback(
    Output("pca-scatter", "figure"),
    Output("kmeans-scatter", "figure"),
    Input("model-dropdown", "value"),  # dummy trigger to init on load
)
def update_unsupervised(_):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    X_pca_model = PCA(n_components=2)
    X_2d = X_pca_model.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
    pca_df["SALARY_TIER"] = ps_encoded["SALARY_TIER"].values

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="SALARY_TIER", color_discrete_map=TIER_COLORS,
        category_orders={"SALARY_TIER": TIER_ORDER},
        title="PCA — Colored by Salary Tier",
        opacity=0.6,
        labels={"SALARY_TIER": "Tier"},
    )
    fig_pca.update_layout(template="plotly_white", margin=dict(t=50, b=20))

    km = KMeans(n_clusters=5, init="random", n_init=10, max_iter=300, random_state=0)
    km.fit(X_2d)
    pca_df["Cluster"] = km.labels_.astype(str)

    fig_km = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="Cluster",
        title="K-Means Clusters (K=5) on PCA Space",
        opacity=0.6,
    )
    fig_km.update_layout(template="plotly_white", margin=dict(t=50, b=20))

    return fig_pca, fig_km


# ---------------------------------------------------------------------------
# Callbacks — Tab 4: Predictor
# ---------------------------------------------------------------------------

@callback(
    Output("pred-salary", "children"),
    Output("pred-tier-badge", "children"),
    Output("pred-proba-chart", "figure"),
    Input("sl-pts", "value"),
    Input("sl-reb", "value"),
    Input("sl-ast", "value"),
    Input("sl-stl", "value"),
    Input("sl-blk", "value"),
    Input("sl-tov", "value"),
    Input("sl-usage", "value"),
    Input("sl-exp", "value"),
    Input("sl-age", "value"),
    Input("sl-win", "value"),
    Input("sl-sent", "value"),
    Input("sl-pos", "value"),
)
def predict_salary(pts, reb, ast, stl, blk, tov, usage, years_exp, age, win, sent, pos):
    pos_enc = {f"POS_{p}": 0 for p in ["C", "PF", "PG", "SF", "SG"]}
    if pos and f"POS_{pos}" in pos_enc:
        pos_enc[f"POS_{pos}"] = 1

    # REDDIT_POST_COUNT held at training mean (0.31) — slider scale is incompatible
    REDDIT_POST_COUNT_MEAN = 0.31

    row = np.array([[
        pts, reb, ast, stl, blk, tov, usage, years_exp, age, win, sent, REDDIT_POST_COUNT_MEAN,
        pos_enc["POS_C"], pos_enc["POS_PF"], pos_enc["POS_PG"],
        pos_enc["POS_SF"], pos_enc["POS_SG"],
    ]], dtype=float)

    row_scaled = scaler.transform(row)

    log_sal = linreg.predict(row_scaled)[0]
    predicted_salary = float(np.exp(log_sal))

    tier_num = rf.predict(row_scaled)[0]
    tier_name = TIER_NUM_TO_NAME.get(int(tier_num), "unknown")
    tier_proba = rf.predict_proba(row_scaled)[0]

    salary_str = f"${predicted_salary:,.0f}"

    badge = dbc.Badge(
        TIER_LABELS.get(tier_name, tier_name),
        color={"budget": "primary", "mid_low": "success", "mid_high": "warning", "max": "danger"}.get(tier_name, "secondary"),
        style={"fontSize": "16px", "padding": "8px 16px"},
    )

    class_names = [TIER_NUM_TO_NAME.get(c, str(c)) for c in rf.classes_]
    proba_fig = px.bar(
        x=tier_proba * 100,
        y=class_names,
        orientation="h",
        labels={"x": "Probability (%)", "y": "Tier"},
        color=class_names,
        color_discrete_map=TIER_COLORS,
        range_x=[0, 100],
    )
    proba_fig.update_layout(
        showlegend=False,
        template="plotly_white",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis={"categoryorder": "array", "categoryarray": TIER_ORDER},
    )

    return salary_str, badge, proba_fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
