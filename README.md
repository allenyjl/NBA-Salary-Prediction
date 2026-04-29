# NBA Salary Prediction

**CIS 2450: Big Data Analytics — Final Project**

A full end-to-end data science pipeline that predicts NBA player salaries from five seasons (2020–21 through 2024–25) of game-log performance stats, player bio data, and Reddit sentiment. The project covers data collection, wrangling, exploratory analysis, supervised and unsupervised modeling, and an interactive Plotly Dash dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Findings](#key-findings)
3. [Repository Structure](#repository-structure)
4. [Data Sources](#data-sources)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Notebook 1 — Data Collection](#notebook-1--data-collection)
   - [Notebook 2 — Data Wrangling](#notebook-2--data-wrangling)
   - [Notebook 3 — Exploratory Data Analysis](#notebook-3--exploratory-data-analysis)
   - [Notebook 4 — Modeling](#notebook-4--modeling)
6. [Model Results](#model-results)
7. [Interactive Dashboard](#interactive-dashboard)
8. [Setup & Installation](#setup--installation)
9. [Running the Project](#running-the-project)
10. [Dataset Schema](#dataset-schema)
11. [Design Decisions](#design-decisions)

---

## Project Overview

NBA player contracts are the product of many interacting signals — on-court production, age and experience, market timing, team needs, and public profile. This project asks: **how well can we predict a player's salary from objective, measurable data?**

We built a pipeline that:

- Collects ~127,000 player game logs from the official NBA stats API across 5 seasons
- Scrapes season salary and player bio data from Basketball-Reference across 150 team pages
- Gathers and sentiment-scores ~69,000 Reddit posts from r/nba using DistilBERT
- Cleans and joins all three sources into a single 119,227-row, 54-column Parquet dataset
- Performs 10-section exploratory analysis with DuckDB-powered SQL queries
- Trains 7 models (4 regression, 3 classification) with PCA and K-Means for unsupervised exploration
- Serves results through a 4-tab interactive Plotly Dash app with a live salary predictor

**Dataset size:** 119,227 player-game rows · 828 unique players · 5 seasons · salary range $5,318–$55,761,216

---

## Key Findings

| Finding | Detail |
|---|---|
| **Scoring dominates contracts** | PTS/36 is the single strongest salary predictor (r ≈ 0.58 at player-season level; largest OLS coefficient after standardization) |
| **Stats explain ~42% of variance** | Best regression R² = 0.4184 (Ridge) — the remaining ~58% reflects experience, market timing, team context, and positional scarcity |
| **Experience matters almost as much as scoring** | `YEARS_EXP` has the second-largest OLS coefficient, confirming veterans command a premium beyond their box-score output |
| **Random Forest beats simpler classifiers** | 62.0% accuracy on 4-class salary tier prediction vs. 55.8% Decision Tree and 56.9% Logistic Regression; 37 points above the 25% random baseline |
| **Lasso over-regularizes** | At alpha=1.0, Lasso drives all coefficients to zero and predicts the mean (R² = −0.0009) — multicollinearity between per-36 stats makes L1 inappropriate at this regularization level |
| **Reddit sentiment adds negligible signal** | Including sentiment features hurts OLS regression (ΔR² = −0.015) and gives only a marginal boost to Random Forest (+0.003 accuracy); salary volume correlates with performance, not sentiment independently |
| **Salary is highly right-skewed** | Skewness = 1.86; log-transforming salary produces a near-normal distribution used as the regression target |
| **Max-contract outliers are real, not noise** | 10.2% of player-seasons are IQR-flagged outliers — these are legitimate max/supermax deals and are kept in the training set |
| **NBA salaries inflated ~35% over 5 seasons** | Mean salary grew from $7.7M (2020–21) to $10.4M (2024–25); median from $3.6M to $5.0M, tracking salary cap growth |

---

## Repository Structure

```
NBA-Salary-Prediction/
│
├── notebooks/
│   ├── 01_data_collection.ipynb   # Pulls game logs, salaries, bios, Reddit sentiment
│   ├── 02_data_wrangling.ipynb    # Cleaning, entity linking, DuckDB join, feature engineering
│   ├── 03_eda.ipynb               # 10-section exploratory analysis with 10 saved figures
│   └── 04_modeling.ipynb          # PCA, K-Means, regression, and classification models
│
├── data/
│   ├── raw/                       # CSVs produced by Notebook 1 (not committed — re-run to generate)
│   │   ├── game_logs_2020-21.csv  # ~23,000 rows each
│   │   ├── game_logs_2021-22.csv
│   │   ├── game_logs_2022-23.csv
│   │   ├── game_logs_2023-24.csv
│   │   ├── game_logs_2024-25.csv
│   │   ├── salaries_2020-21.csv   # ~500–650 rows each
│   │   ├── salaries_2021-22.csv
│   │   ├── salaries_2022-23.csv
│   │   ├── salaries_2023-24.csv
│   │   ├── salaries_2024-25.csv
│   │   ├── player_bios.csv        # 569 players with position, birth date, height, weight, experience
│   │   └── reddit_sentiment.csv   # 69,336 r/nba posts with DistilBERT sentiment scores
│   │
│   └── processed/
│       ├── nba_player_games_clean.parquet   # Final 119,227 × 54 dataset (primary input to dashboard)
│       ├── nba_player_games_clean.csv       # CSV version of the same dataset
│       └── fig*.png                         # EDA and model figures saved from notebooks
│
├── dashboard.py                   # Plotly Dash app — run this to launch the interactive dashboard
├── requirements.txt               # Python dependencies
└── README.md
```

> **Note:** Raw data files are large and are not committed to version control. Run `01_data_collection.ipynb` to regenerate them. The processed Parquet file is the only file required to run the dashboard.

---

## Data Sources

### 1. NBA API (`nba_api`)
- **Library:** `nba_api` (unofficial Python wrapper for `stats.nba.com`)
- **Endpoint:** `PlayerGameLogs` — pulls all player game logs for an entire season in one call
- **Coverage:** 5 seasons × ~25,000 rows = ~127,000 total rows before cleaning
- **Rate limiting:** 2-second sleep between season calls; up to 3 automatic retries on failure
- **Key columns:** `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `MIN`, `FGM/FGA/FG_PCT`, `FG3M/FG3A/FG3_PCT`, `FTM/FTA/FT_PCT`, `PLUS_MINUS`, `WL`, `MATCHUP`, `GAME_DATE`

### 2. Basketball-Reference (BBRef)
- **Salaries:** Scraped from each team's season page (`#salaries2` table) — 30 teams × 5 seasons = 150 pages
- **Player bios:** Scraped from the `#roster` table on each 2024–25 team page for position, birth date, height, weight, and years of experience
- **Rate limiting:** 5-second sleep between requests (12 req/min, well under BBRef's 20 req/min limit)
- **Deduplication:** Traded players appear on multiple team pages with the same salary and are deduplicated by `BBREF_ID` (extracted from the href anchor)
- **Hidden tables:** BBRef occasionally wraps tables in HTML comments; the scraper parses both standard and comment-wrapped tables via BeautifulSoup

### 3. Reddit (r/nba) + DistilBERT
- **API:** Reddit public JSON search endpoint — no OAuth required
- **Query:** Quoted player name search (`"Player Name"`) restricted to r/nba, up to 50 posts per player
- **Coverage:** 950 unique players searched; 907 with at least one post found; 69,336 total text snippets (titles + self-text)
- **Sentiment scoring:** `distilbert-base-uncased-finetuned-sst-2-english` via HuggingFace Transformers
  - Positive label → score in [0, +1]; Negative label → score in [−1, 0]
  - Batched inference at batch size 64 (~64 minutes total for 69,336 snippets)
- **Aggregation:** Sentiment averaged to player-week level; each game row receives the average sentiment from posts about that player during the same calendar week. Players/weeks with no data receive a neutral 0.

---

## Pipeline Walkthrough

### Notebook 1 — Data Collection

`notebooks/01_data_collection.ipynb`

Produces 12 raw CSV files:

1. **Game logs** (`game_logs_YYYY-YY.csv` × 5): Each file contains one season's player game logs from `nba_api`. Already-fetched seasons are skipped, making re-runs safe and idempotent.

2. **Salaries** (`salaries_YYYY-YY.csv` × 5): Scraped from BBRef team roster pages. A `BBREF_ID` is extracted from each player's anchor tag for reliable cross-source joining in Notebook 2.

3. **Player bios** (`player_bios.csv`): Position, birth date, height, weight, years of experience, and college for all 30 current-season rosters. Deduplicated by `BBREF_ID`.

4. **Reddit sentiment** (`reddit_sentiment.csv`): Raw post titles and self-text for each salary-listed player, with `SENTIMENT_LABEL` (POSITIVE/NEGATIVE) and `SENTIMENT_SCORE` (continuous, −1 to +1) from DistilBERT.

---

### Notebook 2 — Data Wrangling

`notebooks/02_data_wrangling.ipynb`

**Step 1 — Load & concat game logs:** Polars reads 5 CSVs and stacks them with `diagonal_relaxed` to handle minor schema differences across seasons. Result: 127,694 rows × 31 columns.

**Step 2 — Clean game logs:** ISO datetime strings parsed to `Date` type; numeric stats cast to `Float32` (saves ~50% memory vs. Float64); DNP (Did Not Play) rows filtered by `MIN > 0`. Only 10 DNP rows removed — the API is clean.

**Step 3 — Clean salaries:** Drops zero/null salaries (two-way contract placeholders). Combines all 5 seasons to 2,677 rows.

**Step 4 — Entity linking:** Player names differ between nba_api and BBRef (e.g. "Marcus Morris Sr." vs. "Marcus Morris", special characters in international names). Resolution strategy:
- Lowercase + strip whitespace
- Remove name suffixes (Jr., Sr., II, III, IV, V)
- Remove punctuation
- Apply a manual override dictionary for known nickname mismatches (e.g. "cam reddish" → "cameron reddish")
- **RapidFuzz fallback:** `token_sort_ratio ≥ 88` for remaining unmatched names (catches Unicode encoding differences like "Nikola Jokić" → "Nikola Jokiä") — 86 fuzzy matches found

**Step 5 — DuckDB SQL join:** `INNER JOIN` on `(PLAYER_NAME_NORM, SEASON)` using DuckDB. The inner join naturally excludes two-way/G-League players who lack a BBRef salary record. Result: 119,227 rows.

**Step 5b — Reddit join:** Reddit sentiment aggregated to player-week (Monday-anchored). Each game row receives the sentiment from that player's posts during the same calendar week. 14.6% of game rows (17,408) receive non-zero sentiment data.

**Step 6 — Feature engineering:**
- **Per-36 stats:** `PTS_PER36`, `REB_PER36`, `AST_PER36`, `STL_PER36`, `BLK_PER36`, `TOV_PER36` (raw stat / MIN × 36)
- **Usage proxy:** `(FGA + 0.44×FTA + TOV) / MIN` — approximates NBA usage rate without needing team totals
- **Binary flags:** `WIN` (W/L → 1/0), `IS_HOME` (MATCHUP contains "vs." → 1/0)
- **LOG_SALARY:** `ln(SEASON_SALARY)` — regression target; addresses right-skew
- **AGE_AT_GAME:** `(GAME_DATE − BIRTH_DATE) / 365.25` — exact age at time of each game

**Step 7 — Salary tiers (DuckDB NTILE):** 4-class classification target created via `NTILE(4) OVER (PARTITION BY SEASON ORDER BY SEASON_SALARY)`, ensuring balanced tiers within each season and accounting for salary cap growth year over year.

| Tier | Percentile | Count |
|---|---|---|
| `budget` | Bottom 25% | 29,809 |
| `mid_low` | 25–50% | 29,809 |
| `mid_high` | 50–75% | 29,805 |
| `max` | Top 25% | 29,804 |

**Output:** `nba_player_games_clean.parquet` — 119,227 rows × 54 columns, 5.8 MB.

---

### Notebook 3 — Exploratory Data Analysis

`notebooks/03_eda.ipynb`

Ten analysis sections, each producing at least one saved figure and a markdown summary with a modeling decision.

| Section | Key Finding | Figure |
|---|---|---|
| **1. Dataset Overview** | 2020–21 has fewer rows (21,134) due to the COVID-shortened 72-game season; all others are ~24k–25k | `fig1_rows_per_season.png` |
| **2. Salary Distribution** | Skewness = 1.86; log-salary is approximately normal — confirms LOG_SALARY as the right regression target | `fig2_salary_distribution.png` |
| **3. Salary by Position** | PGs have highest median salary ($7.6M); bimodal distribution visible in violin plots across all positions | `fig3_salary_by_position.png` |
| **4. Age vs. Salary** | LOWESS trend peaks at ages 27–30; rookies cluster at league minimum regardless of talent | `fig4_age_vs_salary.png` |
| **5. Performance vs. Salary** | PPG strongest (r = 0.74), APG second (r = 0.61), RPG weakest (r = 0.49); wide scatter due to rookie contracts and legacy deals | `fig5_perf_vs_salary.png` |
| **6. Correlation Heatmap** | TOV correlates highly with salary (r = 0.65) — it's a usage signal, not a penalty; REB_PER36 near-zero correlation | `fig6_correlation_heatmap.png` |
| **7. Outlier Detection** | 248 (10.2%) high-salary outliers via DuckDB `PERCENTILE_CONT`; zero low-salary outliers (two-way players already removed) | `fig7_salary_outliers.png` |
| **8. Season Salary Trends** | Mean salary $7.7M → $10.4M across 5 seasons; mean-median gap widens, indicating increasing concentration at the top | `fig8_season_salary_trends.png` |
| **9. Reddit Sentiment Distribution** | Post counts are highly right-skewed — a handful of stars dominate r/nba discussion; 69,336 posts across 907 players | `fig9_sentiment_distribution.png` |
| **10. Sentiment vs. Salary** | Weak correlation between sentiment score and salary; post volume correlates with salary but is largely redundant with performance stats | `fig10_sentiment_vs_salary.png` |

---

### Notebook 4 — Modeling

`notebooks/04_modeling.ipynb`

**Feature preparation:** Game-level rows aggregated to 2,427 player-season records. After dropping rows missing bio data (players not on 2024–25 BBRef rosters), **1,765 player-season rows** are used for modeling. 24 features total after one-hot encoding position.

**Train/test split:** 80/20 stratified random split (random_state=42) → 1,412 train / 353 test samples. All features standardized with `StandardScaler` fit only on training data.

#### Unsupervised Learning

**PCA:**
- 5 principal components explain 55.75% of variance (PC1: 18.2%, PC2: 15.1%)
- PC1 separates high-usage scorers from low-minute role players
- PC2 separates frontcourt (big) vs. backcourt (guard) profiles
- Used as preprocessing in the Logistic Regression pipeline

**K-Means Clustering (K=5):**
- Elbow method on the first 2 PCA components suggests K=5 as the natural knee
- 5 clusters correspond roughly to: stars, role-player guards, defensive bigs, stretch bigs, and fringe/G-League borderline players
- Cluster labels used as engineered features in downstream models

#### Supervised — Regression

Target: `LOG_SALARY` (continuous). All models use the same 24-feature, standardized train/test split.

| Model | Test R² | RMSE | MAE |
|---|---|---|---|
| OLS Linear Regression | 0.4178 | 1.0408 | 0.7268 |
| **Ridge (L2, α=1.0)** | **0.4184** | **1.0402** | **0.7263** |
| ElasticNet (α=1.0, l1=0.5) | 0.2171 | 1.2069 | 0.8950 |
| Lasso (L1, α=1.0) | −0.0009 | 1.3646 | 1.0447 |

Ridge is the best regression model, marginally improving OLS by shrinking correlated per-36 coefficients. Lasso collapses entirely — multicollinearity among the per-36 stats causes the L1 penalty at α=1.0 to zero out nearly all coefficients, effectively predicting the mean.

**Top OLS coefficients (standardized):**
1. `PTS_PER36` (positive) — scoring is the primary salary driver
2. `YEARS_EXP` (positive) — veterans command experience premiums
3. `USAGE_PROXY` (negative after controlling for raw stats) — high-usage but inefficient players are penalized

#### Supervised — Classification

Target: `SALARY_TIER` (4 classes: budget / mid_low / mid_high / max). Baseline (random guess) = 25%.

| Model | Test Accuracy |
|---|---|
| Decision Tree (max_depth=4, tuned) | 55.81% |
| Logistic Regression (PCA pipeline) | 56.94% |
| **Random Forest (200 trees, max_depth=10)** | **62.04%** |

Random Forest is the best classifier — the ensemble averaging of 200 trees substantially reduces variance vs. a single Decision Tree. Budget and Max tiers are predicted most reliably (F1 ≈ 0.64 and 0.80 respectively) because their statistical profiles are most distinct. Mid-Low vs. Mid-High is the hardest boundary — the percentile cutoff is inherently noisy.

**Reddit feature impact (ablation):**

| Metric | With Reddit | Without Reddit | Δ |
|---|---|---|---|
| OLS R² | 0.4178 | 0.4328 | −0.0150 |
| RF Accuracy | 0.6204 | 0.6175 | +0.0029 |

Reddit features slightly hurt regression (post volume correlates with performance stats, adding noise) and provide negligible classification gain. They are retained in the dashboard's predictor for completeness.

---

## Interactive Dashboard

`dashboard.py` — a Plotly Dash application with 4 tabs.

**Run it:**
```bash
python dashboard.py
# Open http://localhost:8050 in your browser
```

The app loads `data/processed/nba_player_games_clean.parquet` at startup, trains three live models (Linear Regression, Ridge, Random Forest), and launches on port 8050.

### Tab 1 — Overview
High-level stats cards (total game logs, unique players, seasons, salary range) and a bar chart of game logs per season. Includes a data sources summary and salary tier color legend.

### Tab 2 — EDA Explorer
Interactive dropdown to select from 7 EDA views with contextual controls:

| View | Controls |
|---|---|
| Salary Distribution | Raw vs. Log scale toggle |
| Salary by Position | Season filter |
| Age vs. Salary | Season filter |
| Performance vs. Salary | Stat selector (PTS/REB/AST per 36) |
| Season Salary Trends | — |
| Correlation Heatmap | — |
| Reddit Sentiment vs. Salary | Season filter |

Each view displays a key finding text beneath the chart.

### Tab 3 — Modeling Results
- Summary table of all regression and classification model scores, with best scores highlighted in green
- Deep-dive dropdown to explore: OLS coefficients, Ridge coefficients, Decision Tree confusion matrix, Logistic Regression confusion matrix, or Random Forest feature importances
- Interpretation text for each model
- Side-by-side PCA scatter (colored by salary tier) and K-Means cluster scatter

### Tab 4 — Salary Predictor
Live salary predictor with sliders for all model inputs:
- **Performance sliders:** Points, Rebounds, Assists, Steals, Blocks, Turnovers (all per 36 minutes)
- **Player profile sliders:** Usage Proxy, Years of Experience, Age, Win Rate
- **Reddit presence slider:** Average Sentiment (−1 to +1)
- **Position dropdown:** PG / SG / SF / PF / C

Outputs update in real-time:
- Predicted salary in dollars (from Ridge regression via log-salary inversion)
- Predicted salary tier badge with color coding (from Random Forest)
- Tier probability bar chart showing confidence across all 4 tiers

> **Note:** The predictor's models are trained on all available data (no holdout). The R²=0.42 / 62% accuracy figures on the Modeling tab are from notebook models with a proper held-out test set.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

# Install all requirements
pip install -r requirements.txt
```

### Additional dashboard dependencies

The dashboard uses a few libraries not in the base `requirements.txt`:

```bash
pip install dash dash-bootstrap-components plotly scikit-learn
```

### (Optional) Reddit sentiment collection

The Reddit notebook section uses HuggingFace Transformers for DistilBERT inference:

```bash
pip install transformers torch
```

Sentiment inference for all 950 players takes approximately 64 minutes on a CPU. A pre-computed `reddit_sentiment.csv` is expected in `data/raw/`.

---

## Running the Project

### Option A — Run the dashboard directly (recommended)

The processed Parquet file is the only dependency:

```bash
python dashboard.py
```

Visit [http://localhost:8050](http://localhost:8050).

### Option B — Run the full pipeline from scratch

Execute the notebooks in order. Each notebook skips already-completed steps (idempotent):

```bash
# 1. Collect raw data (~45–60 min due to scraping rate limits + DistilBERT)
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb

# 2. Clean, join, and engineer features (~2–5 min)
jupyter nbconvert --to notebook --execute notebooks/02_data_wrangling.ipynb

# 3. Exploratory analysis (~1–2 min, saves 10 figures)
jupyter nbconvert --to notebook --execute notebooks/03_eda.ipynb

# 4. Train and evaluate all models (~5–10 min)
jupyter nbconvert --to notebook --execute notebooks/04_modeling.ipynb

# 5. Launch dashboard
python dashboard.py
```

Or open them interactively in JupyterLab:

```bash
jupyter lab
```

---

## Dataset Schema

Final processed dataset: `data/processed/nba_player_games_clean.parquet`
**Shape:** 119,227 rows × 54 columns

| Column | Type | Description |
|---|---|---|
| `SEASON` | String | NBA season string, e.g. "2023-24" |
| `PLAYER_ID` | Int64 | NBA API player identifier |
| `PLAYER_NAME` | String | Player's full name (nba_api format) |
| `TEAM_ABBREVIATION` | String | Team code at time of game |
| `GAME_ID` | Int64 | Unique game identifier |
| `GAME_DATE` | Date | Date of the game |
| `MATCHUP` | String | e.g. "LAL vs. GSW" or "LAL @ GSW" |
| `WL` | String | Game result: "W" or "L" |
| `MIN` | Float32 | Minutes played |
| `FGM / FGA / FG_PCT` | Float32 | Field goal makes, attempts, percentage |
| `FG3M / FG3A / FG3_PCT` | Float32 | Three-point makes, attempts, percentage |
| `FTM / FTA / FT_PCT` | Float32 | Free throw makes, attempts, percentage |
| `OREB / DREB / REB` | Float32 | Offensive, defensive, total rebounds |
| `AST / TOV / STL / BLK` | Float32 | Assists, turnovers, steals, blocks |
| `PTS` | Float32 | Points scored |
| `PLUS_MINUS` | Float32 | +/− for the game |
| `NBA_FANTASY_PTS` | Float32 | NBA fantasy point total |
| `PLAYER_NAME_NORM` | String | Normalized name used for entity linking |
| `SEASON_SALARY` | Float64 | Player's annual salary for that season (USD) |
| `BBREF_ID` | String | Basketball-Reference player ID (e.g. "curryst01") |
| `REDDIT_SENTIMENT_AVG` | Float64 | Mean DistilBERT sentiment that week (0 if no posts) |
| `REDDIT_POST_COUNT` | UInt32 | Number of r/nba posts that week (0 if none) |
| `PTS_PER36` | Float32 | Points per 36 minutes |
| `REB_PER36` | Float32 | Rebounds per 36 minutes |
| `AST_PER36` | Float32 | Assists per 36 minutes |
| `STL_PER36` | Float32 | Steals per 36 minutes |
| `BLK_PER36` | Float32 | Blocks per 36 minutes |
| `TOV_PER36` | Float32 | Turnovers per 36 minutes |
| `USAGE_PROXY` | Float32 | `(FGA + 0.44×FTA + TOV) / MIN` |
| `WIN` | Int8 | 1 if the player's team won, 0 otherwise |
| `IS_HOME` | Int8 | 1 if home game ("vs." in MATCHUP), 0 otherwise |
| `LOG_SALARY` | Float64 | `ln(SEASON_SALARY)` — regression target |
| `POSITION_SIMPLE` | String | Primary position: PG / SG / SF / PF / C (null for ~18% not on 2024–25 rosters) |
| `BIRTH_DATE_DT` | Date | Player's date of birth (from BBRef) |
| `YEARS_EXP` | Int16 | NBA years of experience (0 = rookie) |
| `HEIGHT` | String | Height string, e.g. "6-6" |
| `WEIGHT` | Int64 | Weight in pounds |
| `AGE_AT_GAME` | Float32 | Player's exact age (years) at game date |
| `SALARY_TIER_NUM` | Int64 | Numeric tier: 1 (budget) through 4 (max) |
| `SALARY_TIER` | String | Tier label: budget / mid_low / mid_high / max |

---

## Design Decisions

**Why Polars instead of pandas for data loading?**
Polars is significantly faster on large CSVs (lazy evaluation, columnar Arrow backend) and catches type mismatches earlier. Pandas is used for sklearn compatibility downstream.

**Why DuckDB for the join?**
DuckDB provides SQL semantics over Polars/pandas DataFrames without a database server. It satisfies the course requirement for SQL usage while keeping the pipeline fully local and reproducible. The `NTILE(4)` window function for salary tier creation and `PERCENTILE_CONT` for outlier detection are natural fits for SQL.

**Why INNER JOIN instead of LEFT JOIN?**
An inner join naturally excludes two-way contract players, 10-day signings, and G-League players who appear in game logs but have no BBRef salary record. These players have incomparable contracts and would distort both regression and classification targets.

**Why NTILE(4) per season for salary tiers?**
If tiers were defined on the full multi-season dataset, the growing salary cap would cause earlier seasons to be over-represented in the "budget" tier simply due to lower nominal salaries. Partitioning by season makes each tier represent the same relative percentile within its year.

**Why keep max-contract outliers in the training set?**
Max-contract players are real, legally-structured contracts — not measurement errors. Removing them would bias regression predictions downward and undermine classification on the "max" tier, which is exactly the class we want to identify.

**Why per-36 stats instead of raw totals?**
Raw totals are confounded by minutes — a player averaging 2 points in 5 minutes per game is not comparable to one averaging 2 points in 30 minutes. Per-36 stats normalize production rate, reducing collinearity with `MIN` and making the features more comparable across role players and starters.

**Why log-transform the salary target?**
Raw salary has a skewness of 1.86. OLS regression assumes normally distributed residuals — a highly right-skewed target produces heteroscedastic residuals that violate this assumption. Log-transforming salary compresses the tail and produces a near-normal distribution. RMSE is reported in log-dollar units and can be exponentiated to recover approximate dollar-scale errors.
