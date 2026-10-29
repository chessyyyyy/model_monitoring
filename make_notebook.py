"""Writes drift_demo.ipynb, the Google Colab notebook for the live drift demonstration.

Usage: python make_notebook.py
"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# Drift demonstration

A credit model is trained on months 1–12 and then frozen. From month 19 the world changes in one of three ways. Each animation shows what that change does to the data and to the model.

**Each picture has four panels**

- **A · Inputs:** one applicant feature, training months (grey) vs the current month. PSI measures how far it moved; above 0.25 means investigate.
- **B · Relationship:** the actual default rate by credit-score band, before (grey) and now (colour), and what the frozen model predicts (dashed).
- **C · Default rate:** actual vs predicted, month by month.
- **D · Ranking:** the model's AUC, month by month (0.5 = coin flip, 1.0 = perfect).

Defaults are only known 6 months after a loan is made, so the last 6 months in C and D are shaded as "not yet known".

**To run:** Runtime → Run all. Setup takes under a minute; each animation takes about 15 seconds to build.
""")

md("## 1. Setup")
code("""
# Option A: clone the project from GitHub (replace with your repository's URL).
REPO_URL = "https://github.com/YOUR-USERNAME/drift-monitoring.git"

# Option B: upload drift-monitoring.zip with the folder icon on the left, then set USE_ZIP = True.
USE_ZIP = False

import os
if not os.path.exists("demo_plots.py"):
    if USE_ZIP:
        !unzip -q -o drift-monitoring.zip
    elif not os.path.exists("drift-monitoring"):
        !git clone -q $REPO_URL
    %cd drift-monitoring
!pip install -q -r requirements.txt
""")
code("""
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from demo_plots import simulate, draw, animate, summary_table

plt.rcParams["animation.embed_limit"] = 60   # MB, so the animations fit in the notebook
""")

md("""
## 2. The model

Four inputs go in (income, debt-to-income, credit score, years employed). One output comes out: the probability that the borrower defaults. The model is a logistic regression trained on 12 months of loans, then frozen.

The demo uses 10,000 applicants a month (the experiment uses 2,000) so the pictures are smoother.
""")
code("""
base = simulate("none")
train = base["train"]
print(f"Training loans: {len(train):,}   default rate: {train['default'].mean():.1%}")
display(train[["income", "dti", "credit_score", "years_employed", "default", "pred"]].head()
        .rename(columns={"pred": "predicted probability"}).round(3))
lr = base["model"][-1]
print("Fitted coefficients (per standard deviation):",
      dict(zip(["log income", "DTI", "credit score", "years employed"], lr.coef_[0].round(2).tolist())))
""")

md("""
## 3. Data drift: the applicants change

From month 19, incomes fall 30% and credit scores drop 30 points. **The risk formula itself does not change.**

Watch: panel A shifts and PSI goes above 0.25, but the model's predictions in B and C follow reality and AUC in D stays put. The model is still right.
""")
code("""
data = simulate("data")
HTML(animate(data).to_jshtml())
""")

md("""
## 4. Label drift: the default rate doubles

From month 19, 16% of borrowers default instead of 8%. Defaulters and non-defaulters still look the way they always did.

Watch: the inputs in A barely move (PSI ≈ 0.01). In C the model keeps predicting about 10% while 16% default, but AUC in D is unchanged: the model still ranks borrowers correctly, it just underestimates everyone's risk.
""")
code("""
label = simulate("label")
HTML(animate(label).to_jshtml())
""")

md("""
## 5. Concept drift: the same applicant means different risk

From month 19, credit score and income stop protecting against default; debt-to-income matters more. The overall default rate stays at 8%.

Watch: nothing changes in A (PSI ≈ 0) or C. Only B shows the problem: the curve flattens while the model still predicts the old one. AUC in D falls from 0.84 to about 0.75, but in reality the bank only sees this 6 months later.
""")
code("""
concept = simulate("concept")
HTML(animate(concept).to_jshtml())
""")

md("""
## 6. Explore any month

Pick a scenario and drag the slider. Useful for answering questions live.
""")
code("""
import ipywidgets as widgets
sims = {"data": data, "label": label, "concept": concept, "none": base}

@widgets.interact(scenario=widgets.Dropdown(options=list(sims), value="concept"),
                  month=widgets.IntSlider(value=19, min=13, max=36, step=1))
def explore(scenario, month):
    draw(sims[scenario], month)
    plt.show()
""")

md("""
## 7. The simple measures, before and after

Month 18 (before drift) vs month 30 (after). Each drift type moves a different measure, so you need all of them.
""")
code("""
summary_table()
""")

md("""
## 8. (Optional) Rerun the full experiment

The repository already contains the results. To reproduce them: 6 scenarios × 30 random seeds, about 15 minutes on Colab's free tier.
""")
code("""
# !python run_experiment.py --seeds 30 --workers 2
# !python make_figures.py
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
    c["source"] = [line + "\n" for line in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
with open("drift_demo.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote drift_demo.ipynb with", len(cells), "cells")
