<div align="center">

# Supervised Learning with Regression Algorithms

**Comparing Linear, Polynomial, and Ridge Regression, built from scratch with NumPy**

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=fff)](#)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Course](https://img.shields.io/badge/CMSC_170-Final_Project-blueviolet?style=for-the-badge)

[About](#about) • [Project Files](#project-files) • [Installation](#installation) • [Running the Site](#running-the-site) • [Credits](#credits)

</div>

---

## About

A final project for **CMSC 170** that compares the predictions of three (3) supervised machine learning algorithms:

| # | Algorithm | Description |
|---|-----------|-------------|
| 1 | **Linear Regression** | Fits a straight-line trend to the data |
| 2 | **Polynomial Regression** | Captures curved trends with higher-degree terms |
| 3 | **Ridge Regression** | Linear model with L2 regularization to control overfitting |

The results are presented in an interactive **Streamlit** site where you can try the models yourself.

---

## Project Files

```
.
├── csv/                  # Dataset used for the project
├── lib/
│   ├── Regressions.py    # All regression algorithms
│   └── utility.py        # Helper functions: parser and performance metrics
├── site_pages/           # Pages of the site
│                         #   About, Discussion, Methodology,
│                         #   Predictor, References, Results
├── main.py               # Calls and fits the models
└── app.py                # Program entry
```

---

## Installation

Clone the repository and install the requirements.

<details open>
<summary><b>Windows / Linux</b></summary>

```bash
git clone https://github.com/Fylows/CMSC-170-Project.git
cd CMSC-170-Project
pip install --upgrade pip
pip3 install -r requirements.txt
```

</details>

<details>
<summary><b>macOS</b></summary>

```bash
git clone https://github.com/Fylows/CMSC-170-Project.git
cd CMSC-170-Project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

---

## Running the Site

```bash
streamlit run app.py
```

Then open the local URL that Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## Credits

This project was made by:

- [Ljiel Saplan]()
- [Francis Rabulan]()
- [Phoebe Serrano]()

<div align="center">

*Made for CMSC 170*

</div>
