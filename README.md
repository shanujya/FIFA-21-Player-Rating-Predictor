# ⚽ FIFA 21 Player Rating Predictor: Advanced Data Preprocessing & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

## 📌 Project Overview
Predicting a professional football player's Overall Rating (`OVA`) based on their raw physical and technical attributes. 

While the FIFA 21 dataset is a popular starting point for data analysis, this project goes far beyond basic exploratory data analysis (EDA). It focuses heavily on **building a professional-grade preprocessing pipeline** to handle incredibly noisy string data, mixed metric/imperial measurements, extreme financial outliers, and the mitigation of **Target Leakage**.

The final result is a mathematically pure Machine Learning engine that successfully reverse-engineers a player's rating strictly using fundamental footballing stats, achieving an $R^2$ score of 96.3%.

## 🎯 Key Highlights & Critical Thinking
* **Advanced String Manipulation:** Developed custom Python functions to extract continuous numerical data from messy financial strings (e.g., converting `€105.5M` to `105500000.0`) and vectorized conditional logic to unify imperial and metric measurement systems (lbs/kg, inches/cm).
* **Mitigating Target Leakage (The "Strict" ML Model):** Identified and systematically removed deterministic, game-engine-generated features (`Potential`, `Value`, `Wage`, `Release Clause`) prior to the train-test split. This forced the model to evaluate players purely on physical/technical merit rather than reverse-engineering the financial outputs of the game.
* **Outlier Preservation via Winsorization:** Applied Interquartile Range (IQR) bounds to cap extreme financial outliers, preserving the mathematical integrity of the dataset without deleting the critical profiles of "superstar" players (e.g., Messi, Ronaldo).
* **Algorithmic Feature Selection:** Utilized Recursive Feature Elimination (RFE) paired with a Random Forest Regressor to reduce the dataset from 77 noisy dimensions down to the Top 15 most mathematically predictive features.

## ⚙️ The Preprocessing Pipeline
1. **Data Cleaning:** Removal of high-sparsity columns (e.g., `Loan Date End`), imputation of missing values (Median imputation for `Hits`), and stripping of hidden text artifacts/symbols (`★`, `\n`).
2. **Feature Engineering:** Extraction of calculable numeric durations (`Contract_Length`, `Joined_Year`) from string date ranges, and creation of composite features (`BMI`).
3. **Categorical Encoding:** Converted ordinal text ratings (`High/Medium/Low`) and ordinal star ratings to integers to preserve natural hierarchical weighting.
4. **Feature Scaling:** Min-Max scaling applied to continuous variables to unify the variance matrix to a `[0, 1]` scale.
5. **Data Split:** Strict 80/20 Train/Test split initialized prior to statistical feature selection to prevent data leakage.

## 📊 Model Performance & Results
A **Random Forest Regressor** was selected to capture the non-linear relationships inherent in the data (e.g., the parabolic trajectory of Age vs. Overall Rating). 

* **R-Squared ($R^2$):** `0.963` (96.3% of the variance explained)
* **Mean Absolute Error (MAE):** `0.999` rating points. 
* **Interpretation:** The model is capable of predicting a player's Overall Rating within exactly 1.0 point, relying *only* on core stats like passing, shooting, height, and stamina.

## 🛠️ Technologies Used
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualisation:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (RandomForestRegressor, RFE, MinMaxScaler, train_test_split)

## 🚀 How to Run the Notebook
1. Clone the repository: `git clone https://github.com/YourUsername/Your-Repo-Name.git`
2. Ensure you have the `fifa21 datatset.csv` in the root directory.
3. Install the required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
4. Open the Jupyter Notebook: `jupyter notebook DPP_Assignment.ipynb`
5. Run the cells sequentially to observe the pipeline transformations and final model evaluation.
