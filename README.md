<div align="center">
  <h1>✈️ Travel Package Purchase Prediction</h1>
  <p><i>An XGBoost classification pipeline and Streamlit application predicting customer purchase propensity for holiday travel packages.</i></p>
</div>

<hr>

<h2>📊 Project Overview</h2>
<p>This repository contains an end-to-end machine learning pipeline designed to predict whether a potential customer will purchase a holiday travel package. By analyzing customer demographics, historical trip data, and sales pitch engagement metrics, the project aims to optimize sales team efficiency by identifying high-conversion leads.</p>

<p>The project follows a rigorous data science workflow, starting with exploratory data analysis and a robust preprocessing pipeline utilizing <code>ColumnTransformer</code> for numerical scaling and categorical one-hot encoding. Multiple baseline algorithms (Random Forest, AdaBoost, Gradient Boosting, Decision Tree, and XGBoost) were evaluated to test for overfitting and out-of-sample generalization. XGBoost was selected as the optimal algorithm and fine-tuned using <code>RandomizedSearchCV</code>. The final serialized model is deployed via a Streamlit web application, providing an intuitive interface for real-time customer predictions.</p>

<h2>📂 Repository Structure</h2>
<ul>
  <li><code>Travel.csv</code> & <code>Travel-selected-columns.csv</code>: The raw and processed datasets containing customer demographics and engagement metrics.</li>
  <li><code>project.ipynb</code>: The complete Jupyter Notebook detailing Exploratory Data Analysis (EDA), feature engineering, multi-model baseline testing, and XGBoost hyperparameter tuning.</li>
  <li><code>preprocessor.joblib</code>: The serialized Scikit-Learn <code>ColumnTransformer</code> (StandardScaler and OneHotEncoder).</li>
  <li><code>model.joblib</code>: The serialized and optimized XGBoost classifier.</li>
  <li><code>main.py</code>: The Streamlit application script for the frontend user interface.</li>
</ul>

<h2>⚙️ Installation & Usage</h2>

<h3>1. Clone the repository</h3>
<pre><code>git clone https://github.com/SudarshanPatil2004/travel-package-purchase-prediction.git
cd travel-package-purchase-prediction</code></pre>

<h3>2. Install dependencies</h3>
<p>Ensure you have Python installed, then install the required libraries:</p>
<pre><code>pip install pandas numpy scikit-learn xgboost streamlit joblib</code></pre>

<h3>3. Run the Streamlit Application</h3>
<p>Launch the frontend interface to make real-time predictions:</p>
<pre><code>streamlit run main.py</code></pre>

<h2>🧠 Model Architecture & Methodology</h2>
<ol>
  <li><strong>Feature Engineering:</strong> Engineered features such as <code>TotalVisiting</code> to
