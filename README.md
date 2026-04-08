<h1>💳 Credit Card Customer Prediction Project</h1>

<p>
A Machine Learning web application that predicts whether a customer is a <b>Good Customer</b> or <b>Bad Customer</b>
based on financial and demographic data. This project demonstrates a complete ML pipeline,
from preprocessing and model building to deployment using Flask.
</p>

<h2>📌 Table of Contents</h2>
<ul>
    <li>Acknowledgements</li>
    <li>Project Overview</li>
    <li>Dataset Description</li>
    <li>Project Structure</li>
    <li>Technology Stack</li>
    <li>Implementation Steps</li>
    <li>Model Development</li>
    <li>Deployment Process</li>
    <li>Results</li>
    <li>Future Enhancements</li>
    <li>Conclusion</li>
</ul>

<h2>🙏 Acknowledgements</h2>
<ul>
    <li><b>Dataset:</b> Credit Card Customer Dataset</li>
    <li><b>Author:</b> Sachin Muskudi</li>
    <li>Thanks to open-source libraries like Scikit-learn and Flask</li>
</ul>

<h2>📊 Project Overview</h2>

<h3>Objective</h3>
<p>
Build a machine learning model to classify customers as good or bad based on their credit-related attributes,
helping in risk assessment and decision-making.
</p>

<h3>Key Features</h3>
<ul>
    <li>Interactive web interface using Flask</li>
    <li>Real-time prediction system</li>
    <li>Multiple financial and demographic input features</li>
    <li>Fast and efficient prediction</li>
</ul>

<h2>📁 Dataset Description</h2>
<ul>
    <li><b>File:</b> creditcard.csv</li>
    <li><b>Type:</b> Structured tabular dataset</li>
    <li><b>Use Case:</b> Classification (Good vs Bad Customer)</li>
</ul>

<h3>Preprocessing Steps</h3>
<ul>
    <li>Handled missing values</li>
    <li>Converted categorical variables into numerical format</li>
    <li>Feature selection and transformation</li>
    <li>Feature scaling using StandardScaler</li>
</ul>

<h2>📂 Project Structure</h2>
<pre>
Credit_Card_Customer_Prediction/
│
├── app.py
├── Model.pkl
├── standard_scalar.pkl
├── requirements.txt
├── creditcard.csv
├── templates/
│   └── index.html
└── README.html
</pre>

<h2>🛠 Technology Stack</h2>

<h3>Backend</h3>
<ul>
    <li>Python</li>
    <li>Flask</li>
    <li>Scikit-learn</li>
    <li>NumPy, Pandas</li>
    <li>Pickle</li>
</ul>

<h3>Frontend</h3>
<ul>
    <li>HTML</li>
    <li>CSS (optional styling)</li>
</ul>

<h3>Deployment</h3>
<ul>
    <li>GitHub</li>
    <li>Gunicorn (included in requirements)</li>
</ul>

<h2>⚙ Implementation Steps</h2>
<ol>
    <li>Collect and preprocess dataset</li>
    <li>Perform feature engineering and encoding</li>
    <li>Scale features using StandardScaler</li>
    <li>Train Logistic Regression model</li>
    <li>Save model and scaler using pickle</li>
    <li>Develop Flask web application</li>
    <li>Integrate model into web app for predictions</li>
</ol>

<h2>📐 Model Development</h2>
<p>
<b>Algorithm Used:</b> Logistic Regression
</p>

<ul>
    <li>Converted input features into numerical format</li>
    <li>Applied feature scaling for better model performance</li>
    <li>Used trained model for binary classification</li>
    <li>Output mapped to Good or Bad customer</li>
</ul>

<h2>🚀 Deployment Process</h2>
<ul>
    <li>Create GitHub repository</li>
    <li>Upload project files</li>
    <li>Install dependencies from requirements.txt</li>
    <li>Run Flask application locally or deploy using Gunicorn</li>
</ul>

<h2>📈 Results and Output</h2>
<ul>
    <li>Provides instant prediction</li>
    <li>Classifies customer as Good or Bad</li>
    <li>Lightweight and fast execution</li>
</ul>

<h2>🔮 Future Enhancements</h2>
<ul>
    <li>Add probability score for predictions</li>
    <li>Improve UI design</li>
    <li>Use advanced models like XGBoost</li>
    <li>Deploy on cloud platforms</li>
</ul>

<h2>✅ Conclusion</h2>
<p>
This project demonstrates an end-to-end machine learning workflow,
from data preprocessing to deployment. It is a practical example of integrating ML models into real-world applications.
</p>

<footer>
<p>
<b>Project By:</b> Sachin Muskudi<br>
<b>Last Updated:</b> April 2026<br>
<b>GitHub:</b> https://github.com/SachinMuskudi/Credit_Card_Customer_Prediction
</p>
</footer>
