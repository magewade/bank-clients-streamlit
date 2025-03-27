# **Bank Clients Prediction: Predicting Customer Acceptance**

Course details: [Machine Learning Course on Stepik](https://stepik.org/course/177215/syllabus)

Project language: Russian

This project is part of my Machine Learning course. In this project, I worked on a classification problem where the goal was to predict whether a bank client would agree to a banking offer based on various features. I used Logistic Regression to build and evaluate the model.

---

## **Files:**

* `bank_clients.ipynb`: A Colab notebook that contains the exploratory data analysis (EDA), modeling process, and results interpretation.
* `app.py`: Streamlit application file that hosts the web interface for users to interact with the model.
* `model.py`: Python script that processes the data, trains the model, and makes predictions.
* `requirements.txt`: Contains the necessary Python packages and dependencies.
* `/data` folder:
  * `clients.csv`: preprocessed clean dataset
  * `best_threshold.txt`: computed optimal threshold
  * `importances.csv`: file with feature importances
  * `model.pkl`: Pretrained model weights saved for efficient inference.
  * PNG and JPEG files of visualizations generated during the exploratory data analysis.

---

## **Service:**

The Streamlit service is hosted online at [https://bank-clients-answers.streamlit.app](https://bank-clients-answers.streamlit.app/).

To run the app locally, follow these steps:

```bash
$ git clone https://github.com/yourusername/bank-clients-streamlit.git
$ cd bank-clients-streamlit
$ python -m venv venv
$ source venv/bin/activate   # For Windows: venv\Scripts\activate
$ pip install -r requirements.txt
$ streamlit run app.py
```

---

## **Model Overview:**

This project involves building a predictive model using the bank clients dataset. The objective is to predict if a client will accept a banking offer based on a range of features, such as their demographic information, account details, and past behavior.

The **Logistic Regression** model was chosen due to its effectiveness in binary classification tasks. It models the probability that a given input point belongs to a certain class, making it well-suited for predicting whether a client will accept a banking offer. Logistic Regression is widely used for classification problems because of its simplicity, interpretability, and efficiency in terms of both computation and storage. Additionally, it helps prevent overfitting when combined with regularization, such as **L2 regularization** (which is often used to improve the model’s generalization ability).

**Steps Involved:**

1. **Exploratory Data Analysis (EDA):**
   The first step was to analyze the dataset visually and statistically to understand the relationships between features and target labels. Correlations, distributions, and outliers were explored.
2. **Data Preprocessing:**
   Data was cleaned and transformed, including handling missing values, encoding categorical variables, and scaling numerical features to prepare it for the model.
3. **Model Training:**
   Ridge Regression was selected for training the model. Hyperparameter tuning was performed to optimize the regularization parameter.
4. **Model Evaluation:**
   The model’s performance was evaluated using cross-validation, and various metrics such as accuracy, precision, recall, and ROC AUC were calculated to assess its ability to generalize to new data.

---

## **How to Use the App:**

1. **Input Data:**
   Users can input the client's details such as age, account status, and other features via the Streamlit interface.
2. **Model Inference:**
   After entering the details, the model will predict whether the client will accept the bank's offer (i.e., the prediction will be either "Yes" or "No").

---

## **Technologies Used:**

* ![Python](https://img.shields.io/badge/python-white?style=flat&logo=python)  Core programming language used in this project.
* ![Streamlit](https://img.shields.io/badge/streamlit-white?style=flat&logo=streamlit) Framework for building the web application.
* ![Scikit-learn](https://img.shields.io/badge/Scikit-white?style=flat&logo=Scikit-learn) For model training, evaluation, and metrics.
* ![Pandas](https://img.shields.io/badge/Pandas-white?style=flat&logo=pandas&logoColor=black) Data manipulation and cleaning.
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=flat&logo=matplotlib) Visualization libraries used for EDA.
* ![Seaborn](https://img.shields.io/badge/Seaborn-white?style=flat&logo=seaborn) Visualization libraries used for EDA.
* ![NumPy](https://img.shields.io/badge/NumPy-white?style=flat&logo=numpy&logoColor=black) For numerical operations and data manipulation.
