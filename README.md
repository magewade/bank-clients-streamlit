
# **Bank Clients Prediction: Predicting Customer Acceptance**

This project is part of my Machine Learning course. In this project, I worked on a classification problem where the goal was to predict whether a bank client would agree to a banking offer based on various features. I used linear models, specifically Ridge Regression, to build and evaluate the model.

Course details: [Machine Learning Course on Stepik](https://stepik.org/course/177215/syllabus)

Project language: Russian

---

## **Overview**

This project involves building a predictive model using the bank clients dataset. The objective is to predict if a client will accept a banking offer based on a range of features, such as their demographic information, account details, and past behavior.

The Ridge Regression model was chosen due to its ability to handle multicollinearity, prevent overfitting, and provide regularization, making it ideal for this classification problem. Moreover, Ridge Regression is a simple and fast model, which makes it efficient for real-time predictions without requiring significant computational resources.

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

The Streamlit service is hosted online at** **[https://bank-clients-answers.streamlit.app](https://bank-clients-answers.streamlit.app/).

To run the app locally, follow these steps:

<pre class="!overflow-visible" data-start="1927" data-end="2182"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none rounded-t-[5px]">bash</div><div class="sticky top-9"><div class="absolute bottom-0 right-0 flex h-9 items-center pr-2"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none px-4 py-1" aria-label="Copy"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy</button></span><span class="" data-state="closed"><button class="flex select-none items-center gap-1 px-4 py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path d="M2.5 5.5C4.3 5.2 5.2 4 5.5 2.5C5.8 4 6.7 5.2 8.5 5.5C6.7 5.8 5.8 7 5.5 8.5C5.2 7 4.3 5.8 2.5 5.5Z" fill="currentColor" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"></path><path d="M5.66282 16.5231L5.18413 19.3952C5.12203 19.7678 5.09098 19.9541 5.14876 20.0888C5.19933 20.2067 5.29328 20.3007 5.41118 20.3512C5.54589 20.409 5.73218 20.378 6.10476 20.3159L8.97693 19.8372C9.72813 19.712 10.1037 19.6494 10.4542 19.521C10.7652 19.407 11.0608 19.2549 11.3343 19.068C11.6425 18.8575 11.9118 18.5882 12.4503 18.0497L20 10.5C21.3807 9.11929 21.3807 6.88071 20 5.5C18.6193 4.11929 16.3807 4.11929 15 5.5L7.45026 13.0497C6.91175 13.5882 6.6425 13.8575 6.43197 14.1657C6.24513 14.4392 6.09299 14.7348 5.97903 15.0458C5.85062 15.3963 5.78802 15.7719 5.66282 16.5231Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M14.5 7L18.5 11" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>Edit</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre language-bash"><span><span>$ git </span><span>clone</span><span> https://github.com/yourusername/bank-clients-streamlit.git
$ </span><span>cd</span><span> bank-clients-streamlit
$ python -m venv venv
$ </span><span>source</span><span> venv/bin/activate   </span><span># For Windows: venv\Scripts\activate</span><span>
$ pip install -r requirements.txt
$ streamlit run app.py
</span></span></code></div></div></pre>

After running the command, the app will be accessible at** **`http://localhost:8501`.

---

## **Model Overview:**

The model used to predict whether a client will accept a banking offer is based on** ** **Ridge Regression** , a linear model that adds an L2 regularization term to the linear regression cost function. This approach helps reduce overfitting and provides more stable predictions, especially when features are highly correlated.

### **Steps Involved:**

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

* **Python** : Core programming language used in this project.
* **Streamlit** : Framework for building the web application.
* **Scikit-learn** : For model training, evaluation, and metrics.
* **Pandas** : Data manipulation and cleaning.
* **Matplotlib/Seaborn** : Visualization libraries used for EDA.
* **NumPy** : For numerical operations and data manipulation.
