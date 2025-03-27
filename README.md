# Linear models and their presentation

This is a project for ML course: [link ](https://stepik.org/course/177215/syllabus)

## Files:

* `Bank_clients.ipynb`: Colab notebook with exploratory data analysis, modelling and interpretations
* `app.py`: Streamlit app file
* `model.py`: script that transforms data, trains and runs the model
* `requirements.txt`: package requirements files
* `/data` folder has:
  * `clients.csv`: copy of dataset
  * `model_weights.mw`: pretrained model
  * and png visualizations for EDA

## Service:

Streamlit service is available at [https://bank-clients-answers.streamlit.app](https://bank-clients-answers.streamlit.app)

To run locally, clone the repo and do the following:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
```

The app will be available at `http://localhost:8501`
