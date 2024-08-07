# IMDRF Codes App
The idea came from Vladimir. Since he works at a medical company, there are some situations where they receive a complaint about a specific medical device. Someone must analyze that complaint and assign an IMDRF code. You can see the full list [Here](https://www.imdrf.org/documents/terminologies-categorized-adverse-event-reporting-aer-terms-terminology-and-codes).  
The project is a prototype that takes a string (complaint) and assigns the respective IMDRF Code.  

## Table of Contents
- [Research](#research)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)

## Research

**Introduction:**
In the healthcare sector, accurate documentation and communication are critical for effective patient care and regulatory compliance. 
Messages from doctors often contain a wealth of information that needs to be accurately categorized according to the International Medical Device Regulators Forum (IMDRF) codes. 
These codes are essential for classifying and standardizing medical device-related information. 
However, manually identifying the correct IMDRF codes from doctors' messages is a time-consuming and error-prone process.

**Problem statement:**
**Context:** Accurate documentation in healthcare is critical for patient care and regulatory compliance.
**Problem:** Manually identifying IMDRF codes from doctors' messages is inefficient and error-prone due to the complexity of medical terminology.
**Impact:** This inefficiency leads to increased workload, potential misclassification, and compromised data quality.
**Solution**: To create an app that assists companies in identifying the appropriate IMDRF codes for messages sent by doctors.

**Research Questions:**
- How accurately can the AI model classify doctors' messages into the correct IMDRF codes?
- What are the most common errors made by the AI model in identifying IMDRF codes from doctors' messages?
- Which model is better suited for predicting IMDRF codes?

**Hypotheses:**
- The AI model will classify doctors' messages into the correct IMDRF codes with an accuracy rate of at least 90%.
- The most common errors made by the AI model will be related to ambiguous or highly specialized medical terminology.

## Requirements

- Python 3.6+
- pip (Python package installer)

## Installation - CURRENTLY BROKEN have to fix 

1. **Clone the repository:**

    ```bash
    git clone https://github.com/mhi1001/IMDRF-Code-Classificator
    cd IMDRF-Code-Classificator
    ```

2. **Create and activate a virtual environment:**

    On Windows:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    On macOS/Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```
4. **Possible dependencies problem??**
    During the installation in a different computer I had to install the following
    ```bash
    pip install transformers[torch]
    ```

## Running the Application

1. **Start the Flask application:**
    app.py executes the code first for the BERT model and then for LSTM and for Naive Bayes.  
    When running for the first time, or re-running, if some error happened with running the python for each model, make sure you delete all the model folders in the directory.

    **if app.py doesn't work because some model is crashing**
   Try to delete all the directories that the models created (it has their respective name), and then run each script 1 by 1.  
   **1st** run bert_model.py  
   **2nd** run lstm_model.py  
   **3rd** run naivebayes_model.py  
   Lastly run ```python app.py ``` for opening the webapp with the models.  


    ```bash
    python app.py
    ```

3. **Wait a bit...** The app.py will execute the three different Python scripts, each training a model and storing it in the repository directory.

4. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

## Dataset

We used the official IMDRF JSON file as a dataset, but if you verified it, the file has a single code and a single definition, which we believe caused our initial failure to train the models. So we grabbed the JSON, kept the code and definition, and created multiple definitions for each code.

```
{
      "code": "A0401",
      "term": "Break",
      "definition": ["The patient experienced issues due to damage to the materials used in the device construction", "The device malfunctioned because of breakage in its materials", "We received complaints about undesired damage to the device materials", "The device's materials broke unexpectedly, causing problems for the patient", "There was damage to the device materials, leading to patient discomfort", "The patient reported issues with the device due to material breakage", "The device failed because its construction materials were damaged", "We had problems with the device due to undesired damage to its materials", "The patient experienced adverse effects because the device materials broke", "The device malfunctioned due to damage in the construction materials", "The patient was affected by unexpected breakage of the device materials", "The device had to be replaced due to damage in its construction materials", "The patient reported discomfort due to breakage in the device materials", "The device stopped working because its materials were damaged", "There was a problem with the device due to undesired breakage of its materials", "The patient experienced issues because the device materials were damaged", "The device malfunctioned because of unexpected damage to its materials", "The patient reported problems with the device due to material breakage", "The device's materials broke, leading to patient discomfort", "There was undesired damage to the device materials, causing issues", "The patient experienced adverse effects due to breakage in the device materials", "The device failed because its materials were unexpectedly damaged", "We received complaints about the device due to breakage in its construction materials", "The patient was affected by damage to the materials used in the device", "The device malfunctioned due to undesired breakage of its materials"]
  },
  {
      "code": "A0501",
      "term": "Detachment of Device or Device Component",
      "definition": ["The patient experienced issues due to the device separating from its physical construct", "The device malfunctioned because it detached from its chassis", "We received complaints about the device separating from its integrity", "The device's chassis separated, causing problems for the patient", "There was separation in the device's construct, leading to patient discomfort", "The patient reported issues with the device due to its physical separation", "The device failed because it lost its integrity", "We had problems with the device due to separation from its chassis", "The patient experienced adverse effects because the device detached from its construct", "The device malfunctioned due to separation in its physical integrity", "The patient was affected by unexpected separation of the device from its chassis", "The device had to be replaced due to separation from its physical construct", "The patient reported discomfort due to the device losing its integrity", "The device stopped working because it separated from its chassis", "There was a problem with the device due to separation from its physical integrity", "The patient experienced issues because the device separated from its construct", "The device malfunctioned because of unexpected separation from its chassis", "The patient reported problems with the device due to its physical separation", "The device's integrity was compromised, leading to patient discomfort", "There was undesired separation of the device from its construct, causing issues", "The patient experienced adverse effects due to the device detaching from its chassis", "The device failed because it unexpectedly separated from its physical construct", "We received complaints about the device due to separation from its integrity", "The patient was affected by the device detaching from its physical construct", "The device malfunctioned due to undesired separation from its chassis"]
  },
```
Some statistics about the dataset we have used for training:
![code_distribution](https://github.com/user-attachments/assets/868f7dde-5841-45eb-a5e9-9376231e73d6)
![text_length_distribution](https://github.com/user-attachments/assets/a28da1c1-aa7e-41b3-bf4d-08433b83ed5f)
![word_count_distribution](https://github.com/user-attachments/assets/10ccd393-cd70-4b90-aa3b-33dae3abd1fd)

The codes seem evenly distributed, with most occurring around 25 times each.  
Text lengths show a normal distribution centred around 75-100 characters, with some outliers.  
Word counts vary significantly across codes, with median counts ranging from about 5 to 25 words per definition.


## Models

![metrics_comparison](https://github.com/user-attachments/assets/906c64da-2c5a-4795-b092-047645070156)


![error_analysis](https://github.com/user-attachments/assets/5daf37c9-981e-406d-9fda-0e227b6b59de)



