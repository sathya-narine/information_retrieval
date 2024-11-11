# Information Retrieval

This project implements a simple information retrieval system using BM25 and TF-IDF algorithms, with a web-based user interface built using Streamlit.

## Setup Instructions

### Step 1: Create a Virtual Environment
To create a virtual environment, run the following command:

python -m venv ir_env

### Step 2: Activate the Virtual Environment
Activate the virtual environment using the appropriate command for your operating system:

Windows:
cd ir_env/Scripts
activate
Linux/MacOS:

sh
source ir_env/bin/activate

### Step 3: Install Dependencies
Install the required Python packages using pip:

pip install -r requirements.txt

Running the Application
### Step 4: Run the Streamlit App
To run the Streamlit application, use the following command:

streamlit run app.py

This will start the Streamlit server and open the application in your default web browser.

Evaluating the System
### Step 5: Evaluate the System for Metrics
To evaluate the system and compute metrics such as Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), Precision@k, Recall@k, and F1@k, run the following command:

python Search_Algorithm.py
