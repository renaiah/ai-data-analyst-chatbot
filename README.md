# Chat Bot – Installation & User Guide

## 1. Overview

This Chat Bot application is designed to answer Analytical, Descriptive, and Visualization-based questions using your Sales Data.
It integrates Python, Streamlit UI, and Ollama LLM models for enhanced accuracy.

---

## Requirements

### Software & Libraries

| Component          | Purpose                              |
| ------------------ | ------------------------------------ |
| Python 3.10 – 3.12 | Required runtime environment         |
| Ollama             | To run LLM models locally            |
| Streamlit          | Web UI for the chat bot              |
| Pandas             | CSV reading & processing             |
| psutil             | CPU & Memory usage monitoring        |
| matplotlib         | Graph visualizations                 |
| langchain-ollama   | Connect Ollama models with LangChain |
| tabulate           | Table formatting                     |
| mysql              | For DB operations                    |

---

## Folder Structure (AI Data Analyst Project)

```
AI-Data-Analyst-Chatbot
│
├── datasets               → Input CSV files
│
├── store
│     ├── chat history files (pickle)
│     ├── merged_data.csv → Complete sales dataset
│
├── app.py                 → Main ChatBot application
│
└── uploadDataset.py       → Script to upload new sales data
```

---

## How to Run the Chat Bot

NOTE: If already setup is done in server, follow only: Step 1, Step 3, Step 4 (Activate venv), Step 6 (if needed), Step 7.

---

### Step 1 — Install & Setup Ollama + LLM Model

1. Open terminal and check:

```
ollama
```

2. If not installed, visit: https://ollama.com
3. Select your OS and install

For Linux:

```
curl -fsSL https://ollama.com/install.sh | sh
```

4. Verify:

```
ollama
```

5. Install model:

```
ollama pull gpt-oss:20b
```

6. Check models:

```
ollama list
```

7. Test model:

```
ollama run gpt-oss:20b
```

Exit:

```
Ctrl + D or /bye
```

---

### Step 2 — Navigate to Project

```
cd AI-Data-Analyst-Chatbot
ls
```

---

### Step 3 — Set Up Python Virtual Environment

Check Python version:

```
python --version
```

Create virtual environment:

```
python3.12 -m venv venv
```

Activate:

```
source venv/bin/activate
```

---

### Step 4 — Install Required Packages

```
pip install streamlit pandas psutil matplotlib langchain-ollama tabulate
```

Check installed packages:

```
pip list
```

---

### Step 5 — Upload Sales CSV Dataset

```
python3 uploadDataset.py --csv datasets/Day_wise_20250401_20251110.csv --out store
```

Example output:

```
Loading new CSV data...
New rows loaded: 2436690
Cleaned missing values
Created merged dataset
Saved to: store/merged_data.csv
Done!
```

---

### Step 6 — Run the ChatBot UI

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## You’re All Set!

You now have:

* Installed Ollama
* Set up LLM model
* Uploaded sales dataset
* Run the ChatBot

Chat history is available on the left sidebar.
