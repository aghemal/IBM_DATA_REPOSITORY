# 🔍 AI Data Quality Monitoring Dashboard

An **AI-powered Streamlit application** that automates data quality checks in datasets. It detects **missing values**, **duplicate records**, and performs **anomaly detection** using machine learning algorithms like **Isolation Forest**. It also provides interactive visualizations and allows downloading anomaly reports.

---

## 📌 Features

* ✅ Missing value analysis with visual heatmaps
* ✅ Duplicate record detection
* ✅ AI-based anomaly detection using Isolation Forest
* ✅ Interactive Streamlit dashboard with charts and metrics
* ✅ Downloadable reports and anomaly files
* ✅ Supports **CSV**, **Excel** files

---

## 📂 Project Structure

```
├── main.py               # Main Streamlit app file
├── requirements.txt     # Required Python packages
└── README.md            # This file
```

---

## ⚙️ Requirements

* Python 3.8 or later
* pip

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-quality-monitor.git
cd ai-data-quality-monitor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Launch the Streamlit app
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 How to Use

1. Upload a `.csv` or `.xlsx` file using the uploader.
2. Navigate through the tabs:

   * **Data Overview**: View data and type distributions.
   * **Missing Values**: Explore null values and heatmaps.
   * **Duplicates**: Detect duplicate records.
   * **Anomaly Detection**: Select numeric columns and detect outliers using AI.
3. Download results and data quality reports from the sidebar.



## 🧑‍💻 Author

**Hemal A.G**



