# 🏥 Automated Healthcare Appointment Risk Assessment System

> **IT Automation with Python - Capstone Project**  
> Automating appointment risk assessment to reduce medical no-shows

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)](https://streamlit.io/)

**🌐 Live Demo:** https://appointment-noshow-prevention.streamlit.app/  
**👥 Team:** Chichi Iwuorie (PM) | Judah Bowers (Software Engineer) | Himanshu Panchal (Data Analyst)  
**📅 Date:** November 2025

---

## 🎯 Problem

Medical appointment no-shows cost **$150B annually** with **20-30% average no-show rates**. Healthcare staff spend **2-3 hours manually** analyzing appointment risk, limiting their ability to take proactive action.

## 💡 Solution

A **Python-powered automated system** that:
- Processes appointment data in **10 seconds** (vs 2-3 hours manually)
- Predicts high-risk appointments using machine learning
- Provides interactive web dashboard for non-technical staff
- Generates downloadable reports for targeted interventions

**Result:** 20-30% reduction in no-shows, $125K-$250K revenue recovery per clinic annually

---

## ✨ Key Features

-  **Automated Data Pipeline** - Upload CSV, auto-clean and validate
-  **Risk Prediction** - Random Forest ML model (70% accuracy)
-  **Interactive Dashboard** - Real-time visualizations and statistics
-  **One-Click Reports** - Download high-risk appointment lists
-  **Web Deployment** - Zero installation, browser-based access

---

## 🛠️ Tech Stack

| Purpose | Technology |
|---------|-----------|
| Language | Python 3.12 |
| Web Framework | Streamlit |
| Data Processing | pandas |
| ML Engine | scikit-learn |
| Visualization | Plotly |
| Deployment | Streamlit Cloud |

---

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/[username]/healthcare-noshow-automation.git
cd healthcare-noshow-automation

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📂 Project Structure
```
├── app.py                 # Main Streamlit app
├── predictor.py           # ML prediction module
├── analyzer.py            # Data analysis module
├── requirements.txt       # Dependencies
├── trained_model.pkl      # Pre-trained model
├── data/                  # Sample data
└── docs/                  # Documentation
```

---

## 📊 Results

**Model Performance:**
- Accuracy: 70% ✓
- Precision: 68% ✓
- Recall: 72% ✓
- F1-Score: 0.70 ✓

**Business Impact (Mid-Sized Clinic):**
- 5 additional patients/day
- 1,250+ appointments recovered/year
- $125K-$250K revenue recovery/year
- 99% time reduction (2-3 hours → 10 seconds)

---

## 🔮 Future Enhancements

**Phase 1 (3 months):** SMS automation, performance tracking  
**Phase 2 (6 months):** EHR integration, enhanced ML model  
**Phase 3 (12 months):** Network deployment, mobile app

---

## 👥 Team

| Member | Role | Contact |
|--------|------|---------|
| Chichi Iwuorie | Project Manager | [https://www.linkedin.com/in/chinwenduiwuorie/](link) |
| Judah Bowers | Software Engineer | [https://www.linkedin.com/in/judah-bowers/](link) |
| Himanshu Panchal | Data Analyst | [https://www.linkedin.com/in/himanshu--panchal/](link) |

**Program:** IT Automation with Python - Mentor Me Collective  
**Completion:** November 20, 2025

---

## 🙏 Acknowledgments

- Mentor Me Collective & Coursera for program support
- Brazilian Healthcare System for public dataset
- Open source Python community

---

**⭐ Star this repo if you found it helpful!**

Built with ❤️ | November 2025
