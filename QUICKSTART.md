# Quick Start Guide

Get started with the Big Data Analytics course in 15 minutes!

## 🚀 Installation

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/chebil/BigData.git
cd BigData

# Build and run
docker-compose up

# Access Jupyter at http://localhost:8888
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/chebil/BigData.git
cd BigData

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Option 3: Conda

```bash
# Clone repository
git clone https://github.com/chebil/BigData.git
cd BigData

# Create environment
conda env create -f environment.yml
conda activate bigdata

# Launch Jupyter
jupyter notebook
```

---

## 📚 Course Structure

### Week-by-Week Guide

**Week 1-2: Foundations**
- Chapter 1: Introduction to Big Data
- Chapter 2: Statistical Foundations
- Lab 2: Statistics & Hypothesis Testing

**Week 3-4: Unsupervised Learning**
- Chapter 3: Clustering
- Lab 3: Customer Segmentation

**Week 5-6: Supervised Learning - Classification**
- Chapter 4: Classification
- Lab 4: Fraud Detection

**Week 7-8: Supervised Learning - Regression**
- Chapter 5: Regression
- Lab 5: House Price Prediction

**Week 9-10: Pattern Mining**
- Chapter 6: Association Rules
- Lab 6: Market Basket Analysis

**Week 11: Model Optimization**
- Chapter 7: Model Selection & Evaluation
- Lab 7: Hyperparameter Tuning

**Week 12: Time Series**
- Chapter 8: Time Series Analysis
- Lab 8: Sales Forecasting

**Week 13: Text Analytics**
- Chapter 9: Text Analytics & NLP
- Lab 9: Sentiment Analysis

**Week 14: Big Data Infrastructure**
- Chapter 10: Distributed Computing

**Week 15-16: Capstone**
- Final Project Implementation
- Presentation & Deployment

---

## 🎯 First Steps

### 1. Verify Installation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

print("✓ All packages installed successfully!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

### 2. Run Your First Analysis

```python
# Load sample data
from sklearn.datasets import load_iris
iris = load_iris()

# Create DataFrame
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Basic statistics
print(df.describe())

# Visualization
import matplotlib.pyplot as plt
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### 3. Start First Lab

```bash
# Navigate to labs
cd labs/02-statistics-lab

# Open lab notebook
jupyter notebook lab_exercises.md
```

---

## 📖 Learning Path

### For Beginners
1. Start with Chapter 1 (Introduction)
2. Complete Chapter 2 (Statistics)
3. Do Lab 2 thoroughly
4. Move to Clustering
5. Follow week-by-week guide

### For Intermediate Users
1. Review Chapters 1-2 quickly
2. Start with Chapter 3 (Clustering)
3. Complete labs in order
4. Focus on capstone project

### For Advanced Users
1. Skip to specific topics of interest
2. Focus on labs and capstone
3. Explore bonus challenges
4. Implement production deployments

---

## 🔧 Common Issues

### Package Installation Errors

```bash
# Update pip
pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Install specific package
pip install scikit-learn==1.3.0
```

### Jupyter Not Starting

```bash
# Install Jupyter explicitly
pip install jupyter notebook

# Try alternative port
jupyter notebook --port 8889
```

### Import Errors

```python
# Verify installation location
import sys
print(sys.executable)
print(sys.path)

# Install in current environment
!pip install package-name
```

---

## 📊 Sample Datasets

All datasets are available in the `data/` directory:

```bash
data/
├── raw/              # Original datasets
├── processed/        # Cleaned datasets
└── README.md         # Dataset documentation
```

### Download Datasets

```python
# Built-in datasets (no download needed)
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    fetch_california_housing, fetch_20newsgroups
)

# External datasets
import pandas as pd

# Wine Quality
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(url, sep=';')

# Mall Customers
url = 'https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv'
customers = pd.read_csv(url)
```

---

## 🎓 Getting Help

### Resources
- **Documentation**: Each chapter has detailed explanations
- **Examples**: 45+ real-world examples
- **Solutions**: Check `solutions/` directory
- **Office Hours**: [Schedule TBD]

### Discussion Forum
- Post questions with code examples
- Share insights and solutions
- Collaborate on projects

### Contact
- Instructor: [Email]
- TA Support: [Email]
- GitHub Issues: For technical problems

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] All packages installed
- [ ] Jupyter running successfully
- [ ] Sample code executed
- [ ] First dataset loaded
- [ ] Chapter 1 read
- [ ] Lab 2 started
- [ ] Joined discussion forum

---

## 🚀 Next Steps

1. **Complete Setup**: Verify all installations
2. **Read Syllabus**: Understand expectations
3. **Start Chapter 1**: Learn Big Data fundamentals
4. **Begin Lab 2**: Practice statistics
5. **Join Community**: Engage with peers

---

## 💡 Tips for Success

1. **Stay Consistent**: Follow weekly schedule
2. **Practice Daily**: Code every day
3. **Ask Questions**: Don't hesitate to ask
4. **Collaborate**: Learn from peers
5. **Build Portfolio**: Complete all labs
6. **Focus on Projects**: Showcase your work

---

## 🎯 Course Objectives

By the end of this course, you will:

✅ Master Big Data analytics fundamentals
✅ Implement 55+ machine learning algorithms
✅ Build production-ready ML pipelines
✅ Deploy models to cloud platforms
✅ Create comprehensive data visualizations
✅ Conduct statistical analysis
✅ Develop end-to-end data projects

---

## 📅 Important Dates

- **Week 1**: Course begins
- **Week 3**: Lab 2 due
- **Week 5**: Lab 3 due
- **Week 8**: Midterm exam
- **Week 13**: Labs 8-9 due
- **Week 15**: Capstone presentations
- **Week 16**: Final exam

---

**Ready to start?** Open `01-introduction/01-big-data-fundamentals.md` and begin your Big Data journey! 🚀
