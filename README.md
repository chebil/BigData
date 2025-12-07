# Big Data Analytics Course

[![Jupyter Book](https://img.shields.io/badge/Jupyter-Book-orange?logo=jupyter)](https://chebil.github.io/BigData/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](Dockerfile)

**Course Materials for Big Data Analytics - PSAU**

## 📚 About

Comprehensive course materials for teaching Big Data Analytics using modern Python-based tools and technologies. This repository contains:

- 📖 **11 Chapters** covering foundations to advanced topics
- 💻 **Interactive Jupyter Notebooks** for hands-on learning
- 🧪 **11 Practical Labs** with real-world datasets
- 🐳 **Docker Environment** for easy setup
- 📊 **Visualization Examples** using Matplotlib, Seaborn, and Plotly
- ⚡ **Big Data Processing** with Apache Spark

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/chebil/BigData.git
cd BigData

# Start all services (Jupyter, Spark, PostgreSQL)
docker-compose up -d

# Access Jupyter Lab at http://localhost:8888
# Access Spark UI at http://localhost:8080
```

### Local Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bigdata-course

# Or use pip
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

### Build the Book

```bash
# Install Jupyter Book
pip install jupyter-book

# Build the book
jupyter-book build .

# Open _build/html/index.html in your browser
```

## 📖 Course Structure

### Part I: Foundations
1. **Introduction to Big Data** - Concepts, lifecycle, data types
2. **Data Analytics Lifecycle** - Six-phase approach
3. **Statistical Foundations** - Python, NumPy, Pandas, visualization

### Part II: Machine Learning
4. **Clustering** - K-means, hierarchical, DBSCAN
5. **Association Rules** - Market basket analysis, Apriori
6. **Regression** - Linear, multiple, regularization
7. **Classification** - Logistic regression, Naïve Bayes, decision trees
8. **Time Series** - ARIMA, forecasting, Prophet
9. **Text Analytics** - NLP, sentiment analysis, topic modeling

### Part III: Big Data Technologies
10. **Distributed Computing** - Hadoop, Spark, PySpark
11. **Advanced Topics** - Deep learning, deployment, cloud platforms

## 🧪 Labs

| Lab | Topic | Duration |
|-----|-------|----------|
| Lab 0 | Environment Setup | 30 min |
| Lab 1 | Data Exploration | 2 hours |
| Lab 2 | Python & Pandas | 2 hours |
| Lab 3 | Statistics & Visualization | 3 hours |
| Lab 4 | Clustering | 2.5 hours |
| Lab 5 | Association Rules | 2 hours |
| Lab 6 | Regression | 2.5 hours |
| Lab 7 | Classification | 3 hours |
| Lab 8 | Time Series | 2.5 hours |
| Lab 9 | Text Analytics | 3 hours |
| Lab 10 | Apache Spark | 3 hours |
| Lab 11 | Capstone Project | 10+ hours |

## 🛠️ Technologies

**Core Stack:**
- Python 3.10+
- Jupyter Lab
- NumPy, Pandas, SciPy
- scikit-learn
- Matplotlib, Seaborn, Plotly

**Big Data:**
- Apache Spark 3.4
- PySpark
- Dask

**Machine Learning:**
- XGBoost, LightGBM
- TensorFlow, Keras, PyTorch
- Prophet, statsmodels

**NLP:**
- NLTK, spaCy, Gensim
- Transformers

**Infrastructure:**
- Docker & Docker Compose
- PostgreSQL
- Git

## 📊 Datasets

All labs use real-world datasets:
- US Census 2020 data
- Retail transactions
- Customer segmentation data
- Time series (stocks, weather)
- Text corpora (reviews, social media)
- Image datasets

## 🎓 Learning Outcomes

After completing this course, students will be able to:

✅ Apply the data analytics lifecycle to real-world problems  
✅ Perform exploratory data analysis using Python  
✅ Implement machine learning algorithms from scratch  
✅ Build and evaluate classification and regression models  
✅ Process large datasets using Apache Spark  
✅ Perform text analytics and sentiment analysis  
✅ Deploy machine learning models  
✅ Work with big data technologies  

## 📝 Assessment

- **Labs:** 50% (10 labs × 5% each)
- **Midterm:** 20%
- **Capstone Project:** 25%
- **Participation:** 5%

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👨‍🏫 Instructor

**Dr. Chebil Khalil**  
Department of Computer Science  
Prince Sattam bin Abdulaziz University (PSAU)  
Email: chebilkhalil@gmail.com

## 🔗 Links

- 📚 [Course Website](https://chebil.github.io/BigData/)
- 💬 [Discussions](https://github.com/chebil/BigData/discussions)
- 🐛 [Issues](https://github.com/chebil/BigData/issues)
- 📖 [Documentation](https://chebil.github.io/BigData/)

## ⭐ Star History

If you find this repository helpful, please consider giving it a star!

---

**Built with** ❤️ **using Jupyter Book and MyST Markdown**