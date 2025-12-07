# Big Data Analytics

**Course Materials for PSAU**  
**Instructor:** Dr. Chebil Khalil  
**Department:** Computer Science  
**Academic Year:** 2025

```{figure} assets/bigdata-banner.png
:name: fig-banner
:align: center
:width: 100%

Big Data Analytics Course
```

## Welcome

Welcome to **Big Data Analytics**, a comprehensive course that combines theoretical foundations with hands-on practice using modern Python-based tools and technologies.

```{admonition} Course Overview
:class: tip
This course covers the complete data analytics lifecycle, from data collection and preparation to advanced machine learning techniques and distributed computing with Apache Spark.
```

## Course Structure

The course is organized into three main parts:

### Part I: Foundations
- Introduction to Big Data
- Data Analytics Lifecycle
- Statistical Foundations with Python

### Part II: Machine Learning Methods
- Clustering Analysis
- Association Rules Mining
- Regression Analysis
- Classification Methods
- Time Series Analysis
- Text Analytics

### Part III: Big Data Technologies
- Distributed Computing with Hadoop & Spark
- Advanced Topics (Deep Learning, Cloud Platforms)

## Learning Approach

```{mermaid}
graph LR
    A[Theory] --> B[Interactive Notebooks]
    B --> C[Hands-On Labs]
    C --> D[Real Projects]
    D --> E[Industry Skills]
```

Each chapter includes:
- 📖 **Theory:** Core concepts and algorithms
- 💻 **Code Examples:** Python implementations
- 🧪 **Labs:** Practical exercises with datasets
- 📊 **Projects:** Real-world case studies

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/chebil/BigData.git
cd BigData

# Start the environment
docker-compose up -d

# Access Jupyter Lab
# Open http://localhost:8888 in your browser
```

### Option 2: Local Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bigdata-course

# Or use pip
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

### Option 3: Google Colab

Click the 🚀 **Colab** button at the top of any notebook to run it in Google Colab.

## Technologies Used

```{list-table}
:header-rows: 1
:name: tech-stack

* - Category
  - Technologies
* - Programming
  - Python 3.10+, Jupyter
* - Data Processing
  - NumPy, Pandas, Dask
* - Machine Learning
  - scikit-learn, XGBoost, TensorFlow
* - Visualization
  - Matplotlib, Seaborn, Plotly
* - Big Data
  - Apache Spark, PySpark
* - NLP
  - NLTK, spaCy, Gensim
* - Deployment
  - Docker, Flask, FastAPI
```

## Course Resources

- 📚 [Syllabus](00-syllabus/index.md)
- 📅 [Schedule](00-syllabus/schedule.md)
- 🎯 [Learning Objectives](00-syllabus/prerequisites.md)
- 🔗 [Additional Resources](00-syllabus/resources.md)

## Lab Materials

All labs are designed as Jupyter notebooks with:
- Clear learning objectives
- Step-by-step instructions
- Checkpoints and exercises
- Complete solutions
- Real-world datasets

[View All Labs →](labs/README.md)

## Prerequisites

```{admonition} Before You Begin
:class: warning
- Basic programming knowledge (Python preferred)
- Understanding of basic statistics
- Familiarity with linear algebra
- Command line/terminal basics
```

## Assessment

| Component | Weight |
|-----------|--------|
| Labs (10 × 5%) | 50% |
| Midterm Exam | 20% |
| Capstone Project | 25% |
| Participation | 5% |

## Getting Help

- 💬 **Discussion Forum:** [GitHub Discussions](https://github.com/chebil/BigData/discussions)
- 🐛 **Issues:** [Report bugs or suggest improvements](https://github.com/chebil/BigData/issues)
- 📧 **Email:** chebilkhalil@gmail.com
- 🕐 **Office Hours:** Sunday & Tuesday, 2:00 PM - 4:00 PM

## Acknowledgments

This course is based on:
- **Textbook:** "Data Science and Big Data Analytics" by EMC Education Services
- **Modern adaptations:** Python ecosystem, Jupyter notebooks, cloud platforms

## License

```{admonition} Educational Use
:class: note
These materials are provided for educational purposes at PSAU. All code examples are open source (MIT License). Datasets are attributed to their original sources.
```

---

```{tableofcontents}
```