# Course Resources

## Primary Resources

### Textbook

#### Main Text
- **Title**: Data Science & Big Data Analytics
- **Author**: EMC Education Services
- **Publisher**: John Wiley & Sons, 2015
- **ISBN**: 978-1-118-87613-8
- **Access**: Available through university library or for purchase

**Why This Book?**
Industry-standard curriculum used by Fortune 500 companies, comprehensive coverage of the data analytics lifecycle, practical focus with real-world examples.

### Course Materials

All course materials are available in the [BigData GitHub repository](https://github.com/chebil/BigData):

- 📄 Lecture notes and slides (Markdown format)
- 📓 Jupyter notebooks for all labs (40+ notebooks)
- 📊 Datasets for exercises and projects
- 💻 Code templates and examples
- 📚 Supplementary readings and papers
- 🎥 Video tutorials (selected topics)

---

## Software & Tools

### Development Environment

#### Docker Environment (Provided)
✅ **Recommended approach**
- Pre-configured with all required libraries
- Consistent across all platforms (Windows, Mac, Linux)
- Includes Jupyter Lab, Python 3.11, and Apache Spark
- Quick setup with docker-compose

**Getting Started**:
```bash
git clone https://github.com/chebil/BigData.git
cd BigData
docker-compose up -d
# Access at http://localhost:8888
```

#### Alternative: Local Installation

If you prefer not to use Docker:
- **Python 3.11+** from [python.org](https://www.python.org/)
- **Jupyter Lab**: `pip install jupyterlab`
- **All packages**: `pip install -r requirements.txt`

---

### Recommended IDEs

#### Visual Studio Code (Recommended)
- **Features**: Excellent Python support, integrated terminal, Git integration
- **Extensions**: Python, Jupyter, Docker, GitLens
- **Download**: [code.visualstudio.com](https://code.visualstudio.com/)

#### PyCharm Community
- **Features**: Full-featured Python IDE, debugging, refactoring
- **Best for**: Large projects, advanced development
- **Download**: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)

#### Jupyter Lab
- **Features**: Interactive notebook environment, visualization
- **Best for**: Exploratory analysis, prototyping
- **Included**: In Docker environment

---

### Version Control

- **Git**: Distributed version control system - [git-scm.com](https://git-scm.com/)
- **GitHub**: Repository hosting and collaboration
- **GitHub Desktop**: GUI for Git (optional) - [desktop.github.com](https://desktop.github.com/)

---

## Online Resources

### Python Programming

#### Official Documentation
- [Python Documentation](https://docs.python.org/3/) - Complete language reference
- [Python Package Index (PyPI)](https://pypi.org/) - Find and install packages

#### Learning Platforms
- [Real Python](https://realpython.com/) - High-quality tutorials
- [Python.org Tutorial](https://docs.python.org/3/tutorial/) - Official tutorial
- [W3Schools Python](https://www.w3schools.com/python/) - Quick reference

#### Community
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python) - Q&A
- [r/learnpython](https://www.reddit.com/r/learnpython/) - Reddit community
- [Python Discord](https://pythondiscord.com/) - Real-time help

---

### Data Science Libraries

#### Pandas
- [Official Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Pandas Cookbook](https://github.com/jvns/pandas-cookbook)

#### NumPy
- [Official Documentation](https://numpy.org/doc/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

#### Scikit-learn
- [Official Documentation](https://scikit-learn.org/stable/)
- [Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Examples Gallery](https://scikit-learn.org/stable/auto_examples/)

#### Visualization
- **Matplotlib**: [Gallery](https://matplotlib.org/stable/gallery/), [Tutorials](https://matplotlib.org/stable/tutorials/)
- **Seaborn**: [Tutorial](https://seaborn.pydata.org/tutorial.html), [Gallery](https://seaborn.pydata.org/examples/index.html)
- **Plotly**: [Documentation](https://plotly.com/python/), [Dash](https://plotly.com/dash/)
- **Python Graph Gallery**: [python-graph-gallery.com](https://www.python-graph-gallery.com/)

---

### Big Data Tools

#### Apache Spark
- [Official Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Learning Spark (O'Reilly)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
- [Spark By Examples](https://sparkbyexamples.com/)

#### Hadoop Ecosystem
- [Apache Hadoop Documentation](https://hadoop.apache.org/docs/current/)
- [Hadoop: The Definitive Guide](https://www.oreilly.com/library/view/hadoop-the-definitive/9781491901687/)
- [Cloudera Tutorials](https://www.cloudera.com/tutorials.html)

---

### Statistics & Machine Learning

#### Video Courses
- [StatQuest](https://www.youtube.com/c/joshstarmer) - Best ML explanations on YouTube!
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) - Visual mathematics
- [Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning) - Andrew Ng's course
- [Fast.ai](https://www.fast.ai/) - Practical deep learning

#### Free Textbooks
- [Introduction to Statistical Learning](https://www.statlearning.com/) - Excellent ML book
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) - Advanced
- [Think Stats](https://greenteapress.com/thinkstats2/) - Python-based statistics
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.

#### Interactive Learning
- [Seeing Theory](https://seeing-theory.brown.edu/) - Visual statistics
- [Distill.pub](https://distill.pub/) - Visual explanations of ML concepts
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) - Google

---

## Datasets

### Included with Course

All datasets in the `datasets/` directory:
- Sample datasets for each lab
- Real-world datasets from various domains
- Synthetic data for specific exercises
- Pre-processed versions for quick start

### External Dataset Sources

#### General
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) - Classic ML datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets) - 50,000+ datasets
- [Google Dataset Search](https://datasetsearch.research.google.com/) - Search across web
- [data.gov](https://www.data.gov/) - US Government open data
- [Data.world](https://data.world/) - Collaborative data platform

#### Domain-Specific
- **Health**: [WHO Data](https://www.who.int/data), [CDC](https://data.cdc.gov/)
- **Finance**: [Quandl](https://www.quandl.com/), [Yahoo Finance](https://finance.yahoo.com/)
- **Science**: [NASA Open Data](https://data.nasa.gov/), [NOAA](https://www.ncdc.noaa.gov/)
- **Social**: [Reddit Datasets](https://www.reddit.com/r/datasets/), [Twitter API](https://developer.twitter.com/)
- **Government**: [World Bank](https://data.worldbank.org/), [UN Data](http://data.un.org/)

---

## Community & Support

### Course-Specific

- **GitHub Discussions**: Ask questions, share insights
- **Office Hours**: Weekly virtual sessions (schedule TBD)
- **Study Groups**: Connect with classmates
- **Course Wiki**: Collaborative knowledge base

### General Data Science Community

#### Forums & Discussion
- [r/datascience](https://www.reddit.com/r/datascience/) - Reddit community (350k+ members)
- [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/) - Learning-focused
- [Cross Validated](https://stats.stackexchange.com/) - Statistics Q&A
- [Data Science Stack Exchange](https://datascience.stackexchange.com/)

#### Blogs & Publications
- [Towards Data Science](https://towardsdatascience.com/) - Medium publication
- [KDnuggets](https://www.kdnuggets.com/) - News and tutorials
- [Analytics Vidhya](https://www.analyticsvidhya.com/) - Tutorials and competitions
- [Machine Learning Mastery](https://machinelearningmastery.com/) - Practical guides

#### Podcasts
- [Data Skeptic](https://dataskeptic.com/)
- [Linear Digressions](https://lineardigressions.com/)
- [Talking Machines](https://www.thetalkingmachines.com/)
- [Towards Data Science Podcast](https://towardsdatascience.com/podcast/home)

---

## Professional Organizations

- [Association for Computing Machinery (ACM)](https://www.acm.org/)
- [Institute of Electrical and Electronics Engineers (IEEE)](https://www.ieee.org/)
- [Data Science Association](https://www.datascienceassn.org/)
- [International Institute of Analytics](https://www.iianalytics.com/)

---

## Books & Publications

### Recommended Reading

#### Fundamentals
- **Python for Data Analysis** by Wes McKinney (creator of Pandas)
- **Hands-On Machine Learning** by Aurélien Géron
- **Data Science from Scratch** by Joel Grus
- **Think Python** by Allen Downey

#### Advanced Topics
- **Deep Learning** by Goodfellow, Bengio, and Courville
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **The Elements of Statistical Learning** by Hastie, Tibshirani, and Friedman
- **Probabilistic Graphical Models** by Daphne Koller

#### Big Data
- **Designing Data-Intensive Applications** by Martin Kleppmann
- **Big Data** by Viktor Mayer-Schönberger
- **Hadoop: The Definitive Guide** by Tom White
- **Spark: The Definitive Guide** by Bill Chambers and Matei Zaharia

#### Career & Industry
- **Data Science for Business** by Foster Provost and Tom Fawcett
- **Weapons of Math Destruction** by Cathy O'Neil
- **The Master Algorithm** by Pedro Domingos

---

### Academic Journals

- [Journal of Machine Learning Research (JMLR)](https://jmlr.org/)
- [Journal of Statistical Software](https://www.jstatsoft.org/)
- [IEEE Transactions on Knowledge and Data Engineering](https://www.computer.org/csdl/journal/tk)
- [ACM Transactions on Knowledge Discovery from Data](https://dl.acm.org/journal/tkdd)

---

## Tools & Utilities

### Data Visualization

- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [Bokeh](https://bokeh.org/) - Web-based interactive plots
- [Altair](https://altair-viz.github.io/) - Declarative visualization
- [D3.js](https://d3js.org/) - Advanced web visualizations (JavaScript)

### Development Tools

- [Black](https://black.readthedocs.io/) - Python code formatter
- [Flake8](https://flake8.pycqa.org/) - Code linter
- [Pytest](https://pytest.org/) - Testing framework
- [JupyterLab Extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html)

### Cloud Platforms

- [Google Colab](https://colab.research.google.com/) - Free Jupyter notebooks with GPU
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) - ML platform
- [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/)
- [Databricks](https://databricks.com/) - Unified analytics (built on Spark)

---

## Course Updates

This resource list is updated regularly. Check back for:
- ✨ New tutorials and guides
- 📚 Updated library versions
- 🌐 Additional dataset sources
- 🛠️ Emerging tools and technologies

**Last Updated**: December 2025

---

## Need Help Finding Resources?

If you're looking for something specific:
1. Check the course [README](../README.md)
2. Search the [GitHub discussions](https://github.com/chebil/BigData/discussions)
3. Post in the course forum
4. Email the instructor

Happy learning! 📚🚀
