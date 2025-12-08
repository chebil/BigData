# Phase 1: Discovery

## Overview

The Discovery phase is where we define the problem, understand the context, and plan the project approach. This critical first step sets the foundation for the entire analytics project.

## Learning Objectives

- Frame business problems as analytics questions
- Identify key stakeholders and requirements
- Assess data availability and quality
- Define success criteria
- Create project charter

## 1. Problem Formulation

### Business Understanding

**Key Questions**:
1. What business problem are we solving?
2. Who are the stakeholders?
3. What are the expected outcomes?
4. What decisions will be influenced?
5. What is the timeline and budget?

### Example: Retail Churn Prediction

**Business Problem**: 
"We're losing 20% of customers annually, costing $2M in revenue."

**Analytics Question**:
"Can we predict which customers are likely to churn in the next 90 days, and what factors drive churn?"

**Success Criteria**:
- Predict churn with 80%+ accuracy
- Identify top 3 churn drivers
- Enable targeted retention campaigns
- Reduce churn by 5% in 6 months

## 2. Stakeholder Analysis

### Key Stakeholders

```python
import pandas as pd

stakeholders = pd.DataFrame({
    'Role': ['Executive Sponsor', 'Business Owner', 'Data Team', 'IT', 'End Users'],
    'Interest': ['ROI, Strategic', 'Operations', 'Technical', 'Infrastructure', 'Usability'],
    'Influence': ['High', 'High', 'Medium', 'Medium', 'Low'],
    'Involvement': ['Monthly reviews', 'Weekly', 'Daily', 'As needed', 'Training']
})

print(stakeholders)
```

### RACI Matrix

| Activity | Business Owner | Data Scientist | Data Engineer | Analyst |
|----------|---------------|----------------|---------------|----------|
| Define Problem | **R, A** | C | I | I |
| Data Collection | I | C | **R, A** | C |
| Model Building | C | **R, A** | I | C |
| Deployment | A | C | **R** | C |
| Monitoring | A | I | **R** | C |

*R=Responsible, A=Accountable, C=Consulted, I=Informed*

## 3. Resource Assessment

### Data Resources

**Assessment Checklist**:
- [ ] What data is available?
- [ ] Where is it stored?
- [ ] How recent is it?
- [ ] What is the data quality?
- [ ] Are there privacy/compliance issues?
- [ ] What external data might help?

### Technical Resources

**Infrastructure Assessment**:
```python
resources = {
    'Compute': {
        'Current': '16 CPU cores, 64GB RAM',
        'Required': 'Cloud with 32+ cores for distributed processing',
        'Gap': 'Need cloud migration'
    },
    'Storage': {
        'Current': '5TB on-premises database',
        'Required': '20TB for historical analysis',
        'Gap': 'Need data lake'
    },
    'Tools': {
        'Current': 'Excel, SQL',
        'Required': 'Python, Spark, ML frameworks',
        'Gap': 'Team training needed'
    }
}
```

### Team Resources

**Skills Assessment**:
- Data Scientists: 2 (need 3)
- Data Engineers: 1 (need 2)
- Analysts: 3 (sufficient)
- Domain Experts: Available for consultation

## 4. Initial Hypothesis

### Formulating Hypotheses

**Template**:
"We believe that [X factor] influences [Y outcome] because [reasoning]."

**Examples**:

1. **Churn Hypothesis**:
   - H1: Customers with declining usage are more likely to churn
   - H2: Price-sensitive customers churn more during price increases
   - H3: Poor customer service correlates with churn

2. **Testing Framework**:
```python
hypotheses = [
    {
        'hypothesis': 'Usage decline predicts churn',
        'metric': 'Monthly active sessions',
        'threshold': '>30% decline',
        'test_method': 'Logistic regression'
    },
    {
        'hypothesis': 'Support tickets increase churn',
        'metric': 'Number of tickets in last 90 days',
        'threshold': '>3 tickets',
        'test_method': 'Survival analysis'
    }
]
```

## 5. Project Charter

### Charter Components

**1. Executive Summary**
- Problem statement
- Proposed solution
- Expected impact

**2. Objectives**
- Primary: Reduce customer churn by 5%
- Secondary: Identify churn drivers
- Tertiary: Build predictive model

**3. Scope**

**In Scope**:
- Customer data from past 2 years
- Transaction history
- Customer service interactions
- Predictive churn model

**Out of Scope**:
- Real-time prediction (Phase 2)
- External market data
- Competitive intelligence

**4. Timeline**
```
Week 1-2:  Discovery & Data Assessment
Week 3-6:  Data Preparation
Week 7-10: Model Development
Week 11-12: Testing & Validation
Week 13-14: Deployment
Week 15-16: Monitoring & Handoff
```

**5. Success Metrics**
- Model accuracy: >80%
- Precision for top 20% at-risk: >70%
- Business impact: 5% churn reduction
- Deployment: Production-ready model

## 6. Risk Assessment

### Potential Risks

```python
risks = pd.DataFrame({
    'Risk': [
        'Data quality issues',
        'Insufficient historical data',
        'Stakeholder alignment',
        'Resource constraints',
        'Privacy/compliance'
    ],
    'Probability': ['High', 'Medium', 'Medium', 'High', 'Low'],
    'Impact': ['High', 'High', 'Medium', 'Medium', 'High'],
    'Mitigation': [
        'Data quality assessment upfront',
        'Use synthetic data if needed',
        'Regular stakeholder meetings',
        'Prioritize critical features',
        'Legal review before project start'
    ]
})
```

## 7. Communication Plan

### Reporting Structure

**Weekly**: 
- Team standup (internal)
- Progress dashboard

**Bi-weekly**:
- Stakeholder update
- Demo of findings

**Monthly**:
- Executive briefing
- Milestone review

### Deliverables

1. **Discovery Phase**:
   - Project charter
   - Data assessment report
   - Initial hypotheses

2. **Throughout Project**:
   - Weekly progress reports
   - Technical documentation
   - Code repository

3. **Final Deliverables**:
   - Trained model
   - Deployment guide
   - Performance dashboard
   - Recommendations

## Discovery Phase Checklist

- [ ] Problem clearly defined
- [ ] Stakeholders identified and engaged
- [ ] Data availability assessed
- [ ] Resources inventoried
- [ ] Success criteria established
- [ ] Project charter approved
- [ ] Risks identified and mitigated
- [ ] Initial hypotheses formulated
- [ ] Timeline agreed upon
- [ ] Budget allocated

## Key Takeaways

:::{admonition} Summary
:class: note

1. **Clear problem definition** is critical for project success
2. **Stakeholder alignment** prevents scope creep
3. **Resource assessment** identifies gaps early
4. **Success criteria** must be measurable
5. **Risk management** starts in discovery
6. **Documentation** ensures shared understanding
7. **Communication plan** keeps everyone informed
:::

## Next Phase

Once discovery is complete, proceed to **Data Preparation** where we'll:
- Acquire and clean data
- Perform exploratory analysis
- Engineer features
- Prepare modeling datasets
