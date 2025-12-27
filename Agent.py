# ============================================================================
# 1. IMPORTS AND DEPENDENCIES
# ============================================================================
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CrewAI Imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import TavilySearchTool

# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    CSVSearchTool,
    PDFSearchTool,
    VisionTool,
    RagTool,
    ScrapeElementFromWebsiteTool,
    ScrapeWebsiteTool,
    XMLSearchTool,
    CodeInterpreterTool,
    LlamaIndexTool
)


# LangGraph Imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Document Processing
import pdfplumber
from bs4 import BeautifulSoup

# Environment Management
from dotenv import load_dotenv, find_dotenv

# Streamlit
import streamlit as st

# 🔥 DISABLE ALL CREWAI TRACING & PROMPTS
os.environ["CREWAI_DISABLE_TRACING"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_LOG_LEVEL"] = "ERROR"
os.environ["CREWAI_INTERACTIVE"] = "false"

# =====================================================
# 1. ENVIRONMENT SETUP
# =====================================================

# BLOCK OpenAI completely
os.environ["OPENAI_API_KEY"] = "DISABLED"

# ============================================================================
# 2. ENVIRONMENT CONFIGURATION (SECURE API KEY HANDLING)
# ============================================================================

filepath = r"C:\Desktop\code\Aentic workflow part-II\MyApi.env"
_ = load_dotenv(find_dotenv(filepath))

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# ============================================================================
# 3. LLM CONFIGURATION
# ============================================================================

openai_llm = LLM(
    model="openai/gpt-4o",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.7
)

# ============================================================================
# 6. CREWAI AGENT DEFINITIONS (5 SPECIALIZED AGENTS)
# ============================================================================

# Agent 1: File Intake Specialist
file_intake_agent = Agent(
    role="File Intake Specialist",
    goal="Parse and validate uploaded files (CSV, PDF, HTML) and extract structured data for analysis.",
    backstory="You are an expert in data ingestion pipelines with deep knowledge of multiple file formats. You ensure data integrity and proper schema extraction before any processing begins. You handle CSV, PDF, and HTML files with precision.",
    tools=[FileReadTool(), DirectoryReadTool(), CSVSearchTool(), PDFSearchTool()],
    verbose=True,
    llm=openai_llm,
    allow_delegation=False
)


# Agent 2: Data Preprocessing Engineer
data_preprocessing_agent = Agent(
    role="Data Preprocessing Engineer",
    goal="Clean, transform, and prepare data for statistical analysis by handling missing values and data type conversions.",
    backstory="You are a data quality specialist focused on transforming raw data into analysis-ready datasets. You identify and handle missing values, outliers, duplicates and ensure data consistency across all columns.",
    tools=[CSVSearchTool(), FileReadTool(), CodeInterpreterTool()],
    verbose=True,
    llm=openai_llm,
    allow_delegation=False
)

# Agent 3: Statistical Analyst
statistical_analyst_agent = Agent(
    role="Statistical Analyst",
    goal="Perform comprehensive descriptive statistics including mean, median, mode, quartiles, IQR, and correlation analysis.",
    backstory="You are a senior statistician with expertise in descriptive analytics. You compute accurate statistical measures and identify patterns, correlations, and data distributions to inform decision-making.",
    tools=[CodeInterpreterTool(), CSVSearchTool()],
    verbose=True,
    llm=openai_llm,
    allow_delegation=False
)

# Agent 4: Visualization Specialist
visualization_specialist_agent = Agent(
    role="Visualization Specialist",
    goal="Create insightful visualizations including line charts, bar charts, and scatter plots to represent data patterns.",
    backstory="You are a data visualization expert who transforms numbers into compelling visual stories. You select the most appropriate chart types and design clear, professional visualizations that highlight key insights.",
    tools=[CodeInterpreterTool()],
    verbose=True,
    llm=openai_llm,
    allow_delegation=False
)

# Agent 5: Insights & Reporting Analyst
reporting_analyst_agent = Agent(
    role="Insights & Reporting Analyst",
    goal="Synthesize all findings into actionable recommendations and generate comprehensive reports for stakeholders.",
    backstory="You are a business intelligence analyst who excels at translating technical findings into strategic recommendations. You create executive summaries, identify trends, and provide data-driven recommendations based on statistical evidence.",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    verbose=True,
    llm=openai_llm,
    allow_delegation=False
)

# ============================================================================
# 7. CREWAI TASK DEFINITIONS (ONE TASK PER AGENT)
# ============================================================================
file_intake_task = Task(
    description="Parse the uploaded file and extract structured data. Identify the file type, validate the schema, and prepare the data for downstream processing. Handle CSV, PDF, and HTML formats appropriately.",
    expected_output="A summary containing: File type and validation status, extracted data structure (columns, data types, shape), sample records for preview and any parsing warnings or errors",
    agent=file_intake_agent
)

data_preprocessing_task = Task(
    description="Clean and preprocess the extracted data from the file intake stage. Handle missing values, convert data types as needed, and ensure data quality. Prepare the dataset for statistical analysis.",
    expected_output="A preprocessing report containing: a) Data cleaning steps performed b) Missing value treatment summary c) Final dataset dimensions and structure d) Data quality metrics",
    agent=data_preprocessing_agent
)

statistical_analysis_task = Task(
    description="Perform comprehensive descriptive statistical analysis on the cleaned data. Calculate mean, median, mode, quartiles (Q1, Q2, Q3), IQR, and correlation matrix for all numerical variables. Identify key statistical patterns.",
    expected_output="A detailed statistical report including:a) Descriptive statistics (mean, median, mode, std dev) for each numerical column b) Quartile analysis (Q1, Q2, Q3) and IQR c) Correlation matrix showing relationships between variables d) Summary of statistical findings",
    agent=statistical_analyst_agent
)

visualization_task = Task(
    description="Create visualizations to represent the data insights. Generate: 1. Line charts for trend analysis 2. Bar charts for distribution analysis 3. Scatter plots for correlation visualization 4. Save all plots for inclusion in the final report.",
    expected_output="A visualization summary containing: 1. List of generated charts with descriptions 2. File paths to saved visualizations 3. Key visual insights observed",
    agent=visualization_specialist_agent
)

reporting_task = Task(
    description="Synthesize all findings from statistical analysis and visualizations into a comprehensive report. Provide actionable recommendations based on the data insights. Structure the report for executive presentation.",
    expected_output="A comprehensive final report including: a) Executive summary of key findings b) Detailed statistical insights with interpretations c) Visualization summaries d) Data-driven recommendations (minimum 3-5 recommendations) e)Conclusion and next steps",
    agent=reporting_analyst_agent
)

# ============================================================================
# 8. CREWAI CREW ORCHESTRATION
# ============================================================================

analysis_crew = Crew(
    agents=[
        file_intake_agent,
        data_preprocessing_agent,
        statistical_analyst_agent,
        visualization_specialist_agent,
        reporting_analyst_agent
    ],
    tasks=[
        file_intake_task,
        data_preprocessing_task,
        statistical_analysis_task,
        visualization_task,
        reporting_task
    ],
    process=Process.sequential,
    verbose=True
)

# ============
# 4. Kickoff
# ============

results = analysis_crew.kickoff(inputs={"csv_file": r"C:\Desktop\code\Aentic workflow part-II\class AI & DS.csv"})
print(results)