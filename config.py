"""
Configuration settings for the Multi-Document AI Knowledge Base.
Defines paths for data, db, and model configurations.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

# Add other core configurations here
