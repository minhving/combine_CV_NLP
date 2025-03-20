# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import login
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import json
from sklearn.ensemble import RandomForestRegressor
import time
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import torch

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from RAG import *
import webbrowser