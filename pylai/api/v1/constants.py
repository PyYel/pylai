# remove this file and replca it with a local project module

import os, sys
import logging
from dotenv import load_dotenv
import json

load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
    ),
    override=False,
)

logger = logging.Logger(name="Constants")


# CONSTANTS
MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(MAIN_DIR)

TMP_DIR = os.path.join(MAIN_DIR, "tmp")
LOGS_DIR = os.path.join(MAIN_DIR, "logs")
ASSETS_DIR = os.path.join(MAIN_DIR, "assets")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
BIN_DIR = os.path.join(ROOT_DIR, "bin")

# CONFIG

# SECRETS
HF_TOKEN = os.getenv("HF_TOKEN", "missing_var")
JWT_SECRET = os.getenv("JWT_SECRET", "missing_var")


BUCKET_NAME = os.getenv("BUCKET_NAME", "missing_var")


MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "missing_var")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "missing_var")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "missing_var")

POSTGRES_MASTER_USER = os.getenv("POSTGRES_MASTER_USER", "missing_var")
POSTGRES_MASTER_PASSWORD = os.getenv("POSTGRES_MASTER_PASSWORD", "missing_var")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "missing_var")
POSTGRES_USER = os.getenv("POSTGRES_USER", "missing_var")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "missing_var")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "missing_var")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "missing_var")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "missing_var")

SLM_MODEL_NAME = os.getenv("SLM_MODEL_NAME", "missing_var")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "missing_var")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "missing_var")


