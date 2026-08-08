import os
import sys
import requests
from dotenv import load_dotenv

MAIN_DIR_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if __name__ == "__main__":
    sys.path.append(MAIN_DIR_PATH)

load_dotenv(os.path.join(os.path.dirname(MAIN_DIR_PATH), ".env"), override=False)
