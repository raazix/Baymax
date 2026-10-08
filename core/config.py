import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{(ROOT / "data" / "lineguard.db").as_posix()}')
if DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = 'postgresql+psycopg://' + DATABASE_URL.removeprefix('postgresql://')
RULE_VERSION = 'demo-engineering-v1'
