import os
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log

_HERE = Path(__file__).parent
_env_file = _HERE / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_file, override=True)

logger = logging.getLogger(__name__)

RETRY_TIMES = int(os.getenv('RETRY_TIMES', '3'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER', '1'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER', '10'))

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1')

common_params = {}

if os.getenv('OPENAI_MAX_TOKENS'):
    common_params["max_tokens"] = int(os.getenv('OPENAI_MAX_TOKENS'))

if os.getenv('OPENAI_TEMPERATURE'):
    common_params["temperature"] = float(os.getenv('OPENAI_TEMPERATURE'))

if os.getenv('OPENAI_TIMEOUT'):
    common_params["timeout"] = int(os.getenv('OPENAI_TIMEOUT'))


client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def llm_request(prompt):
    """Send a prompt to the configured LLM and return the response text."""
    response_obj = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **common_params
    )
    return response_obj.choices[0].message.content


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def llm_request_for_json(prompt):
    """Send a prompt and return the parsed JSON object from a ```json block."""
    import re
    response_obj = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **common_params
    )
    content = response_obj.choices[0].message.content or ""
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in model output: {content}")
    return __import__("json").loads(match.group(1).strip())
