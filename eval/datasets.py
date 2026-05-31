"""
Dataset loaders for evaluation.

Supports:
- LOCOMO (LoCoMo dataset)
- LongMemEval
- PerLTQA
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_locomo(
    data_path: str = None,
    conv_id: str = None,
    num_samples: int = None
) -> List[Dict]:
    """
    Load LOCOMO dataset.

    Args:
        data_path: Path to processed LOCOMO data (locomo_conversations.json)
        conv_id: Filter to specific conversation ID
        num_samples: Limit number of questions

    Returns:
        List of question samples with haystack sessions
    """
    if data_path is None:
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "locomo" / "locomo_conversations.json"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"LOCOMO data not found: {data_path}"
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    if conv_id:
        conversations = [c for c in conversations if c['conversation_id'] == conv_id]
        if not conversations:
            raise ValueError(f"Conversation {conv_id} not found")

    data = []
    for conv in conversations:
        for q in conv['questions']:
            sample = {
                **q,
                'conversation_id': conv['conversation_id'],
                'haystack_sessions': conv['haystack_sessions'],
                'haystack_session_datetimes': conv['haystack_session_datetimes'],
                'num_sessions': conv['num_sessions'],
                'speaker_a': conv.get('speaker_a'),
                'speaker_b': conv.get('speaker_b'),
            }
            data.append(sample)

    if num_samples:
        data = data[:num_samples]

    return data


def load_longmemeval(
    data_path: str = None,
    sample_id: str = None,
    num_samples: int = None,
    subset_file: str = None,
    sample_ids: List[str] = None
) -> List[Dict]:
    """
    Load LongMemEval dataset.

    LongMemEval structure:
    - Each sample has independent haystack (sessions)
    - Each sample = one question with its own conversation history

    Args:
        data_path: Path to LongMemEval data (longmemeval_s_cleaned.json)
        sample_id: Filter to specific sample ID (question_id)
        num_samples: Limit number of samples
        subset_file: Path to subset config file with sample_ids

    Returns:
        List of samples with standardized format
    """
    base_dir = Path(__file__).parent.parent

    if data_path is None:
        data_path = base_dir / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"LongMemEval data not found: {data_path}\n"
            f"Please provide the dataset file."
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if sample_ids:
        sample_id_set = set(sample_ids)
        data = [d for d in data if d['question_id'] in sample_id_set]
        print(f"📦 Using predefined split: {len(data)} samples")

    if subset_file:
        subset_path = Path(subset_file)
        if not subset_path.is_absolute():
            subset_path = base_dir / subset_file

        if not subset_path.exists():
            raise FileNotFoundError(f"Subset config not found: {subset_path}")

        with open(subset_path, 'r', encoding='utf-8') as f:
            subset_config = json.load(f)

        if 'sample_ids' not in subset_config:
            raise ValueError("Invalid subset config: missing 'sample_ids' field")

        sample_id_set = set(subset_config['sample_ids'])
        data = [d for d in data if d['question_id'] in sample_id_set]

        print(f"📦 Using subset: {subset_path.name} ({len(data)} samples)")

    if sample_id:
        data = [d for d in data if d['question_id'] == sample_id]
        if not data:
            raise ValueError(f"Sample ID {sample_id} not found")

    for sample in data:
        if 'haystack_dates' in sample and 'haystack_session_datetimes' not in sample:
            sample['haystack_session_datetimes'] = sample['haystack_dates']
        if 'question_type' not in sample:
            sample['question_type'] = sample.get('type', 'unknown')

    if num_samples:
        data = data[:num_samples]

    print(f"   Loaded LongMemEval: {len(data)} samples")
    return data


def load_perltqa(
    data_path: str = None,
    character_id: str = None,
    num_samples: int = None,
    subset_file: str = None
) -> List[Dict]:
    """
    Load PerLTQA dataset.

    PerLTQA structure:
    - Each character has their own haystack (sessions) and questions
    - Returns list of characters, each containing sessions and questions

    Args:
        data_path: Path to PerLTQA data (perltqa_v2_standard.json or perltqa_standard.json)
        character_id: Filter to specific character ID
        num_samples: Limit number of characters
        subset_file: Path to subset config file with character_ids

    Returns:
        List of character samples, each with:
        - character_id, character_name
        - haystack_sessions (conversation history)
        - questions (list of QA pairs)
        - character_profile (optional metadata)
    """
    base_dir = Path(__file__).parent.parent

    if data_path is None:
        data_path = base_dir / "data" / "perltqa" / "perltqa_v2_standard.json"
        if not data_path.exists():
            data_path = base_dir / "data" / "perltqa" / "perltqa_standard.json"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"PerLTQA data not found: {data_path}\n"
            f"Please provide the dataset file."
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        all_characters = json.load(f)

    if subset_file:
        subset_path = Path(subset_file)
        if not subset_path.is_absolute():
            subset_path = base_dir / subset_file

        if not subset_path.exists():
            raise FileNotFoundError(f"Subset config not found: {subset_path}")

        with open(subset_path, 'r', encoding='utf-8') as f:
            subset_config = json.load(f)

        if 'character_ids' not in subset_config:
            raise ValueError("Invalid subset config: missing 'character_ids' field")

        char_ids = set(subset_config['character_ids'])
        data = [c for c in all_characters if c['character_id'] in char_ids]

        total_q = sum(c.get('num_questions', len(c.get('questions', []))) for c in data)
        print(f"📦 Using subset: {subset_path.name} ({len(data)} characters, {total_q} questions)")
    else:
        data = all_characters
        total_q = sum(c.get('num_questions', len(c.get('questions', []))) for c in data)
        print(f"💡 Full dataset: {len(data)} characters, {total_q} questions")

    if character_id:
        data = [c for c in data if c['character_id'] == character_id]
        if not data:
            raise ValueError(f"Character ID {character_id} not found")

    for char in data:
        if 'haystack_sessions' not in char and 'sessions' in char:
            char['haystack_sessions'] = char['sessions']
        if 'haystack_session_datetimes' not in char:
            if 'session_dates' in char:
                char['haystack_session_datetimes'] = char['session_dates']
            else:
                char['haystack_session_datetimes'] = []
        if 'num_sessions' not in char:
            char['num_sessions'] = len(char.get('haystack_sessions', []))
        for q in char.get('questions', []):
            if 'question_type' not in q:
                q['question_type'] = q.get('type', 'unknown')

    if num_samples:
        data = data[:num_samples]

    return data


def load_dataset(
    dataset_name: str,
    data_path: str = None,
    conv_id: str = None,
    sample_id: str = None,
    character_id: str = None,
    num_samples: int = None,
    subset_file: str = None,
    **kwargs
) -> List[Dict]:
    """
    Unified dataset loader.

    Args:
        dataset_name: 'locomo', 'longmemeval', or 'perltqa'
        data_path: Path to dataset
        conv_id: Conversation ID (for locomo)
        sample_id: Sample ID (for longmemeval)
        character_id: Character ID (for perltqa)
        num_samples: Limit number of samples
        subset_file: Path to subset config file

    Returns:
        List of samples
    """
    sample_ids = kwargs.get('sample_ids', None)
    loaders = {
        'locomo':      lambda: load_locomo(data_path, conv_id, num_samples),
        'longmemeval': lambda: load_longmemeval(data_path, sample_id, num_samples, subset_file, sample_ids),
        'perltqa':     lambda: load_perltqa(data_path, character_id, num_samples, subset_file),
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(loaders.keys())}")

    return loaders[dataset_name]()


def get_current_time(sample: Dict) -> Optional[str]:
    """Get the current time reference from the dataset sample (for relative time reasoning)."""
    current_time = sample.get('question_date')
    if not current_time:
        timestamps = sample.get('haystack_session_datetimes', [])
        current_time = timestamps[-1] if timestamps else None
    return current_time
