import json
from pathlib import Path

from src.utils import ROOT_PATH


def get_texts_from_json(json_path: Path):
    assert json_path.exists(), f'path does not exist: {json_path}'
    with json_path.open() as f:
        index = json.load(f)
    
    texts = [item['text'] for item in index]
    
    return '\n'.join(texts)


def collect_train_text_data():
    """
    Collects all the textual data from the train datasets
    """
    output_path = ROOT_PATH / "saved" / "train_text.txt"
    if output_path.exists():
        return output_path
    
    json_paths = [
        ROOT_PATH / "data" / "datasets" / "librispeech" / "train-clean-100_index.json",
        ROOT_PATH / "data" / "datasets" / "librispeech" / "train-clean-360_index.json",
        ROOT_PATH / "data" / "datasets" / "librispeech" / "train-other-500_index.json",
    ]
    texts = '\n'.join([get_texts_from_json(path) for path in json_paths])

    with output_path.open('w') as f:
        f.write(texts)
    
    return output_path
