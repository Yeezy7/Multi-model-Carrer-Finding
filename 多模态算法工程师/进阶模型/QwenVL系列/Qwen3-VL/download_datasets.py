from pathlib import Path
from datasets import load_dataset

ROOT = Path('/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/datasets')
ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ('DocVQA', 'nlpconnect/DocVQA'),
    ('ChartQA', 'ahmed-masry/ChartQA'),
    ('InfographicVQA', 'mm-eval/InfographicVQA'),
]

for name, dataset_id in DATASETS:
    out_dir = ROOT / name
    print(f'\n=== Downloading {name} from {dataset_id} to {out_dir} ===')
    try:
        ds = load_dataset(dataset_id)
        ds.save_to_disk(str(out_dir))
        print('SUCCESS')
        print('Splits:', list(ds.keys()))
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}')
        print('Please check network connectivity and HF access.')
