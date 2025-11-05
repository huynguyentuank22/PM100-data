Run file `finetune_SBERT_multitask.py`

```bash
python .\finetune_SBERT_multitask.py --input .\data\job_table.parquet
```

đã thử
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

đang thử
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)