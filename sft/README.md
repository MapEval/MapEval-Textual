# MapEval SFT Experiment

## Results

### llama-3.2-3b

Response file: `llama-3.2-3b.json`

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 34.98 | 39.53 | 48.21 | 35.56 | 24.44 | 0.00 |
| Finetuned | 35.96 | 41.86 | 50.00 | 33.33 | 26.67 | 0.00 |

### phi-3.5-mini

Response file: `phi-3.5-mini.json`

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 39.90 | 44.19 | 53.57 | 53.33 | 17.78 | 0.00 |
| Finetuned | 34.48 | 46.51 | 48.21 | 26.67 | 24.44 | 0.00 |

### qwen-2.5-7b

Response file: `qwen-2.5-7b.json`

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 41.87 | 53.49 | 46.43 | 44.44 | 33.33 | 7.14 |
| Finetuned | 43.35 | 55.81 | 51.79 | 42.22 | 33.33 | 7.14 |

### llama-3.1-8b

Response file: `llama-3.1-8b.json`

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 46.31 | 58.14 | 62.50 | 51.11 | 20.00 | 14.29 |
| Finetuned | 44.33 | 34.88 | 60.71 | 60.00 | 26.67 | 14.29 |

### gemma-2.0-9b

Response file: `gemma-2.0-9b.json`

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 46.80 | 51.16 | 48.21 | 57.78 | 35.56 | 28.57 |
| Finetuned | 51.23 | 55.81 | 58.93 | 66.67 | 33.33 | 14.29 |

### llama-3.1-70b

|   | Overall | POI | Nearby | Routing | Trip | Unanswerable |
|---|:-------:|:---:|:------:|:-------:|:----:|:------------:|
| Pretrained | 59.11 | 69.77 | 66.07 | 68.89 | 33.33 | 50.00 |

## Datasets

- train.json - 97 samples
- test.json - 203 samples

## Codes

- Code for SFT: `train.py`
- Code for evaluation: `test.py`
