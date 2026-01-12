# Recommender System Pipeline

A machine learning pipeline for building and evaluating recommender systems.

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

Run the main pipeline:

```bash
python main.py
```

## Output

The pipeline generates the following outputs:

- `output/models/` - Trained model files
- `output/logs/` - Execution logs
- `output/evaluations/` - Evaluation metrics and results

## Demo

Run the interactive demo to test recommender models:

```bash
python demo.py
```

### Available Commands

| Command | Description |
|---------|-------------|
| `[number]` | Enter a test user index (0 to N-1) to view recommendations for that user |
| `r` | Select a random test user |
| `c` | Create a custom user history by entering item IDs |
| `q` | Quit the demo |

### Creating Custom User History

Use the `c` command to test recommendations for custom browsing history:

```
Enter command: c
Available item IDs are numeric (e.g., 1000, 1001, 1002, ...)
Enter item IDs separated by comma: 1243,1025,1062
```

The demo will display your custom history and show recommendations from all trained models.
