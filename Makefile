PYTHONPATH := PYTHONPATH=.
PY := $(PYTHONPATH) .venv/bin/python

EPOCHS ?= 50

.PHONY: dataset train cnn feature_extractor fine_tuning plots clean

dataset:
	$(PY) -c "from src.dataset import get_dataloaders; get_dataloaders()"

train:
	$(PY) main.py --epochs $(EPOCHS)

cnn:
	$(PY) main.py --step cnn --epochs $(EPOCHS)

feature_extractor:
	$(PY) main.py --step feature_extractor --epochs $(EPOCHS)

fine_tuning:
	$(PY) main.py --step fine_tuning --epochs $(EPOCHS)

plots:
	$(PY) main.py --step plots

clean:
	rm -rf logs/ checkpoints/ loss_comparison.png accuracy_comparison.png
