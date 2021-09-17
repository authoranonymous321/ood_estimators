# Out-of-Distribution estimators

This repository contains out-of-distribution evaluation metrics 
introduced in article "Unsupervised Estimation of Out-of-Distribution Performance of Language Models"

## Reproduce results

Install the package

```bash
# presumes an existing python3.8 environment of your choice
git clone github.com/{this_repo}/ood_estimators.git
cd ood_estimators
pip install -e ood-estimators
```

Train the classifier, or translator, to collect the models with evaluations of the measures:
```bash
python classification_training.py [-h] [--base_model bert-base-cased] [--data_dir ./data] [--output_dir ./training_output]
```
or
```bash
python translation_training.py [-h] [--base_model xlm-roberta-base] [--data_dir ./data] [--output_dir ./training_output]
```

The raw values of the evaluated metrics are collected in `runs/{run-start-date}/callbacks.tsv`. 
The correlations of each of the metrics to OOD Accuracy metric are the ones reported in the article.

As discussed, the metrics can be used, for example, for picking the most-robust checkpoint of the training, 
where using some of the novel metrics as an indicator of the robustness is more informative, than using ID accuracy or loss. 

We implement this use-case in `classification_evaluation.py` script:

```bash
python classification_evaluation.py [-h] [--checkpoints_path ./training_output] [--callbacks_tsv_log ./runs/{run-start-date}/callbacks.tsv]
```
This will print both in-distribution and out-of-distribution (zero-shot) performance of the models 
picked according to the value of the implemented evaluation measures.

\ | ID Accuracy | ID Loss | DC | OC | PC  
--- | --- | --- | --- | --- | --- 
ID Accuracy | 88.33% | 92.50% | 92.50% | 90.83% | 94.16% 
OD Accuracy | 83.330% | 88.88% | 88.88% | 80.55% | 77.77% 


## Use metrics for your own use-case

Install the package (or include it into the requirements of your project)

```bash
# we presume an existing python3.8 environment of your choice
pip install git+git://github.com/{this_repo}/ood_estimators.git
```

Then, use any of the implemented estimators in your training, as callback. 
Callbacks can be used out-of-box with HuggingFace Transformers' Trainer, see `classification_training.py`,
but can be easily integrated to any other training pipeline, just by passing the evaluation dataloader and a model
to a selected metric:

```python
from ood_estimators.discriminative.consistency import PredictionConsistency
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor

# get some model to evaluate
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)

# get a DataLoader for some Dataset
eval_dataloader = DataLoader(TensorDataset(tensor([[0.2, 0.3], [0.4, 0.5]])), batch_size=32)

pc_estimator = PredictionConsistency()
pc_estimator.evaluate(model, eval_dataloader)
```

See the documentation of selected metric for a full list of its parameters.
