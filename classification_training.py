import argparse

from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from ood_estimators.datasets.twenty_news import TwentyNewsDataset
from ood_estimators.custom_callbacks import get_callbacks

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--base_model', type=str,
                           help='Minimal confidence threshold for sentences to train on.',
                           default='bert-base-cased')
    argparser.add_argument('--data_dir', type=str,
                           help='Minimal confidence threshold for sentences to train on.',
                           default='data')
    argparser.add_argument('--ood_targets', type=str,
                           help='Coma-separated list of low-level OOD categories of TwentyNews '
                                'to be used for zero-shot evaluation')
    argparser.add_argument('--output_dir', type=str,
                           help='Minimal confidence threshold for sentences to train on.',
                           default='training_output')
    argparser.add_argument('--device', type=str,
                           help='Device to use for training')


    args = argparser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    ood_targets = args.ood_targets.split(",")

    train_dataset = TwentyNewsDataset(domain="id", split="train", data_dir="data", tokenizer=tokenizer, ood_targets=ood_targets)
    val_dataset = TwentyNewsDataset(domain="id", split="val", data_dir="data", tokenizer=tokenizer, ood_targets=ood_targets)
    ood_val_dataset = TwentyNewsDataset(domain="ood", split="val", data_dir="data", tokenizer=tokenizer, ood_targets=ood_targets)

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model,
                                                               num_labels=train_dataset.get_num_labels())

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        warmup_steps=50,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=5,
        logging_first_step=True,
        weight_decay=0.1,
        learning_rate=2e-5,
        gradient_accumulation_steps=10,
        save_steps=5,
        label_smoothing_factor=0.1
    )

    callbacks_args = {
        "ood_val_dataset": ood_val_dataset,  # OOD accuracy callback
        "cyclic_translation_device": args.device,  # DC callback
        "pc_r": 10, "pc_dropout": 0.4, "pc_buffer_device": args.device,  # PC callback
        "threshold": 0.95  # OC callback
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=get_callbacks(task="classification", **callbacks_args)
    )

    trainer.train()
