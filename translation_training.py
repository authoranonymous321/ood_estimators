import argparse
import os

from transformers import Trainer, Seq2SeqTrainingArguments, SchedulerType

from ood_estimators.datasets.opus import OPUSDataset, SRC_LANG, TGT_LANG
from ood_estimators.custom_callbacks import get_callbacks
from ood_estimators.utils.model_utils import get_initial_tokenizer_model
from ood_estimators.utils.logging_utils import CALLBACKS_LOGGING_FNAME

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_dir', type=str,
                           help='Directory where to download store the training data.',
                           default='data')
    argparser.add_argument('--output_dir', type=str,
                           help="Where to store the fine-tuned models' checkpoints",
                           default='training_output')

    args = argparser.parse_args()

    tokenizer, model = get_initial_tokenizer_model(src_lang_iso=SRC_LANG,
                                                   tgt_lang_iso=TGT_LANG,
                                                   reinit_weights=True)

    train_dataset = OPUSDataset(domain="id", split="train", data_dir="data", tokenizer=tokenizer)
    val_dataset = OPUSDataset(domain="id", split="val", data_dir="data", tokenizer=tokenizer, firstn=100)
    ood_val_dataset = OPUSDataset(domain="ood", split="val", data_dir="data", tokenizer=tokenizer, firstn=100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        warmup_steps=5000,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=250,
        save_steps=250,
        logging_first_step=True,
        weight_decay=0.05,
        learning_rate=5e-4,
        gradient_accumulation_steps=8,
        dataloader_num_workers=5,
        # label_smoothing_factor=0.15,
        # lr_scheduler_type=SchedulerType.POLYNOMIAL,  # originally trained with inv-sqrt
        max_grad_norm=2,
    )

    print("Evaluation metrics will be logged to %s" % os.path.join(training_args.logging_dir, CALLBACKS_LOGGING_FNAME))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=get_callbacks(task="translation", ood_val_dataset=ood_val_dataset)
    )

    trainer.train()
