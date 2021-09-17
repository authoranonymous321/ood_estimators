import argparse
import csv
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ood_estimators.custom_callbacks import classification_estimators
from ood_estimators.datasets.twenty_news import TwentyNewsDataset
from ood_estimators.discriminative.accuracy import TestAccuracy

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--checkpoints_path', type=str,
                           help='A directory with checkpoints to pick the evaluated models from',
                           default='training_output/')
    argparser.add_argument('--callbacks_tsv_log', type=str,
                           help='A path to callbacks.tsv from training',
                           default='runs/')
    argparser.add_argument('--do_report', type=bool,
                           help='Whether to collect results to ')
    argparser.add_argument('--report_to', type=str,
                           help='A report tsv file to add the evaluation outputs to',
                           default='zero_shot_classification_report.tsv')
    argparser.add_argument('--ood_targets', type=str,
                           help='Coma-separated list of low-level OOD categories of TwentyNews '
                                'to be used for zero-shot evaluation')
    argparser.add_argument('--device', type=str,
                           help='Device to use for evaluation')

    args = argparser.parse_args()

    # callbacks.tsv file is expected to have a format (without header):
    # <step>\t<epoch>\t<value>\t<metric_spec>
    with open(args.callbacks_tsv_log, "r") as csv_file:
        logs = list(csv.reader(csv_file, delimiter="\t"))

    checkpoint_path_template = os.path.join(args.checkpoints_path, "checkpoint-%s")
    # we'll resolve tokenizer in advance so that we don't reload the same object repetitively
    first_tokenizer_path = checkpoint_path_template % logs[0][0]
    tokenizer = AutoTokenizer.from_pretrained(first_tokenizer_path)

    ood_targets = args.ood_targets.split(",")

    id_test_dataset = TwentyNewsDataset(domain="id", split="test", data_dir="data", tokenizer=tokenizer, ood_targets=ood_targets)
    ood_test_dataset = TwentyNewsDataset(domain="ood", split="test", data_dir="data", tokenizer=tokenizer, ood_targets=ood_targets)
    with open(args.report_to, "a") as report_file:
        if args.do_report:
            report_csv = csv.writer(report_file, delimiter="\t")

        # header = ["metric", "value", "checkpoint", "test_datasets", "distribution"]
        # report_csv.writerow(header)
        test_dataset_id = ", ".join(ood_test_dataset.ood_subtargets)

        for Estimator_cls in classification_estimators:
            primary_metric_logs = {int(row[0]): float(row[3]) for row in logs
                                   if Estimator_cls.label in row[2] and Estimator_cls.primary_metric in row[2]}

            # pick the best-performing checkpoint according to the logic of given Estimator
            picked_checkpoint = Estimator_cls.pick_best_performing_checkpoint(primary_metric_logs)

            print("Checkpoint picked by %s_%s: %s" % (Estimator_cls.label, Estimator_cls.primary_metric, picked_checkpoint))
            print()
            # load model at selected checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path_template % picked_checkpoint).to(args.device)

            id_metrics = TestAccuracy(test_dataset=id_test_dataset).evaluate(model, batch_size=4)
            if args.do_report:
                report_csv.writerow(["%s_%s" % (Estimator_cls.label, Estimator_cls.primary_metric),
                                     args.checkpoints_path, test_dataset_id, id_metrics['id_Accuracy_test_Accuracy'], "id"])

            ood_metrics = TestAccuracy(test_dataset=ood_test_dataset).evaluate(model, batch_size=4)
            if args.do_report:
                report_csv.writerow(["%s_%s" % (Estimator_cls.label, Estimator_cls.primary_metric),
                                     checkpoint_path_template % picked_checkpoint, test_dataset_id,
                                     ood_metrics['ood_Accuracy_test_Accuracy'], "ood"])

            print("Checkpoint %s performance summary: %s" % (picked_checkpoint, {**id_metrics, **ood_metrics}))
