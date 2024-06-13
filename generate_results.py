import glob
import logging
import os

import pandas as pd
import torch.nn
from torch import tensor
from torchmetrics.classification import BinaryCohenKappa, BinaryPrecision, \
    BinaryRecall, BinaryMatthewsCorrCoef, BinaryAUROC

from src.misc.utils import set_logging, set_seed, create_args


def run(file_pattern: str):
    parser = create_args(seeds_args=False, benchmark_args=False)
    args = parser.parse_args()

    set_logging(args.project_path, file_name="logs_results")
    set_seed(1)
    logging.info("Args: %s", args)

    # Read predictions:
    filepaths = glob.glob(
        os.path.join(args.project_path, file_pattern))
    if not filepaths:
        logging.warning("No files found. Quitting...")
        return
    results = []
    for fp in filepaths:
        with open(fp, 'r') as f:
            results.append(pd.read_pickle(f.name))
    results = pd.concat(results)

    # Apply sigmoid to predictions based on the model type:
    mask = (results['model'].isin(['eegnet', 'lstm', 'uercm']))
    masked_results = results.loc[mask]
    results.loc[mask, 'predictions'] = masked_results['predictions'].apply(
        lambda x: torch.nn.Sigmoid()(torch.FloatTensor([x])).tolist().pop())

    # Calculate classification metrics:
    groups = results.groupby(['seed', 'user', 'model', 'strategy', "reading_task"])
    mcc = groups.apply(
        lambda x: BinaryMatthewsCorrCoef()(tensor(x.predictions.tolist()),
                                           tensor(x.targets.tolist())).item(),
        include_groups=False).reset_index()
    mcc.rename(columns={0: 'mcc'}, inplace=True)
    kappa = groups.apply(
        lambda x: BinaryCohenKappa()(tensor(x.predictions.tolist()),
                                     tensor(x.targets.tolist())).item(),
        include_groups=False).reset_index()
    kappa.rename(columns={0: 'kappa'}, inplace=True)

    precision = groups.apply(
        lambda x: BinaryPrecision()(tensor(x.predictions.tolist()),
                                    tensor(x.targets.tolist())).item(),
        include_groups=False).reset_index()
    precision.rename(columns={0: 'precision'}, inplace=True)

    recall = groups.apply(
        lambda x: BinaryRecall()(tensor(x.predictions.tolist()),
                                 tensor(x.targets.tolist())).item(),
        include_groups=False).reset_index()
    recall.rename(columns={0: 'recall'}, inplace=True)

    auc = groups.apply(
        lambda x: BinaryAUROC()(tensor(x.predictions.tolist()),
                                tensor(x.targets.tolist())).item(),
        include_groups=False).reset_index()
    auc.rename(columns={0: 'auc'}, inplace=True)

    # Concatenate metrics:
    metrics = mcc.merge(precision, on=['seed', 'user', 'model', 'strategy', "reading_task"])
    metrics = metrics.merge(kappa, on=['seed', 'user', 'model', 'strategy', "reading_task"])
    metrics = metrics.merge(recall, on=['seed', 'user', 'model', 'strategy', "reading_task"])
    metrics = metrics.merge(auc, on=['seed', 'user', 'model', 'strategy', "reading_task"])

    for model in metrics.model.unique():
        latex_output = ""
        for strategy in metrics.strategy.unique()[::-1]:
            logging.info("\n\nModel: %s, Strategy: %s", model, strategy)
            for metric in ['auc', 'precision', 'recall']:
                mean = metrics[(metrics['model'] == model) & (
                            metrics['strategy'] == strategy)][
                    metric].mean()
                std = metrics[(metrics['model'] == model) & (
                            metrics['strategy'] == strategy)][
                    metric].std()
                logging.info("%s: %.2f +- %.2f", metric, mean, std)
                latex_output += f'& {mean:.2f} ({std:.2f}) '

        logging.info(latex_output)


if __name__ == "__main__":
    run(file_pattern="w_relevance_seed*.pkl")
    run(file_pattern="s_relevance_seed*.pkl")
