"""
Preprocesses and prepares EEG data for benchmarking.
"""

import logging

from neurips.data_operations.preparator import DataPreparator
from neurips.data_operations.preprocessor import DataPreprocessor, \
    load_evoked_response, plot_erp
from neurips.misc.utils import set_logging, set_seed, create_args


def run():
    plot = False  # Should the figures be produced?
    parser = create_args()
    args = parser.parse_args()

    set_logging(args.project_path, file_name="logs_prepare")
    set_seed(1)
    logging.info('Args: %s', args)

    # Data pre-processing:
    data_preprocessor = DataPreprocessor(project_path=args.project_path)
    data_preprocessor.filter()
    data_preprocessor.create_epochs()
    data_preprocessor.clean()

    data_preparator = DataPreparator(
        data_dir=data_preprocessor.cleaned_data_dir)
    data_preparator.save_cleaned_data()
    data_preparator.prepare_data_for_benchmark()

    if plot:
        epochs = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            annotations=data_preparator.annotations,
            average=False)
        for electrode in epochs[0].ch_names:
            epochs = load_evoked_response(
                dir_cleaned=data_preprocessor.cleaned_data_dir,
                annotations=data_preparator.annotations,
                average=False)
            plot_erp(work_dir=args.project_path,
                     epos=epochs,
                     title=f"{electrode} electrode.",
                     queries=['annotation == 1',
                              'annotation == 0'],
                     file_id=electrode,
                     ch_names=[electrode],
                     l=['Semantically relevant',
                        'Semantically irrelevant'])

        relevant = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            filter_flag='annotation == 1',
            annotations=data_preparator.annotations)
        relevant.plot_joint(picks="eeg", times=[0.3, 0.4, 0.6],
                            title=None,
                            show=False,
                            ts_args=dict(ylim=dict(eeg=[-4.5, 5]), gfp=True))

        irrelevant = load_evoked_response(
            dir_cleaned=data_preprocessor.cleaned_data_dir,
            filter_flag='annotation == 0',
            annotations=data_preparator.annotations)
        irrelevant.plot_joint(picks="eeg", times=[0.3, 0.4, 0.6],
                              title=None,
                              ts_args=dict(ylim=dict(eeg=[-4.5, 5]), gfp=True))


if __name__ == '__main__':
    run()
