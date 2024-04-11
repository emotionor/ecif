import os

import pandas as pd
from pyecif import WriteEcif, LoadEcif
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)
tasknames = mb.metadata.keys()
for taskname in tasknames:
    task = getattr(mb, taskname)
    task.load()
    for fold in range(5):
        train_input, train_output = task.get_train_and_val_data(fold)
        test_input, test_output = task.get_test_data(fold, include_target=True)
        train_df = pd.concat([train_input, train_output], axis=1)
        test_df = pd.concat([test_input, test_output], axis=1)

        train_path = f"matbench/{taskname}/train_fold_{fold}.ecif"
        test_path = f"matbench/{taskname}/test_fold_{fold}.ecif"
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        if task.metadata.input_type == 'structure':
            WriteEcif(train_df, train_path, cifColName='structure', properties=train_df.columns)
            WriteEcif(test_df, test_path, cifColName='structure', properties=test_df.columns)
        else:
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
