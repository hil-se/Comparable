Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 964, in <module>
    run_experiments(
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 634, in run_experiments
    _, df_name, train, test = _load_dataset(dataset)
                              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 607, in _load_dataset
    return dataset_loaders[dataset_name]()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 326, in make_heart
    X_train, X_test = _split_with_output(df, dependent)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 42, in _split_with_output
    X = df.drop(columns=[dependent])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/frame.py", line 5581, in drop
    return super().drop(
           ^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/generic.py", line 4788, in drop
    obj = obj._drop_axis(labels, axis, level=level, errors=errors)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/generic.py", line 4830, in _drop_axis
    new_axis = axis.drop(labels, errors=errors)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 7070, in drop
    raise KeyError(f"{labels[mask].tolist()} not found in axis")
KeyError: "['output'] not found in axis"
