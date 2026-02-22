Traceback (most recent call last):
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'gender'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 945, in <module>
    run_experiments(
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 616, in run_experiments
    _, df_name, train, test = _load_dataset(dataset)
                              ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 600, in _load_dataset
    return dataset_loaders[dataset_name]()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 252, in make_adult
    df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)
                   ~~^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'gender'
