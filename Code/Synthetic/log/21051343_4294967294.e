Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 639, in <module>
    df, df_name, train, test = make_scut()
                               ^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 355, in make_scut
    X = features.drop([dependent], axis=1)
        ^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'drop'
