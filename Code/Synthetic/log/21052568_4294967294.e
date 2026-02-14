Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 775, in <module>
    dual_encoder = Classification.train_model(train=data_tr_encoder.loc[train_idx], val=data_tr_encoder.loc[val_idx],
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 71, in train_model
    dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared, train_weights=train_weights, val_weights=val_weights, df_name=df_name)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 24, in learn
    source = np.array([emb for emb in td_s])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (1658, 2) + inhomogeneous part.
