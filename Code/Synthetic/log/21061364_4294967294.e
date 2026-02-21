/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 941, in <module>
    run_experiments(
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 643, in run_experiments
    dual_encoder = Classification.train_model(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 146, in train_model
    dual_encoder = learn(
                   ^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 88, in learn
    encoder = SharedDualEncoder.create_encoder(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/SharedDualEncoder.py", line 72, in create_encoder
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
                                  ^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/ops/operation.py", line 228, in input
    return self._get_node_attribute_at_index(0, "input_tensors", "input")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/ops/operation.py", line 259, in _get_node_attribute_at_index
    raise ValueError(
ValueError: The layer sequential has never been called and thus has no defined input.
