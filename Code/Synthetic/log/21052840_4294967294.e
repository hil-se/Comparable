2026-02-14 23:11:40.561513: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/Synthetic/synthetic.py", line 828, in <module>
    dual_encoder = Classification.train_model(train=data_tr_encoder, val=None,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 64, in train_model
    dual_encoder = learn(train, epochs=epochs, validation_data=None, y_true=y_true, shared=shared,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/Classification.py", line 54, in learn
    dual_encoder.fit(
  File "/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/xx4455/Comparable/Code/Synthetic/SharedDualEncoder.py", line 154, in train_step
    loss = self.compute_loss(encodings_A, encodings_B, y, sample_weight=sample_weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xx4455/Comparable/Code/Synthetic/SharedDualEncoder.py", line 108, in compute_loss
    encodings_A = tf.squeeze(encodings_A, axis=-1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Can not squeeze dim[1], expected a dimension of 1, got 2622 for '{{node Squeeze}} = Squeeze[T=DT_FLOAT, squeeze_dims=[-1]](dual_encoder_all_1/sequential_1/activation_1/Softmax)' with input shapes: [?,2622].
