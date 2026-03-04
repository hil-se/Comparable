/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-03-03 09:28:09.693938: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_1/functional_39_1/dropout_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-03-03 09:49:37.371617: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_1_1/functional_79_1/dropout_2_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1772550670.364632  737636 service.cc:145] XLA service 0x7f1dbf55f0e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1772550670.366466  737636 service.cc:153]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1772550678.187216  737636 asm_compiler.cc:369] ptxas warning : Registers are spilled to local memory in function 'input_reduce_select_fusion_10', 164 bytes spill stores, 164 bytes spill loads
ptxas warning : Registers are spilled to local memory in function 'input_reduce_select_fusion_9', 164 bytes spill stores, 164 bytes spill loads

I0000 00:00:1772550678.195913  737636 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
/home/xx4455/Comparable/Code/Synthetic/metrics.py:45: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return spearmanr(self.y, self.y_pred)[0]
/home/xx4455/Comparable/Code/Synthetic/metrics.py:39: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(self.y, self.y_pred)[0]
/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-03-03 10:28:20.345492: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_2_1/functional_159_1/dropout_6_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-03-03 10:50:47.808952: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_3_1/functional_199_1/dropout_8_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
/home/xx4455/Comparable/Code/Synthetic/metrics.py:45: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return spearmanr(self.y, self.y_pred)[0]
/home/xx4455/Comparable/Code/Synthetic/metrics.py:39: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(self.y, self.y_pred)[0]
/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-03-03 11:28:57.642546: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_4_1/functional_279_1/dropout_12_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-03-03 11:50:25.135766: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_5_1/functional_319_1/dropout_14_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
/home/xx4455/Comparable/Code/Synthetic/metrics.py:45: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return spearmanr(self.y, self.y_pred)[0]
/home/xx4455/Comparable/Code/Synthetic/metrics.py:39: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(self.y, self.y_pred)[0]
/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-03-03 12:29:05.213741: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_6_1/functional_399_1/dropout_18_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-03-03 12:50:33.326520: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_7_1/functional_439_1/dropout_20_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
/home/xx4455/Comparable/Code/Synthetic/metrics.py:45: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return spearmanr(self.y, self.y_pred)[0]
/home/xx4455/Comparable/Code/Synthetic/metrics.py:39: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(self.y, self.y_pred)[0]
/.autofs/tools/spack/var/spack/environments/default-ml-x86_64-25052701/.spack-env/view/lib/python3.11/site-packages/keras/src/layers/reshaping/zero_padding2d.py:72: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
2026-03-03 13:28:43.810101: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_8_1/functional_519_1/dropout_24_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-03-03 13:50:11.386064: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/dual_encoder_all_9_1/functional_559_1/dropout_26_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
/home/xx4455/Comparable/Code/Synthetic/metrics.py:45: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return spearmanr(self.y, self.y_pred)[0]
/home/xx4455/Comparable/Code/Synthetic/metrics.py:39: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return pearsonr(self.y, self.y_pred)[0]
