import tensorflow as tf

print(tf.__version__)

print('gpu available',tf.test.is_gpu_available(cuda_only=True))
print('built with cuda',tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
