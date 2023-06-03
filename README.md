# Implementation-of-Mixed-depthwise-layer-kernel-in-tensorflow-2.0

```
def split_layer(total_channels, num_groups):
    split = [int(tf.math.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split

class DepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, in_channels, kernel_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            depth_multiplier=1,
            use_bias=bias
        )

    def call(self, x):
        out = self.depthwise_conv(x)
        return out

class MDConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)
        
        self.layers = []
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(
                DepthwiseConv2D(
                    in_channels=self.split_out_channels[idx],
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias
                )
            )

    def call(self, x):
        
        split = tf.split(x, self.split_out_channels, axis=3)
        out = tf.concat([layer(s) for layer, s in zip(self.layers, split)], axis=3)
        return out
```
