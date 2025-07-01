在PyTorch中，`enc_output[-1]` 这种语法是用来获取张量最后一个时间步的输出，这是Python/PyTorch中常见的索引操作。具体解释如下：

1.  **张量形状**\
    `enc_output` 的形状是 `(num_steps, batch_size, num_hiddens)`，其中：
    *   `num_steps`：序列长度（时间步数）
    *   `batch_size`：批大小
    *   `num_hiddens`：隐藏层维度

2.  **`[-1]` 的含义**
    *   在Python中，`-1` 表示倒数第一个元素
    *   对三维张量 `enc_output` 来说，`enc_output[-1]` 表示取第一个维度（时间步维度）的最后一个时间步
    *   相当于 `enc_output[num_steps-1]`

3.  **结果形状**\
    操作后的 `context` 形状为 `(batch_size, num_hiddens)`，即：
    *   去掉了时间步维度
    *   保留了批次和隐藏层维度

4.  **在Seq2Seq中的意义**\
    这里获取的是编码器在最后一个时间步的所有隐藏状态，通常包含了对整个输入序列的编码信息，用作解码器的初始上下文向量。

5.  **等价写法**\
    以下写法效果相同：
    ```python
    context = enc_output[-1, :, :]  # 显式指定所有批次和隐藏单元
    ```

这种语法简洁高效，是处理时序数据时的常用操作模式。
