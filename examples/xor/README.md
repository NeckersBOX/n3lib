# N3L Example - XOR

## Description
This is the most common example provided by various resources online when they explain how a neural network works.

# Reports
## XOR - Report
### Report ID: fe041f3b-5b67-4297-b594-bd65b22f4dc5

### Configuration

| Conf              | Value          |
|-------------------|----------------|
| Iterations        | `10000`     |
| Learning Rate     | `1.00`      |
| Input Neurons     | `2`          |
| Hidden Neurons    | `3`          |
| Hidden Layers     | `1`          |
| Output Neurons    | `1`          |
| Input Act         | `None`       |
| Hidden Act        | `Sigmoid`    |
| Output Act        | `Sigmoid`    |
| **Extra Args**    | `-b 0.5` |

### Learning Graph
- **MNS:** It's the Mobile Network Success rate. Range from 0 to 1. Higher is better.
- **MNE:** It's the Mobile Network Error rate. Lower is better.

![MNE Plot](xor.report.fe041f3b-5b67-4297-b594-bd65b22f4dc5.plot-mne.png)

![MNS Plot](xor.report.fe041f3b-5b67-4297-b594-bd65b22f4dc5.plot-mns.png)

### Memory Usage Graph
Memory usage was evaluated by _massif_ tool.

![Massif](xor.report.fe041f3b-5b67-4297-b594-bd65b22f4dc5.memory.png)

### Execution Time

| Mode                 | Time ( seconds )   |
|----------------------|--------------------|
| Forward Propagation  | `2.678789063`  |
| Backward Propagation | `3.952529540` |
