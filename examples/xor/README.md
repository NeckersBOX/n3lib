# N3L Example - XOR

## Description
This is the most common example provided by various resources online when they explain how a neural network works.

# Reports
## XOR - Report
### Report ID: cfec5e21-6b28-4837-affd-66555f59fde2

### Configuration

| Conf              | Value          |
|-------------------|----------------|
| Iterations        | `10000`     |
| Learning Rate     | `1.0`          |
| Input Neurons     | `2`          |
| Hidden Neurons    | `3`          |
| Hidden Layers     | `1`          |
| Output Neurons    | `1`          |
| Input Act         | `None`       |
| Hidden Act        | `Sigmoid`    |
| Output Act        | `Sigmoid`    |
| **Extra Args**    | `-b 0.5`     |

### Learning Graph
- **MNS:** It's the Mobile Network Success rate. Range from 0 to 1. Higher is better.
- **MNE:** It's the Mobile Network Error rate. Lower is better.

![MNE Plot](xor.report.cfec5e21-6b28-4837-affd-66555f59fde2.plot-mne.png)

![MNS Plot](xor.report.cfec5e21-6b28-4837-affd-66555f59fde2.plot-mns.png)

### Memory Usage Graph
Memory usage was evaluated by _massif_ tool.

![Massif](xor.report.cfec5e21-6b28-4837-affd-66555f59fde2.memory.png)

### Execution Time

| Mode                 | Time ( seconds )   |
|----------------------|--------------------|
| Forward Propagation  | `2.207691120`  |
| Backward Propagation | `2.389012975` |
