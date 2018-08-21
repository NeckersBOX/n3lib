# N3L Example - IRIS

## Description
A common example of use neural network to classificate the inputs to a recognized pattern.
Data set and description reference: https://archive.ics.uci.edu/ml/datasets/Iris

It's designed with the idea that each output neuron give the probabily about each of the three possibily classification.

| Name | Case ID |
|------|---------|
| Iris-virginica | 0 |
| Iris-setosa | 1 |
| Iris-versicolor | 2 |

# Reports

## IRIS - Report
### Report ID: 59d477f2-bbe7-4b52-b5e0-d04844d82073

### Configuration

| Conf              | Value          |
|-------------------|----------------|
| Iterations        | `50000`     |
| Learning Rate     | `0.01`      |
| Input Neurons     | `4`          |
| Hidden Neurons    | `3`          |
| Hidden Layers     | `2`          |
| Output Neurons    | `3`          |
| Input Act         | `None`       |
| Hidden Act        | `Swish`    |
| Output Act        | `Swish`    |
| **Extra Args**    | `` |

### Learning Graph
- **MNS:** It's the Mobile Network Success rate. Range from 0 to 1. Higher is better.
- **MNE:** It's the Mobile Network Error rate. Lower is better.

![MNE Plot](iris.report.59d477f2-bbe7-4b52-b5e0-d04844d82073.plot-mne.png)

![MNS Plot](iris.report.59d477f2-bbe7-4b52-b5e0-d04844d82073.plot-mns.png)

### Memory Usage Graph
Memory usage was evaluated by _massif_ tool.

![Massif](iris.report.59d477f2-bbe7-4b52-b5e0-d04844d82073.memory.png)

### Execution Time

| Mode                 | Time ( seconds )   |
|----------------------|--------------------|
| Forward Propagation  | `27.208896672`  |
| Backward Propagation | `87.079262506` |
## IRIS - Report
### Report ID: aae271a9-2bbb-4b9f-931c-decdc4c576c5

### Configuration

| Conf              | Value          |
|-------------------|----------------|
| Iterations        | `50000`     |
| Learning Rate     | `0.05`          |
| Input Neurons     | `4`          |
| Hidden Neurons    | `3`          |
| Hidden Layers     | `1`          |
| Output Neurons    | `3`          |
| Input Act         | `None`       |
| Hidden Act        | `Sigmoid`    |
| Output Act        | `Sigmoid`    |
| **Extra Args**    | `` |

### Learning Graph
- **MNS:** It's the Mobile Network Success rate. Range from 0 to 1. Higher is better.
- **MNE:** It's the Mobile Network Error rate. Lower is better.

![MNE Plot](iris.report.aae271a9-2bbb-4b9f-931c-decdc4c576c5.plot-mne.png)

![MNS Plot](iris.report.aae271a9-2bbb-4b9f-931c-decdc4c576c5.plot-mns.png)

### Memory Usage Graph
Memory usage was evaluated by _massif_ tool.

![Massif](iris.report.aae271a9-2bbb-4b9f-931c-decdc4c576c5.memory.png)

### Execution Time

| Mode                 | Time ( seconds )   |
|----------------------|--------------------|
| Forward Propagation  | `15.152438212`  |
| Backward Propagation | `31.286549418` |
