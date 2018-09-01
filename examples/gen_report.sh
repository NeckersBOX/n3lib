#!/bin/sh

FLAGS=""
EX_NAME=""
ITERS=1
RUN_EXTRA=""
REPORT_ID="$(uuidgen)"

usage() {
    echo "./gen_report.sh [-e|--example name] [-i|--iterations number]"
    echo " optional: [-c|--cflags \"flags to gcc\"]"
    echo "           [-h|--help]"
    echo "           [-r|--run-extra \"flags to exampls\"]"
}

while [ "$1" != "" ]; do
    case $1 in
        -c | --cflags)        shift;
                              FLAGS=$1
                              ;;
        -e | --example)       shift;
                              EX_NAME=$1
                              ;;
        -i | --iterations)    shift;
                              ITERS=$1
                              ;;
        -r | --run-extra )    shift;
                              RUN_EXTRA=$1
                              ;;
        -h | --help )         usage
                              exit 1
                              ;;
        * )                   usage
                              exit -1
    esac
    shift
done

echo "First run to build CSV.."
gcc $EX_NAME/$EX_NAME.c -o $EX_NAME/gen_report -ln3l -lm -DN3L_ENABLE_STATS $FLAGS
./$EX_NAME/gen_report -i $ITERS -l -p -t $EX_NAME/$EX_NAME.report.$REPORT_ID.csv -o $EX_NAME/$EX_NAME.report.$REPORT_ID.n3l $RUN_EXTRA

echo "Second run to build massif heap stats"
gcc $EX_NAME/$EX_NAME.c -o $EX_NAME/gen_report -ln3l -lm $FLAGS
valgrind --tool=massif --massif-out-file=$EX_NAME/$EX_NAME.report.$REPORT_ID.massif ./$EX_NAME/gen_report -i $ITERS -p -l $RUN_EXTRA

echo "Third run to evaluate execution time in forward mode"
START=$(date +%s.%N)
./$EX_NAME/gen_report -i $ITERS $RUN_EXTRA -m
END=$(date +%s.%N)
TIME_FORWARD="$(echo "$END - $START" | bc)"

echo "Fourth run to evaluate execution time in backward mode"
START=$(date +%s.%N)
./$EX_NAME/gen_report -i $ITERS $RUN_EXTRA -m -l
END=$(date +%s.%N)
TIME_BACKWARD="$(echo "$END - $START" | bc)"

echo "Removing executable"
rm $EX_NAME/gen_report

echo "Plotting graphs.."
gnuplot -p << _EOF_
set term png
set output "$EX_NAME/$EX_NAME.report.$REPORT_ID.plot-mne.png"
set datafile separator ","
set xlabel "Iterations"
set ylabel "Error rate"
set style data line
plot "$EX_NAME/$EX_NAME.report.$REPORT_ID.csv" using 1:4 title "Mobile Network Error" lt 1 lc 3 lw 2
_EOF_

gnuplot -p << _EOF_
set term png
set output "$EX_NAME/$EX_NAME.report.$REPORT_ID.plot-mns.png"
set datafile separator ","
set xlabel "Iterations"
set ylabel "Success rate [0, 1]"
set style data line
plot "$EX_NAME/$EX_NAME.report.$REPORT_ID.csv" using 1:5 title "Mobile Network Success" lt 1 lc 4 lw 2
_EOF_

echo "Generating raw report..."

cat <<-_EOF_ >> $EX_NAME/README.md
## $(echo $EX_NAME | tr '[:lower:]' '[:upper:]') - Report
### Report ID: $REPORT_ID

### Configuration

| Conf              | Value          |
|-------------------|----------------|
| Iterations        | \`$ITERS\`     |
| Learning Rate     | \`0\`          |
| Input Neurons     | \`0\`          |
| Hidden Layers     | \`0\`          |
| Hidden Neurons    | \`0\`          |
| Output Neurons    | \`0\`          |
| Input Act         | \`None\`       |
| Hidden Act        | \`Sigmoid\`    |
| Output Act        | \`Sigmoid\`    |
| **Extra Args**    | \`$RUN_EXTRA\` |

### Learning Graph
- **MNS:** It's the Mobile Network Success rate. Range from 0 to 1. Higher is better.
- **MNE:** It's the Mobile Network Error rate. Lower is better.

![MNE Plot]($EX_NAME.report.$REPORT_ID.plot-mne.png)

![MNS Plot]($EX_NAME.report.$REPORT_ID.plot-mns.png)

### Memory Usage Graph
Memory usage was evaluated by _massif_ tool.

![Massif]($EX_NAME.report.$REPORT_ID.memory.png)

### Execution Time

| Mode                 | Time ( seconds )   |
|----------------------|--------------------|
| Forward Propagation  | \`$TIME_FORWARD\`  |
| Backward Propagation | \`$TIME_BACKWARD\` |

_EOF_
