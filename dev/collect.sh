DATA_DIR=road_data
NUM_REPEAT=8

mkdir -p $DATA_DIR

python3 collect.py --track cornfield_crossing --output_dir $DATA_DIR --num_repeat $NUM_REPEAT
python3 collect.py --track hacienda --output_dir $DATA_DIR --num_repeat $NUM_REPEAT
python3 collect.py --track snowmountain --output_dir $DATA_DIR --num_repeat $NUM_REPEAT
python3 collect.py --track lighthouse --output_dir $DATA_DIR --num_repeat $NUM_REPEAT
python3 collect.py --track zengarden --output_dir $DATA_DIR --num_repeat $NUM_REPEAT

mkdir -p $DATA_DIR/train
mkdir -p $DATA_DIR/val
mkdir -p $DATA_DIR/test

mv $DATA_DIR/*_00 $DATA_DIR/train
mv $DATA_DIR/*_01 $DATA_DIR/train
mv $DATA_DIR/*_02 $DATA_DIR/train
mv $DATA_DIR/*_03 $DATA_DIR/train

mv $DATA_DIR/*_04 $DATA_DIR/val
mv $DATA_DIR/*_05 $DATA_DIR/val

mv $DATA_DIR/*_06 $DATA_DIR/test
mv $DATA_DIR/*_07 $DATA_DIR/test
