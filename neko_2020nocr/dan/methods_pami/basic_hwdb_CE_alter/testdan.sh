export PYTHONPATH=../../../../
export CUDA_VISIBLE_DEVICES=$1
python test.py 20 &> TESTDAN20.log
python test.py 15 &> TESTDAN15.log
python test.py 10 &> TESTDAN10.log
python test.py 5 &> TESTDAN5.log
