export PYTHONPATH=../../../../
export CUDA_VISIBLE_DEVICES=$1
python test.py &> TESTDAN.log
