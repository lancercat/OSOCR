export PYTHONPATH=../../../../
export CUDA_VISIBLE_DEVICES=$1
python test.py $2 &> TESTDAN.log
