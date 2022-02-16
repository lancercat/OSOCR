export PYTHONPATH=../../../../
export CUDA_VISIBLE_DEVICES=$1
python main.py 5 &> PLAYDAN5.log
python main.py 20 &> PLAYDAN20.log
