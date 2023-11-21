export PYTHONPATH=../../../../
export CUDA_DEVICE_ORDER=PCI_BUS_ID;export CUDA_VISIBLE_DEVICES=$1

python main.py 5 &> PLAYDAN5.log
python main.py 15 &> PLAYDAN15.log
python main.py 20 &> PLAYDAN20.log
python main.py 10 &> PLAYDAN10.log