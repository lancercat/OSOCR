export PYTHONPATH=../../../../
export CUDA_DEVICE_ORDER=PCI_BUS_ID;export CUDA_VISIBLE_DEVICES=$1
python main.py 10 &> PLAYDAN10.log
python main.py 15 &> PLAYDAN15.log