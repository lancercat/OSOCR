export PYTHONPATH=../../../../
export CUDA_DEVICE_ORDER=PCI_BUS_ID;export CUDA_VISIBLE_DEVICES=$1
python main.py &> PLAYDAN.log
