CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=2333 ../../tools/train.py --cfg config.yaml &
python -u ../../tools/test.py --snapshot model.pth --dataset VOT2019 --config config.yaml
python ../../tools/eval.py --tracker_path ./results/ --dataset VOT2019 --num 2
