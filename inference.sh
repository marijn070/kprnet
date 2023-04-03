
date=`date +"%m_%d_%Y_%H_%M"`

if [ "$1" == "ittik" ]; then
	echo running inference on the ittik dataset
	python ~/kprnet/run_inference.py --checkpoint-path ~/pretrained_weights/kpr_trained.pth --output-path ~/outputs/inference_$1_${date}_$3 --semantic-kitti-dir ~/dataset_kitti/sequences/ --split $2 --semantic-ittik-dir ~/dataset_ittik/sequences $3
else
	echo running inference on the kitti dataset
	python ~/kprnet/run_inference.py --checkpoint-path ~/pretrained_weights/kpr_trained.pth --output-path ~/outputs/inference_$1_${date} --semantic-kitti-dir ~/dataset_kitti/sequences/ --split $2 $3
fi

