	====================================================================================================================================
	Quantization of Resnet-18 network using ALigN technique
	
	Reference network is taken from below repository  
	https://github.com/dalgu90/resnet-18-tensorflow
	
	1. One might need to install some libraries
           see the imports in eval.py file
	
	2. Download the ResNet-18 torch checkpoint
		wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
		# Convert into tensorflow checkpoint
		python extract_torch_t7.py
	
	3. Modify `train_scratch.sh`(training from scratch) or `train.sh`(finetune pretrained weights) to have valid values of following arguments
		- `train_dataset`, `train_image_root`, `val_dataset`, `val_image_root`: Path to the list file of train/val dataset and to the root
		- `num_gpus` and corresponding IDs of GPUs(`CUDA_VISIBLE_DEVICES` at the first line)

  
	4. Evaluate the trained model
        /> python eval.py 
	  
	  Other schemes quantization can also be applied by modifying the utils.py file.
	
	
	=============================================================================================================================
	For questions/suggestions please email Siddharth Gupta (ms1804101006@iiti.ac.in) and Salim Ullah (salim.ullah@tu-dresden.de)
	==============================================================================================================================