	====================================================================================================================================
	1. One might need to install some libraries
	
	2. Download Imagenet validation set and place in one folder
	   http://image-net.org/download
	   put the path of dataset folder in main_alexnet file
	
	3. Download the weights file of pre-trained AlexNet from 
	   https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
	
	
	4. Run the below command for classification
	   /> python main_alexnet.py align_3_4 align_4_3
	   
	   Two command line arguments for quantization schemes. Other schemes can also be applied
	   First quantization scheme for initial layer parameters and second quantization scheme for further layer parameters
	   
	=============================================================================================================================
	For questions/suggestions please email Siddharth Gupta (ms1804101006@iiti.ac.in) and Salim Ullah (salim.ullah@tu-dresden.de)
	==============================================================================================================================