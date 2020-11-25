	====================================================================================================================================
	1. One might need to install some libraries
           see the imports in main_vgg16
	
	2. Download Imagenet validation set and place in one folder
	   http://image-net.org/download
	   put the path of dataset folder in main_vgg16 file
	
	3. Download the weights file of pre-trained VGG-16 from 
	   http://www.cs.toronto.edu/~frossard/post/vgg16/
	
	
	4. Run the below command for classification
	   /> python main_vgg16.py align_3_4 align_4_3 align_4_3
	   
	   three command line arguments for different layer of network for quantization. Other schemes can also be applied
	
	
	=============================================================================================================================
	For questions/suggestions please email Siddharth Gupta (ms1804101006@iiti.ac.in) and Salim Ullah (salim.ullah@tu-dresden.de)
	==============================================================================================================================