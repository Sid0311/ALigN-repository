	====================================================================================================================================
	Quantization of FCN-8 network for image segmentation using ALigN technique
	
	The code is based on FCN implementation by Sarath Shekkizhar with MIT license and below reposiroty  
	https://github.com/0merjavaid/Retina-segmentation-with-FCN
	
	1. One might need to install some libraries
        see the imports in eval.py file
	
	2. Download a pre-trained vgg16 net and put in the /Model_Zoo subfolder 
		A pre-trained vgg16 net can be download from here
		https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] 
		or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]
	
	3. Download the dataset for segmentation and place in the folder Data_Zoo

  
	4. Evaluate the mean intersection over union
        /> python Evaluate_Net_IOU.py
	  
	  By defaul log_2_lead quantization  is set for evaluation.
	  Other ALigN schemes can be applied by modifying BuildNetVgg16.py file
	
	
	=============================================================================================================================
	For questions/suggestions please email Siddharth Gupta (ms1804101006@iiti.ac.in) and Salim Ullah (salim.ullah@tu-dresden.de)
	==============================================================================================================================