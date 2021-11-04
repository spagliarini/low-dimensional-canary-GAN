# Data Analysis Birdsong
## Datasets
   The function datasets.py is meant to build the datasets I used to train the 
   WaveGAN. Please remember to tune the parameter before apply the function
   (using first the test function in features.py and then the function relative to the 
   dataset one needs). 
 
   There are several functions to build the dataset as you need. 
   Sometimes it is needed to combine more than one function to obtain a dataset. 
   
   Below the steps to follow to build a dataset based on single syllables:
   * Use the test function in features.py to tune the parameter for each syllable type.


   * If it is not already the case, prepare the songs/phrases for the analysis. That is, downsample them to
      16kHz. I used MATLAB to do this step.


   * Select from each phrase the single syllables.

	python datasets.py --option selection --data_dir DIRECTORY_NAME --syll_name SYLL_NAME --min_syl_dur X --min_silent_dur X --threshold X


   * Then create the dataset as you prefer: 
	
	- Create a dataset of uniform partitions of one second from a song, starting from a particular syllable. 

	    	python datasets.py --option uniform_partition --data_dir DIRECTORY_NAME

	- Create a dataset of single syllables + silence for a total duration of one second.
	    
	    A) After the selection, check the errors (semi-automatic code).

	    	python datasets.py --option control_dur --data_dir DIRECTORY_NAME

            B) Filter the syllable with high-pass filter of order 5 with 700 Hz as lower bound. Add silence to each syllables (this function writes the new wave over the old one so make 
               a copy of the directory for further analysis):

	    	python datasets.py --option uniform_dur_syllable --data_dir DIRECTORY_NAME
 
            C) Control the length (to fix an error coming from AudioSegment I had to make the recordings ~10 samples longer).
               This function writes the new wave over the old one.

            	python datasets.py --option control --data_dir DIRECTORY_NAME


