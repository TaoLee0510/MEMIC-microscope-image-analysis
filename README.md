
Requirements
	•	Fiji (with the ability to run macros)
	•	cellpose-sam
	•	Tested on arm64 Macs
Usage

Directory Structure
	•	input_root: Folder containing raw image files for each well.
Example: "/images/MEMIC"
	•	output_root: Folder to store processed image outputs.
Example: "/images/MEMIC_output"



Step-by-Step Instructions
	1.	Install Dependencies
        pip install -r requirements.txt


	2.	Preprocess Raw Images
        Run the preprocessing script:

        python run_preprocess_memic.py <input_root> <output_root>

        •	Expected raw image file structure:
            <input_root>/<well_name>/<time_point_name>/<image_name>.tiff
    	•	The script will automatically identify well names and time point directories.


    3.	Run Fiji for EDoF Processing
	    •	Install the required plug-in in Fiji.
	    •	Open Fiji and run the macro file: EDoF_Macro.ijm
	    •	This step will generate processed images under:
            <output_root>/EDoF/<well_name>/<time_point_name>/

	4.	Run Main Analysis Script
        Run the full pipeline on the EDoF-processed images:

        python run_memic_image_analysis.py <output_root>/EDoF <output_root>/full_pipeline


