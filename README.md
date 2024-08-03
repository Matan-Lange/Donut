# HeDonut


## HE - SynthDog
HE-Synthdog is a modified version of the original Synthdog framework, 
specifically tailored to generate syntactic documents in Hebrew. 
This project includes minor changes to the original Synthdog to support the Hebrew language, enabling the creation of documents with Hebrew text.<br>

dependencies:<br>
python-bidi==0.4.2<br>
numpy==1.26.0<br>
pillow==9.4.0<br>
mlflow<br>
dicttoxml<br>
zss



Usage:
1. ```pip install synthtiger```
2. Add your custom Hebrew corpus to the following path: synthdog/resources/corpus
3. Change corpus path in config_he.yaml
4. ```cd synthdog```
5. Run ```synthtiger -o ./outputs/SynthDoG_he -c 50 -w 4 -v template.py SynthDoG config_he.yaml```

For more detailed usage instructions and examples, please refer to the original synthtiger paper+code
https://github.com/clovaai/synthtiger/blob/master/README.md

## Pre-Training HeDonut 



## FineTuning document parsing 
<b>snythtic invoice generation</b><br>
run ```python finetune/generate_invoices.py --total_invoices 1000 --output_directory path/to/dir```






