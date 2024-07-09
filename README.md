# HE-Dount


## HE - SynthDog
HE-Synthdag is a modified version of the original Synthdog framework, 
specifically tailored to generate syntactic documents in Hebrew. 
This project includes minor changes to the original Synthdog to support the Hebrew language, enabling the creation of documents with Hebrew text.<br>

Usage:
1. ```pip install synthtiger```
2. Add your custom Hebrew corpus to the following path: synthdog/resources/corpus
3. Change corpus path in config_he.yaml
4. ```cd synthdog```
5. Run ```synthtiger -o ./outputs/SynthDoG_he -c 50 -w 4 -v template.py SynthDoG config_he.yaml```

For more detailed usage instructions and examples, please refer to the original synthtiger paper+code
https://github.com/clovaai/synthtiger/blob/master/README.md

