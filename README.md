# timeline_extraction

This is a continuation project. The original project can be found here: https://github.com/savac/cdeo


## Instructions for installation and usage:

1.  Download UWTime following the instructions from: https://bitbucket.org/kentonl/uwtime-standalone. Then update the absolute path of the field 'uwtime_loc' in code/cdeo_config.json

2.  If required download Stanford CoreNLP and update the abdolute path of the field 'stanfordcorenlp_loc' in code/cdeo_config.json
3.  Update the field 'root_dir' in code/cdeo_config.json with the corresponding abdolute path.

4.  Update variable config_json_loc in code/cdeo_config.py witht the absolute path to code/cdeo_config.json

5.  Follow instructions from https://github.com/biplab-iitb/practNLPTools to download practNLPTools. Place practNLPTools in the folder code/practnlptools.
NOTE: the class Annotator is used from practNLPTools. The annotator object is being created using the following command: 'from practnlptools.tools import Annotator' . 

6. Install the python-levenshtein package. If you are using conda, run: conda install -c https://conda.anaconda.org/faircloth-lab python-levenshtein

*The directories data/tmp/ner and data/tmp/timex already contain the results of the UWTime and Stanford Core NLP processings so (1) and (2) are only required if planning to re-run the timex identification and parsing/coref.


Running

It needs to run from the code/ directory

cd code

(Optional) If no preprocessed files exist in data/tmp/ we need start the UWTime server

python -c "import cdeo; cdeo.startUWTimeServer()"

To train on the Apple corpus (corpus 0) and test on the Airbus (corpus 1), GM (corpus 2) and Stock Markets (corpus 3) using the structured perceptron algorithm run:

python -c "import cdeo; cdeo.run(test_corpus_list=[1,2,3], train_corpus_list=[0], link_model='structured_perceptron')"


To get the total micro scores place all predicted and gold timelines in two separate folders. Then # change the directory to cd code/evaluation_tool. Then run the following after adjusting the paths:

python evaluation_all.py ~/projects/cdeo/data/evaluation/combined/gold/ ~/projects/cdeo/data/evaluation/combined/results_structured/
