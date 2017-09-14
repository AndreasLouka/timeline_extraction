# timeline_extraction

This is a continuation project. The original project can be found here: https://github.com/savac/cdeo


## Instructions for installation and usage:

1.  Download UWTime following the instructions from: https://bitbucket.org/kentonl/uwtime-standalone. Then update the absolute path of the field 'uwtime_loc' in code/cdeo_config.json
2.  If required download Stanford CoreNLP and update the abdolute path of the field 'stanfordcorenlp_loc' in code/cdeo_config.json
3.  Update the field 'root_dir' in code/cdeo_config.json with the corresponding abdolute path.
4.  Update variable config_json_loc in code/cdeo_config.py witht the absolute path to code/cdeo_config.json
5.  Follow instructions from https://github.com/biplab-iitb/practNLPTools to download practNLPTools. Place practNLPTools in the folder code/practnlptools. NOTE: the class Annotator is used from practNLPTools. The annotator object is being created using the following command: from practnlptools.tools import Annotator. 


