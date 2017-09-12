import json

config_json_loc = '/Users/andreaslouka/Desktop/thesis/cdeo/code/cdeo_config.json'

def setDefaults():
    defaults = {"restrict_entity_linking_to_sentence_flag": 1, 
    "stanfordcorenlp_loc": "/Users/andreaslouka/Desktop/thesis/stanford-corenlp-full-2017-06-09", 
    "n_epochs_entity": 15, 
    "n_epochs_timex": 15, 
    "levenshtein_threshold": 0.4, 
    "event_entity_link_threshold": 0.1, 
    "root_dir": "/Users/andreaslouka/Desktop/thesis/cdeo/", 
    "uwtime_loc": "/Users/andreaslouka/Desktop/thesis/uwtime-standalone/target/uwtime-standalone-1.0.1.jar",
    'track': 'A'}

    f = open(config_json_loc, 'w')
    json.dump(defaults, f)
    f.close()

def setConfig(field, val):
    f = open(config_json_loc, 'r')
    defaults = json.load(f)
    defaults[field] = val
    f.close()
    f = open(config_json_loc, 'w')
    json.dump(defaults, f)
    f.close()
        
def getConfig(field):
    f = open(config_json_loc, 'r')
    defaults = json.load(f)
    res = defaults[field]
    f.close()
    return res 
