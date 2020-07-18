SETTINGS = {'in_channels': 2, 
		'num_layers': 5,
		'filter_width': 11,
		'channels': [16, 32, 64, 128, 256],
		'dropout':0.2,
		'rate':44100,
		'duration':0.5,
		'flat_dim':128}

REF_DIR = '/Users/collin/Dropbox/code/VoiceSampler/ChordGenProj/'
REF_CSV = REF_DIR + 'refset_nospace.csv'

ALT_DIR = '/Users/collin/Dropbox/code/VoiceSampler/chord_classification/SomeLikeItHot/'
ALT_CSV = ALT_DIR + 'cutset1_metadata.csv'
BATCH_SIZE = 32
