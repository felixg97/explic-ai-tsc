"""
constants.py

This file got provdied by: https://github.com/hfawaz/dl-4-tsc/blob/master/utils/constants.py
"""

UNIVARIATE_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
                            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                            'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
                            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']

UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                                'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                                'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                                'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                                'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                                'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                                'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                                'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                                'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                                'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                                'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                                'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                                'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                                'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                                'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                                'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                                'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                                'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                                'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                                'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                                'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                                'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                                'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                                'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']


ITERATIONS = 1 # TODO: (2, 3) # nb of random runs for random initializations

ARCHIVE_NAMES = ['UCRArchive_2018']

dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}

dataset_types = {'ElectricDevices': 'DEVICE', 'FordB': 'SENSOR',
                'FordA': 'SENSOR', 'NonInvasiveFatalECG_Thorax2': 'ECG',
                'NonInvasiveFatalECG_Thorax1': 'ECG', 'PhalangesOutlinesCorrect': 'IMAGE',
                'HandOutlines': 'IMAGE', 'StarLightCurves': 'SENSOR',
                'wafer': 'SENSOR', 'Two_Patterns': 'SIMULATED',
                'UWaveGestureLibraryAll': 'MOTION', 'uWaveGestureLibrary_Z': 'MOTION',
                'uWaveGestureLibrary_Y': 'MOTION', 'uWaveGestureLibrary_X': 'MOTION',
                'Strawberry': 'SPECTRO', 'ShapesAll': 'IMAGE',
                'ProximalPhalanxOutlineCorrect': 'IMAGE', 'MiddlePhalanxOutlineCorrect': 'IMAGE',
                'DistalPhalanxOutlineCorrect': 'IMAGE', 'FaceAll': 'IMAGE',
                'ECG5000': 'ECG', 'SwedishLeaf': 'IMAGE', 'ChlorineConcentration': 'SIMULATED',
                '50words': 'IMAGE', 'ProximalPhalanxTW': 'IMAGE', 'ProximalPhalanxOutlineAgeGroup': 'IMAGE',
                'MiddlePhalanxOutlineAgeGroup': 'IMAGE', 'DistalPhalanxTW': 'IMAGE',
                'DistalPhalanxOutlineAgeGroup': 'IMAGE', 'MiddlePhalanxTW': 'IMAGE',
                'Cricket_Z': 'MOTION', 'Cricket_Y': 'MOTION',
                'Cricket_X': 'MOTION', 'Adiac': 'IMAGE',
                'MedicalImages': 'IMAGE', 'SmallKitchenAppliances': 'DEVICE',
                'ScreenType': 'DEVICE', 'RefrigerationDevices': 'DEVICE',
                'LargeKitchenAppliances': 'DEVICE', 'Earthquakes': 'SENSOR',
                'yoga': 'IMAGE', 'synthetic_control': 'SIMULATED',
                'WordsSynonyms': 'IMAGE', 'Computers': 'DEVICE',
                'InsectWingbeatSound': 'SENSOR', 'Phoneme': 'SENSOR',
                'OSULeaf': 'IMAGE', 'FacesUCR': 'IMAGE',
                'WormsTwoClass': 'MOTION', 'Worms': 'MOTION',
                'FISH': 'IMAGE', 'Haptics': 'MOTION',
                'Epilepsy': 'HAR', 'Ham': 'SPECTRO',
                'Plane': 'SENSOR', 'InlineSkate': 'MOTION',
                'Trace': 'SENSOR', 'ECG200': 'ECG',
                'Lighting7': 'SENSOR', 'ItalyPowerDemand': 'SENSOR',
                'Herring': 'IMAGE', 'Lighting2': 'SENSOR',
                'Car': 'SENSOR', 'Meat': 'SPECTRO',
                'Wine': 'SPECTRO', 'MALLAT': 'SIMULATED',
                'Gun_Point': 'MOTION', 'CinC_ECG_torso': 'ECG',
                'ToeSegmentation1': 'MOTION', 'ToeSegmentation2': 'MOTION',
                'ArrowHead': 'IMAGE', 'OliveOil': 'SPECTRO',
                'Beef': 'SPECTRO', 'CBF': 'SIMULATED',
                'Coffee': 'SPECTRO', 'SonyAIBORobotSurfaceII': 'SENSOR',
                'Symbols': 'IMAGE', 'FaceFour': 'IMAGE',
                'ECGFiveDays': 'ECG', 'TwoLeadECG': 'ECG',
                'BirdChicken': 'IMAGE', 'BeetleFly': 'IMAGE',
                'ShapeletSim': 'SIMULATED', 'MoteStrain': 'SENSOR',
                'SonyAIBORobotSurface': 'SENSOR', 'DiatomSizeReduction': 'IMAGE'}

themes_colors = {'IMAGE': 'red', 'SENSOR': 'blue', 'ECG': 'green',
                'SIMULATED': 'yellow', 'SPECTRO': 'orange',
                'MOTION': 'purple', 'DEVICE': 'gray'}


################################### ArgParse ###################################

HELP_INFO = """
commands: help, run_complete, run_classifier, run_explanations, run_evaluations

argugments:
-data, -d {dataset: default=all, optional}
-classifier, -c {classifier: default=all, optional}
-iterations, -i {iterations: default=?, optional}
-explanation, -ex {explanation: default=all, optional}
-evaluation, -ev {evaluation: default=all, optional}
-output_path, -o {output_path: string, optional}

configurations:
--verbose
--load
--rebuild
--generate_plots
"""

COMMANDS = ['help', 'run_complete', 'run_classifier', 'run_explanations', 'run_evaluations']

ARGUMENTS = [
    '-data', '-d', '-classifier', '-c', '-iterations', '-i', '-explanation', 
    '-ex', '-evaluation', '-ev', '-output_path', '-o'
]

PARAMS = []

CONFIGS = ['--verbose']

################################### Data sets ##################################
UNIVARIATE_DATASET_NAMES = [
    '50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
    'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
    'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
    'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
    'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
    'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
    'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
    'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
    'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
    'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
    'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
    'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
    'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga'
]

DATASETS_NAMES = [
    'BeetleFly', # wird von rise immer geskipped?
    'Earthquakes',
    'OSULeaf', # TODO: IT LAST
    # 'ECG5000', # TODO: 2  (rise bei ecg5000 batch_size 1500 gestoppt -> BeetleFly offen, wurde als lime_interpolation getagged)
    # 'ElectricDevices' # TODO: 2
]

# TODO: (heute) InceptionTime lime eucledian mean osuleaf & rise alle datens??tze

## weitere optionen: ScreenType, UWaveGestureLibraryAll, Phoneme, Lightning2, CricketY, ElectricDevices, Adiac

################################## Classifiers #################################
CLASSIFIERS = [
    'MLP', # TODO: (1)
    'ResNet', # TODO: (2)
    # 'FCN', # TODO: i guess (3.2)
    'InceptionTime' # TODO: i guess (3.1) # TODO: IT LAST
    #'HIVE-COTE', # TODO: keine wahnung wo ich das herbekomme bisher
]


################################## Explanations ################################
EXPLANATIONS = [
    'LIME',
    'RISE',
    # 'Anchor',
    'Occlusion',
    # 'MeaningfulPerturbation',
    # 'ExtremalPerturbation'
]


################################## Evaluations #################################
EVALUATIONS = [
    # 'SanityCheck', 
    'PerturbationAnalysis'
]

############ Data sets Jakob used ############
DATA_SETS_JAKOB = [
'Adiac',
'Beef',
'BeetleFly',
'BirdChicken',
'ChlorineConc',
'Computers',
'CricketY',
'Crop',
'Earthquakes',
'ECG200',
'ECG5000',
'ElectricDevices',
'Haptics',
'Herring',
'InlineSkate',
'Lightning2',
'Lightning7',
'MidPhalOutAgGrp',
'MidPhalTW',
'MoteStrain',
'OliveOil',
'OSULeaf',
'Phoneme',
'ScreenType',
'ShapesAll',
'SmallKitchenApp',
'UWaveGestLibAll',
'Wine',
'WordSynonyms',
'Worms'
]
