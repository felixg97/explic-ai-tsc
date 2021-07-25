"""
resources.py - Resources I use
"""

######################## Data sets I use ########################
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

MY_DATA_SETS = [
    'BeefFly', 
    'Earthquakes',
    'ECG5000',
    'ElectricDevices',
    'OSULeaf',
]

## weitere optionen: ScreenType, UWaveGestureLibraryAll, Phoneme, Lightning2, CricketY, ElectricDevices, Adiac

######################## Classifier I use ########################
MY_CLASSIFIERS = [
    'MLP', # easy TODO: (1)
    'ResNet', # easy TODO: (2)
    'FCN', # easy TODO: i guess (3.2)
    'InceptionTime' # easy TODO: i guess (3.1)
    'HIVE-COTE', # TODO: keine wahnung wo ich das herbekomme bisher
]


######################## Explanation I use ########################
MY_EXPLANATIONS = [
    '',
]


######################## Evaluations I use ########################
MY_EVALUATIONS = [
    'Sanity check', 
    'Perturbation Analysis',
]