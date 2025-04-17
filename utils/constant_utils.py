#region Sentiment
# **********************************************************************************************************************
#                                                        Sentiment  Constants
# **********************************************************************************************************************
SENTIMENT_LABELS = [0,1,2]

SENTIMENT_NAMES = ['negative', 'positive', 'neutral']

SENTIMENT_MAPPING = {
            "negative": 0,
            "neutral": 1,
            "positive": 2   
        }
#endregion

#region meld
# **********************************************************************************************************************
#                                                        MELD  Constants         
# **********************************************************************************************************************

MELD_EMOTION_LABELS = [0,1,2,3,4,5,6]

MELD_EMOTION_NAMES =['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

MELD_TEST_CLASS_WEIGHTS = {
    0: 1256/2610,
    1: 402/2610,
    2: 208/2610,
    3: 345/2610,
    4: 50/2610,
    5: 68/2610,     
    6: 281/2610,
}   

MELD_DEV_CLASS_WEIGHTS = {
    0: 469 / 1108,  # neutral
    1: 163 / 1108,  # joy
    2: 111 / 1108,  # sadness
    3: 153 / 1108,  # anger
    4: 40 / 1108,   # fear
    5: 22 / 1108,   # disgust
    6: 150 / 1108   # surprise
}

MELD_EMOTION_MAPPING = {
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
}


#endregion

#region iemocap
# **********************************************************************************************************************
#                                                        IEMOCAP  Constants
# **********************************************************************************************************************
IEMOCAP_EMOTION_LABELS = [0,1,2,3,4,5,6]

IEMOCAP_EMOTION_LABELS_4_CLASS = [0,1,2,3]

IEMOCAP_EMOTION_LABELS_6_CLASS = [0,1,2,3,4,5]

IEMOCAP_EMOTION_NAMES =['neutral', 'happy', 'sad', 'angry', 'excited', 'frustrated']

IEMOCAP_EMOTION_NAMES_4_CLASS =['neutral', 'exc', 'angry', 'sad']

IEMOCAP_EMOTION_NAMES_6_CLASS =[ 'neutral', 'happy', 'sad', 'angry', 'excited', 'frustrated']

IEMOCAP_EMOTION_MAPPING = {
    'neu': 0,
    'hap': 1,
    'ang': 2,
    'sad': 3,
    'exc': 4,
    'fru': 5,
    'unknown': 6
}

IEMOCAP_EMOTION_MAPPING_4_CLASS={
    'neu': 0,
    'exc': 1,
    'ang': 2,
    'sad': 3,
}

IEMOCAP_EMOTION_MAPPING_6_CLASS={
    'neu': 0,
    'hap': 1,
    'ang': 2,
    'sad': 3,
    'exc': 4,
    'fru': 5,
}

#endregion