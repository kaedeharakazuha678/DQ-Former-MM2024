from .audio_only import AudioOnly
from .text_only import TextOnly
from .vision_only import VisionOnly
from .mult import MultModel
from .MBT import MBT
from .single_stream import SingleStreamModel
from .dqformer import DQ_TAV

def load_model(args):
    model_mapping = {
        'audio_only': AudioOnly,
        'text_only': TextOnly,
        'vision_only': VisionOnly,
        'mult': MultModel,
        'mbt': MBT,
        'single_stream': SingleStreamModel,
        'dq_tav': DQ_TAV
    }
    assert args.model_name in model_mapping, f"Model {args.model_name} not found! \nModels supported: {model_mapping.keys()}"

    model = model_mapping.get(args.model_name, DQ_TAV)
    print(f">>> Using model {model.__name__}...")
    return model(args).to('cuda')
