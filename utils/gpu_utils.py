import GPUtil
import os

def get_free_gpu():
    try:
        available_gpus = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.1, maxMemory=0.1)
        
        if len(available_gpus) > 0:
            return available_gpus[0]
        else:
            return None  
    except Exception as e:
        print("Error while GPU selection:", str(e))
        return None


selected_gpu = get_free_gpu()
if selected_gpu is not None:
    print("Selected GPU:", selected_gpu)
   
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
else:
    print("No available GPU.")