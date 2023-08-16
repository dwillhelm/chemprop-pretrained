from typing import Union, List, Any 
from os import PathLike

import chemprop

class ChempropPredict: 

    def __init__(self, checkpoint_path:Union[str, PathLike]): 
        self.checkpoint_path = checkpoint_path

    def predict(self, smiles:List[str]): 

        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_path', '../models/model-bs.pt'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        model_objects = chemprop.train.load_model(args=args)

        preds = chemprop.train.make_predictions(
            args=args,
            smiles=smiles, 
            model_objects=model_objects)
        return preds