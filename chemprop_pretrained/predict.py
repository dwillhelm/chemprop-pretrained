from pathlib import Path
from typing import Union, List, Any 
from os import PathLike

import chemprop

class ChempropPredict: 
    """Chemprop inference class."""

    def __init__(self, checkpoint_path:Union[str, PathLike]):
        """Initialize.  

        Parameters
        ----------
        checkpoint_path : Union[str, PathLike]
            Path to a persisted Chemprop model/checkpoint (.pt) file. 
        """
        self.checkpoint_path = Path(checkpoint_path)

    def predict(self, smiles:List[str]): 
        """Forward prediction. 

        Parameters
        ----------
        smiles : List[str]
            A list of SMILES strings
        """
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_path', str(self.checkpoint_path)
        ]
        args = chemprop.args.PredictArgs().parse_args(arguments)

        model_objects = chemprop.train.load_model(args=args)
        y_pred = chemprop.train.make_predictions(
            args=args,
            smiles=smiles, 
            model_objects=model_objects)
        return y_pred