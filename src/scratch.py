import pandas as pd 
import chemprop


def main(): 

    example_smiles = [
        "N#C[C@H]1N[C@]21[C@H]1C[C@@H]2C1",
        "Oc1nonc1OC=O,0.2271",
        "C[C@]12CC[C@H]1CN1C[C@@H]21",
        "CN=C1OC(=O)C[C@H]1N",
        "C[C@@H](CO)[C@@H](CO)C#N,0.3048",
        "CC1(C)CC(=O)O[C]1[NH]",
        "C[C@H]1N[C@]2(CN[C@@H]12)C#N",
    ]
    example_smiles = [[i] for i in example_smiles]

    # example_smiles = [['CCC'], ['CCCC'], ['OCC']]

    arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--checkpoint_path', '../models/model-bs.pt'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    model_objects = chemprop.train.load_model(args=args)


    print(model_objects)

    preds = chemprop.train.make_predictions(
        args=args,
        smiles=example_smiles, 
        model_objects=model_objects)
    print(preds)

    # preds = chemprop.train.make_predictions(
    #     args=args,
    #     smiles=example_smiles,
    #     model_objects=model_objects)
    # print(preds)

if __name__ == '__main__':
    main() 
