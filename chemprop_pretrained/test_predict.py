from predict import ChempropPredict 

def main(): 

    smiles = [
        "N#C[C@H]1N[C@]21[C@H]1C[C@@H]2C1",
        "Oc1nonc1OC=O,0.2271",
        "C[C@]12CC[C@H]1CN1C[C@@H]21",
        "CN=C1OC(=O)C[C@H]1N",
        "C[C@@H](CO)[C@@H](CO)C#N,0.3048",
        "CC1(C)CC(=O)O[C]1[NH]",
        "C[C@H]1N[C@]2(CN[C@@H]12)C#N",
    ]
    smiles = [[i] for i in smiles]


    model = ChempropPredict(checkpoint_path='../models/model-bs.pt')
    y_pred = model.predict(smiles)
    print(y_pred)

if __name__ == '__main__':
    main()