 An extraction of _Mixture of Tokens_ model from `llm-random` repository.

 Currently, it uses a small part of `wikipedia` dataset.

 To create a dataset of size `SIZE`, use:
 ```
 python dataset.py dataset.dat --size [SIZE]
```

and then train the vanilla Transformer model via:
```
python train.py vanilla dataset.dat --path vanilla.model --epochs 20
```

or _Mixture of Tokens_ by:
```
python train.py mot dataset.dat --path mot.model --epochs 20
```
