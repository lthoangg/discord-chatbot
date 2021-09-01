# Discord chatbot

## Instruction
* Create *.env* file with these keys:
```env
TOKEN=#secret key from discord
weather_api=#secret keyfrom OpenWeatherAPI
```

* Creat environment and activate (Conda)
```
conda create -n discord-chatbot python==3.8.11
conda activate discord-chatbot
```

* Install packages 
```bash
pip install -r requirements.txt
```

* Train model with *data/intents.json* data
```bash
python trainer.py
```

* Test bot on local before public to discord
```bash
python test.py
```

* Connect to discord server & run chatbot
```bash
python main.py
```


## References
I'm using the same methodology from [this repo](https://github.com/python-engineer/pytorch-chatbot) but using **pytorch_lightning**
