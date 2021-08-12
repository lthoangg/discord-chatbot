# Discord chatbot

## Instruction
* Create *.env* file with these keys:
```env
TOKEN=
weather_api=
```

* Install packages
```bash
pip install -r requirements.txt
```

* Train model with *intents.json* data
```bash
python train.py
```

* Connect to discord server & run chatbot
```bash
python main.py
```