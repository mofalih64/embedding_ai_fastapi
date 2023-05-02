# Fastapi Embedding AI Service

## This repository contains a simple API that provides a service for querying and getting relevant data using Embedding AI. It is built using the FastAPI web framework and the OpenAI library.

### I made this service to make the AI create poems like mine, after i add them in a csv file

### I used in this repository

1. Openai
1. Fastapi
   3.Python
1. csv file

## In order to run the code locally you should do :

1. clone the repo into yours machine
2. create your virtual environment to install dependencies like this command in git bash

```
python -m venv env
```

here you created virtual environment called 'env'

3. activate the env using this command

```
source env/scripts/activate
```

4. install the required packages using this command

```
pip install -r requirements.txt
```

5. here you should prepare you csv file , then you can run script prepare to create the embbeding
   after getting into backend directory , you can run by
   ` python prepare.py`

6. then you can run the fastapi APIs using
   `uvicorn main:app --reload`

## Conclusion

This project provides a simple example of how to use FastAPI and OpenAI to create a service for querying and getting relevant data using Embedding AI. The instructions above will help you get started running the code on your local machine,
