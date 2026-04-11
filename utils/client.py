import openai
from openai import OpenAI
import os

def azure_client(base_url, api_version, ak):
    client = openai.AzureOpenAI(
        base_url=base_url,
        api_key=ak,
        api_version=api_version,
    )
    return client

def openai_client(base_url, ak):
    client = openai.OpenAI(
        base_url=base_url,
        api_key=ak,
    )
    return client


def get_client(type:str=None, base_url=None, api_version=None, ak=None):
    if type is None:
        type = os.getenv("CLIENT_TYPE", "openai")
    if type not in ["azure", "openai"]:
        raise ValueError(f"type must be either 'azure' or 'openai', but got {type}")
    
    missing_vars = []
    
    if base_url is None:
        base_url = os.getenv("OPENAI_ENDPOINT")
        if base_url is None:
            missing_vars.append("OPENAI_ENDPOINT")
    

    if ak is None:
        ak = os.getenv("OPENAI_API_KEY")
        if ak is None:
            missing_vars.append("OPENAI_API_KEY")
    
    if missing_vars:
        raise ValueError(f"The following environment variables are not set: {', '.join(missing_vars)}\nPlease set these environment variables or provide them as arguments.")
    
    if type == "azure":
        if api_version is None:
            api_version = os.getenv("OPENAI_API_VERSION")
            if api_version is None:
                missing_vars.append("OPENAI_API_VERSION")
    
        return azure_client(base_url, api_version, ak)
    else:
        return openai_client(base_url, ak)