import requests
from src.util import load_openapi_key

url_base = "http://apis.data.go.kr/9750000/PrecedentInfomationService"
verification_key = load_openapi_key()
url_judgement = "/getKorPrcdntDetail"

max_row = "1000000"


def get_prcdnt_lists(event_type="헌나"):
    url_api = "/getKorPrcdntList"
    
    params = {"serviceKey": verification_key,
              "eventType": event_type,
              "numOfRows": max_row,
              "type": "json"
              }

    # Send GET request
    response = requests.get(url_base + url_api, params=params)

    # Check status code
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        return None


def get_prcdnt(event_num="84503", panre_type="02"):
    url_api = "/getKorPrcdntDetail"
    
    params = {"serviceKey": verification_key,
              "eventNum": event_num,
              "panreType": panre_type,
              "type": "json"
              }

    # Send GET request
    response = requests.get(url_base + url_api, params=params)

    # Check status code
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        return None

