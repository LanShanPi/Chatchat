import requests
url = "http://localhost:8500/embeddings"
payload = {
    "input": "哈哈哈哈哈",
    "model": "acge_text_embedding"
}
headers = {
    "Content-Type": "application/json"
}
try:
    response = requests.post(url, json=payload, headers=headers)
    print(response)
    response.raise_for_status()  # 如果响应有错误，抛出异常
    print(response)
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")

