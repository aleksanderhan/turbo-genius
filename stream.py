import requests

url = "http://192.168.1.13:8000/stream"
params = {'prompt': 'Tell me about the connection between the spectral gap problem and the halting problem.'}

response = requests.get(url, params=params, stream=True)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode('utf-8'), end='')
