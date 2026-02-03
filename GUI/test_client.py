import requests

url = 'http://10.135.143.248:5000/predict'
filename = r"C:\Users\hp\Documents\INNOVATIVE PROJECT\Images\test.jpg"

with open(filename, 'rb') as f:
    files = {'image': f}
    response = requests.post(url, files=files)
    print(response.json())
