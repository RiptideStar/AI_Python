import requests

# This function will pass your numbers to the machine learning model
# and return the top result with the highest confidence
def classify(numbers):
    key = "4aff97c0-8504-11eb-be4b-734540ae4c3002268c3e-4a2f-4c53-9aef-5885e05a02b1"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"

    response = requests.post(url, json={ "data" : numbers })

    if response.ok:
        responseData = response.json()
        topMatch = responseData[0]
        return topMatch
    else:
        response.raise_for_status()


# CHANGE THIS to something you want your machine learning model to classify
data1 = 1
data2 = "male"
data3 = 30
data4 = 2
data5 = 2
data6 = 4
data7 = "Cherbourg"

demo = classify([ data1, data2, data3, data4, data5, data6, data7  ])

label = demo["class_name"]
confidence = demo["confidence"]


# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))