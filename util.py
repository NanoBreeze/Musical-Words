import json

def getCredentials(file):

    credentials = json.load(open(file))
    return (credentials['username'], credentials['password'])
