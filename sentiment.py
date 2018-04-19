import json
from watson_developer_cloud import ToneAnalyzerV3



class WatsonAnalyzer:

    def __init__(self, username, password):
        self.tone_analyzer = ToneAnalyzerV3(
            version='2017-09-21',
            username=username,
            password=password,
            url='https://gateway.watsonplatform.net/tone-analyzer/api'
        )

        self.tone_analyzer.set_default_headers({'x-watson-learning-opt-out': "true"})


    def analyze(self, text):
        content_type = 'application/json'

        tone = self.tone_analyzer.tone({"text": text}, content_type=content_type, sentences=False)

        return tone




