# To run this code, first edit config.py with your configuration, then:
#
# mkdir data
# python twitter_stream_download.py -q apple -d data
# 
# It will produce the list of tweets for the query "apple" 
# in the file data/stream_apple.json

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import config
import json

consumer_key = "BsjF1cKB1FhgtyQ1LHuQAnxox"
consumer_secret = "DoAQlempkYBoVWOljrkqSOgb5GDUnaLHVxOb0nKGi6Eag6tM54"
access_token = "841572587364802560-GE0NGOye2xOiLVRTPFKWMkdgUrJqBeL"
access_secret = "Xv59uSufXX0CBShDe4fCoItuhGldcgImJX2KLfytWeZcO"

keylst = ["keylogger","data exfiltration","quick-fire attack","internet crippling","internet-crippling","DDoS-for-hire services","DDoS mitigation","DDoS lockdown","computer vandalism","cyberassailant","Volume Based Attacks","Protocol Attacks","Volumetric attacks","Application Layer Attacks","UDP Flood","ICMP Flood","Ping Flood","State-exhaustion attacks","State exhaustion attacks","SYN Flood","Ping of Death","Slowloris","NTP Amplification","HTTP Flood","BOOTERS","STRESSERS","DDOSERS","botnet","DDoS extortion","SYN floods","SSDP amplification","dns amplification","IP fragmentation","bot herder","botmaster","Nitol-infected","IMDDOS","DNS Targeted Attacks","DNS-Targeted Attacks","DISTRIBUTED DENIAL OF SERVICE","Hacktivism","Cyber vandalism","ddos extortion","extortion","Personal rivalry","Cyber warfare","Mitigating Network Layer Attacks","border gateway protocol","Phishing attacks","Malware","spyware","ransomware","cyber fraud","cyber crime","Application Specific Attacks","Reconnaissance","email attack","phishing","Brute force attack","Social Media Attacks","Spear Phishing","Fraudulent Tax Returns","Phishy Phone Calls","Charity Phishing","CEO Phishing","Phishing Websites","Network Security Attacks","Browser Attacks","Worm Attacks","WannaCry ransomware","Malicious websites","Malvertising","Malvertising attacks","Web Attacks","Scan Attacks","SQL Injection","Cross-Site Scripting","Cross Site Scripting","Session Hijacking","Man-in-the-Middle Attacks","Credential Reuse","Denial of Service","Cyber criminal"]
def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Downloader")
    parser.add_argument("-q",
                        "--query",
                        dest="query",
                        help="Query/Filter",
                        default='-')
    parser.add_argument("-d",
                        "--data-dir",
                        dest="data_dir",
                        help="Output/Data Directory")
    return parser


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, data_dir, query):
        query_fname = format_filename(query)
        self.outfile = "%s/stream_%s.json" % (data_dir, query_fname)

    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:
                f.write(data)
                print(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert file name into a safe string.

    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.

    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    twitter_stream = Stream(auth, MyListener(args.data_dir, args.query))
    twitter_stream.filter(track=keylst)

