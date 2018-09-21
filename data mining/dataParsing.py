import re
import pandas as pd 
import numpy as np

format_pat= re.compile(
    r"(?P<host>[\d\.]+)\s"
    r"(?P<identity>\S*)\s"
    r"(?P<user>\S*)\s"
    r"\[(?P<time>.*?)\]\s"
    r'"(?P<request>.*?)"\s'
    r"(?P<status>\d+)\s"
    r"(?P<bytes>\S*)\s"
    r'"(?P<referer>.*?)"\s'
    r'"(?P<user_agent>.*?)"\s*'
)

def add_item(dict,item,value):
    if(item not in dict):
        d[item] = []
    d[item].append(value)

logPath = "./data mining/access_log.txt"

URLCounts = {}
d = {}
with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            for item in access:
                if(item == 'request'):
                    request = access[item]
                    fields = request.split()
                    action = ""
                    protocol = ""
                    URL = ""
                    if (len(fields) == 3):
                        (action, URL, protocol) = request.split()
                        if action == 'GET' and URL.endswith("/"):
                            if URL in URLCounts:
                                URLCounts[URL] = URLCounts[URL] + 1
                            else:
                                URLCounts[URL] = 1
                    add_item(d,"action",action)
                    add_item(d,"URL",URL)
                    add_item(d,"protocol",protocol)
                else:
                    add_item(d,item,access[item])
                
            
df = pd.DataFrame.from_dict(d)
results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for column in df.columns:
    degree_counts = df[column].value_counts()
    print(degree_counts.nlargest(5))

print(df[(df["status"] == "500")]["time"].value_counts().nlargest(10))


for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))