# Documentation https://code.google.com/p/yahoo-finance-managed/wiki/csvHistQuotesDownload
# updated for python 3.5
# Months are 0..11

import http.client

def fetchYahoo(symbol, begYear, begMonth, begDay,  endYear, endMonth, endDay):
  uri = "/table.csv?s=" + symbol + "&d=" + str(endMonth) + "&e=" + str(endDay) + "&f=" + str(endYear) + "&g=d&a=" + str(begMonth) + "&b=" + str(begDay) + "&c=" + str(begYear) + "&ignore=.csv"
  print ("uri=",  uri)
  conn = http.client.HTTPConnection("ichart.finance.yahoo.com")
  conn.request("GET", uri)
  r1 = conn.getresponse()
  data1 = r1.read()
  print (data1)
  fname = "data/" + symbol + ".csv"
  f = open(fname, "w")
  f.write(data1.decode())
  f.close()
  return data1
  
  
#           symbol begin     End
fetchYahoo("SPY",  2010,0,1, 2017,1,27)


