# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:49:52 2016

@author: annaukkola
"""

### Finds drought end days ###

def find_end(consec, start_days, dry_days):
    
    import numpy as np
    
    end_days = np.zeros(len(start_days))
  
  
  
    for k in range(len(end_days)):
          
          index = np.where(np.in1d(dry_days,start_days[k]))[0] #第一个0
          
          #Last dry day         
          if dry_days[index] == max(dry_days):
              end_days[k] = dry_days[index]
          
          #Only one consec drought day
          elif consec[index+1]==0:
              end_days[k] = dry_days[index]
          
          #Several consec drought days
          else:
              while True:
                  if dry_days[index] < max(dry_days):
                      if consec[index+1]==1:
                          index = index + 1
                      else:
                          break
                  else:
                      break

              end_days[k] = dry_days[index]
         


    return end_days.astype(int);    #return as integers

# import numpy as np
# consec = [0., 1., 1., 0., 1., 1., 0., 1., 1., 0.]
# start_days = [0,4,8,13]
# dry_days = [0,1,2,4,5,6,8,9,10,13]
# dry_days = np.array(dry_days)
# consec = np.array(consec)

# dry_end = find_end(consec, start_days, dry_days)


# 输入 
# dry_days: 0,  1,  2,  4,  5,  6,  8,  9, 10, 13;
# consec:[0., 1., 1., 0., 1., 1., 0., 1., 1., 0.]
# start_days = [0,4,8,13]
# 输出
# end_days = [2,6,10,13]