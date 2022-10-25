import sys
import re

def main():
  try:
    filename = sys.argv[1]
    shift = int(sys.argv[2])
  except:
    print("usage: srt-shift filename shift")
    return
  
  out = ''
  
  with open(filename, 'r') as file:
    i = 0
    for line in file:
      line = line.strip()
      if not line:
        out += '\n'
        continue
      
      i += 1
      
      if re.compile('^(\d+)$').match(line):
        i = 1
      
      if i == 1:
        out += '%s\n' % line
      
      elif i == 2:
        start, end = line.split(' --> ')
        
        def parse_time(time):
          hour, minute, second = time.split(':')
          hour, minute = int(hour), int(minute)
          second_parts = second.split(',')
          second = int(second_parts[0])
          microsecond = int(second_parts[1])
          
          return [hour, minute, second, microsecond]
        
        start, end = map(parse_time, (start, end))
        
        def shift_time(time):
          t = time[0] * 3600 + time[1] * 60 + time[2] + time[3] * 0.001 + shift
          if t < 0:
            return [0,0,0,0]
          time[0] = t // 3600
          time[1] = t % 3600 // 60
          time[2] = t % 60 // 1
          time[3] = t * 1000 // 1000
          return time
        
        start, end = map(shift_time, (start, end))
        
        out += '%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n' % (
          start[0],start[1],start[2],start[3],
          end[0],end[1],end[2],end[3])
        
      elif i >= 3:
        out += '%s\n' % line
  
  print(out)

if __name__ == '__main__':
  main()