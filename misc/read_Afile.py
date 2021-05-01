import os
def read_header(header):
    header_info = {}
    header_info['year'] = int(header[1:5])
    header_info['month'] = int(header[5:7])
    header_info['day']  = int(header[7:9])
    header_info['hour'] = int(header[9:11])
    header_info['minute'] = int(header[11:13])
    header_info['second'] = float(header[13:19])
    header_info['lat'] = float(header[19:21])
    header_info['lat_minute'] = float(header[21:26])
    header_info['lon'] = int(header[26:29])
    header_info['lon_minute'] = float(header[29:34])
    header_info['depth'] = float(header[34:40])
    header_info['magnitude'] = float(header[40:44])
    header_info['nsta'] = header[44:46].replace(" ","")
    header_info['Pfilename'] = header[46:58].replace(" ","")
    header_info['newNoPick'] = header[60:63].replace(" ","")
    return header_info
def read_lines(lines):
    trace = []
    for line in lines:
        try:
            line_info = {}
            line_info['code'] = str(line[1:7]).replace(" ","")
            line_info['epdis'] = float(line[7:13])
            line_info['az'] = int(line[13:17])
            line_info['phase'] = str(line[21:22]).replace(" ","")
            line_info['ptime'] = float(line[23:30])
            line_info['pwt'] = int(line[30:32])
            line_info['stime'] = float(line[33:40])
            line_info['swt'] = int(line[40:42])
            line_info['lat'] = float(line[42:49])
            line_info['lon'] = float(line[49:57])
            line_info['gain'] = float(line[57:62])
            line_info['convm'] = str(line[62:63]).replace(" ","")
            line_info['accf'] = str(line[63:75]).replace(" ","")
            line_info['durt'] = float(line[75:79])
            line_info['cherr'] = int(line[80:83])
            line_info['timel'] = str(line[83:84]).replace(" ","")
            line_info['rtcard'] = str(line[84:101]).replace(" ","")
            line_info['ctime'] = str(line[101:109]).replace(" ","")
        except ValueError:
            print(line)
            continue
        trace.append(line_info)

    return trace
def read_afile(afile_path):
    count = 0
    event_info = {}
    f = open(afile_path, 'r')
    header = f.readline()
    lines = f.readlines()
    try:
        header_info = read_header(header)
        trace_info = read_lines(lines)
        event_info['header_info'] = header_info
        event_info['trace_info'] = trace_info
        count = len(trace_info)
    except ValueError :
        print(afile_path)

    return event_info,count
def read_afile_directory(path_list):
    trace_count = 0
    events = []
    for path in path_list:
        for root, _, files in os.walk(path):
            for file in files:
                abs_path = (os.path.join(root, file))

                event,c = read_afile(abs_path)
                events.append(event)
                trace_count += c


    return events,trace_count
# path_list = ['/home/andy/A_file/2016',
#              '/home/andy/A_file/2017',
#              '/home/andy/A_file/2018',
#              '/home/andy/A_file/2019',
#              '/home/andy/A_file/2020']
path_list = ['/home/andy/A_file']
events,trace_count = read_afile_directory(path_list)













