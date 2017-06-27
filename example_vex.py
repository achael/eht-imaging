import vex_reader as vr

"""
vex_reader.Vex('vex file name') returns an object that contains
various observational information.
Note vex_reader.py currently assumes only one "$MODE" exists in vex files.
"""

fname = "vex_files/e17b11.vex"
vex = vr.Vex(fname)


"""
Summary of mnemonics:

--- Read ib from "$FREQ" section in .vex file
vex.freq = observation frequency in Hz
vex.bw_hz = observation bandwidth in Hz
vex.array = Andrew's Array() object constructed from .vex file (but SEFD is read in from arrays/SITES.txt)

----Read in from "$SCHED" section in .vex file
vex.sched[scan ID]['source'] = source name of "scan ID"-th scan
vex.sched[scan ID]['mjd_floor'] = floored MJD of the scan
vex.sched[scan ID]['start_hr'] = start time of the scan in UTC
vex.sched[scan ID]['mode'] = scan mode of the scan
vex.sched[scan ID]['scan'][site ID]['site'] = site name of "site-ID"-th site of the "scan ID"-th scan
vex.sched[scan ID]['scan'][site ID]['scan_sec'] = duration of the scan in second
vex.sched[scan ID]['scan'][site ID]['data_size'] = data size (?) of the scan in GB
vex.sched[scan ID]['scan'][site ID]['scan_sec_start'] = probably 'start_hr' + 'scan_sec_start'(in sec) is the actual time that the scan starts, but not sure

-- Read in from "$SOURCE" section in .vex file
vex.source[source ID]['source'] = source name of "source ID"-th source
vex.source[source ID]['ra'] = RA of the source
vex.source[source ID]['dec'] = DEC of the source
vex.source[source ID]['ref_coord_frame'] = something about the source
"""

# Example usage is shown below

#====== Obtain schedule ======
sched = vex.sched
print "\n%s There are %d scans in total.\n"%(fname,len(sched))

i_scan = 1
sched_ex = sched[i_scan]
print "Access information of scan No.%d."%(i_scan+1)
print "  source=%s\n  mjd(floored)=%f\n  scan start time=%f(UT)\n  mode=%s\n"%(\
sched_ex['source'],\
sched_ex['mjd_floor'],\
sched_ex['start_hr'],\
sched_ex['mode'])

print "  This scan includes %d stations."%len(sched_ex['scan'])
i_station = 2
print "  Properties of the scan for %d-th(st,nd,rd) station:"%(i_station+1)
sched_ex = sched_ex['scan'][i_station]
print "    site=%s\n    scan_sec_start=%f(sec)\n    scan duration=%s(sec)\n    data size=%f(GB)\n"%(\
sched_ex['site'],\
sched_ex['scan_sec_start'],\
sched_ex['scan_sec'],\
sched_ex['data_size'])


#====== Array object ======
# Andrew's array object constructed using .vex file.
# SEFD is still read in from prepared text file, arrays/SITES.txt
eht = vex.array


#====== Frequency and bandwidth ======
print "Frequency(Hz)=%f"%vex.freq
print "Bandwidth(Hz)=%f\n"%vex.bw_hz


#====== Source info ======
source = vex.source
print "%d sources are listed in this vex file (and probably observed)."%len(source)

i_source = 1
print "Info. of e.g. %d-th(st,nd,rd) source:"%(i_source+1)
source_ex = source[i_source]
print "    source=%s\n    RA=%s\n    DEC=%s\n    ref_coord_frame=%s\n"%(\
source_ex['source'],\
source_ex['ra'],\
source_ex['dec'],\
source_ex['ref_coord_frame'])


