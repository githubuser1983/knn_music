from sage.all import *
import numpy as np
import random

 
def tspWithDistanceMatrix(distance_matrix,exact=True):
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
    from python_tsp.exact import solve_tsp_dynamic_programming
    if exact:
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    else:
        random.seed(0)
        permutation, distance = solve_tsp_simulated_annealing(distance_matrix,max_processing_time=30.0)
    return permutation, distance


def parseMidi(fp,part=0,volume_bounded_by_one=False,subtract21=False):
    import os
    from music21 import converter
    score = converter.parse(fp,quantizePost=True)
    score.makeVoices()
    from music21 import chord
    durs = []
    ll0 = []
    vols = []
    isPauses = []
    if subtract21:
        subtractThis = 21
    else:
        subtractThis = 0
    for p in score.elements[part].recurse().notesAndRests: #parts[0].streams()[part].notesAndRests:
        if type(p)==chord.Chord:
            pitches = sorted([e.pitch.midi-subtractThis for e in p])[0] # todo: think about chords
            vol = sorted([e.volume.velocity for e in p])[0]
            dur = float(p.duration.quarterLength)
            isPause = 0
        elif (p.name=="rest"):
            pitches = 64
            vol = 64
            dur = float(p.duration.quarterLength)
            isPause = 1
        else:
            pitches = p.pitch.midi-subtractThis
            vol = p.volume.velocity
            dur = float(p.duration.quarterLength)
            isPause =  0
        if not dur>0 and vol>0:
            continue
        ll0.append(min(87,pitches))    
        durs.append(dur)
        if vol is None or vol == 0:
            vol = 64
        if volume_bounded_by_one:    
            vols.append(vol*1.0/127.0)
        else:
            vols.append(vol*1.0)
        isPauses.append(isPause)
    return ll0,durs,vols,isPauses

def kernPause(a1,a2):
    return  1*(a1==a2)

def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)


import portion as PP
def muInterval(i):
    if i.empty:
        return 0
    return i.upper-i.lower

def jaccard(i1,i2):
    return muInterval(i1.intersection(i2))/muInterval(i1.union(i2))

def kernJacc(x,y):
    if 0.0<= x  and 0.0<= y :
        if x==y==0.0:
            return 1
        X = PP.closed(0,x)
        Y = PP.closed(0,y)
        return jaccard(X,Y)

def kernDurationQuotient(d1,d2):
    q = QQ(d1/d2)
    a,b = (q).numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)    
    
def kernDuration(k1,k2):
    #return kernJacc(k1,k2)
    return min(k1,k2)/max(k1,k2)
    #return kernDurationQuotient(k1,k2)

def kernVolume(v1,v2):
    #return kernJacc(v1,v2)
    return min(v1,v2)/max(v1,v2)

def kernAdd(t1,t2,alphaPitch=0.25):
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    #return 1.0/3*(1-alphaPitch)*kernPause(isPause1,isPause2)+alphaPitch*kernPitch(pitch1,pitch2)+1.0/3*(1-alphaPitch)*kernDuration(duration1,duration2)+1.0/3*(1-alphaPitch)*kernVolume(volume1,volume2)
    apa = alphaPitch["pause"]
    api = alphaPitch["pitch"]
    adu = alphaPitch["duration"]
    avo = alphaPitch["volume"]
    if np.abs(apa+api+adu+avo-1)<10**-5:
        return apa*kernPause(isPause1,isPause2)+api*kernPitch(pitch1,pitch2)+adu*kernDuration(duration1,duration2)+avo*kernVolume(volume1,volume2)
    else:
        return None

def kernMul(t1,t2):
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    alpha = 0.1
    return (1-alpha)*kernPause(isPause1,isPause2)+alpha*(kernPitch(pitch1,pitch2)*kernDuration(duration1,duration2)*kernVolume(volume1,volume2))

def kern(alphaPitch):
    return lambda t1,t2: kernAdd(t1,t2,alphaPitch)


def getRational(k):
    alpha = 2**(1/12.0)
    x = RDF(alpha**k).n(50)
    return x.nearby_rational(max_error=0.01*x)


def ngrams(input, n):
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

def kernNgram(ngrams1,ngrams2,alphaPitch=0.25):
    return 1.0/len(ngrams1)*sum([ kern(alphaPitch)(ngrams1[i], ngrams2[i]) for i in range(len(ngrams1))]) 

def writePitches(fn,inds,tempo=82,instrument=[0,0],add21=True,start_at= [0,0],durationsInQuarterNotes=False):
    from MidiFile import MIDIFile

    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats # In BPM
    volume   = 116 # 0-127, as per the MIDI standard

    ni = len(inds)
    MyMIDI = MIDIFile(ni,adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
    MyMIDI.addTempo(track,time, tempo)


    for k in range(ni):
        MyMIDI.addProgramChange(k,k,0,instrument[k])


    times = [0.0,0.0] #start_at
    for k in range(len(inds)):
        channel = k
        track = k
        for i in range(len(inds[k])):
            pitch,duration,volume,isPause = inds[k][i]
            #print(pitch,duration,volume,isPause)
            track = k
            channel = k
            if not durationsInQuarterNotes:
                duration = 4*duration#*maxDurations[k] #findNearestDuration(duration*12*4)            
            #print(k,pitch,times[k],duration,100)
            if not isPause: #rest
                #print(volumes[i])
                # because of median:
                pitch = int(floor(pitch))
                if add21:
                    pitch += 21
                #print(pitch,times[k],duration,volume,isPause)    
                MyMIDI.addNote(track, channel, int(pitch), float(times[k]) , float(duration), int(volume))
                times[k] += duration*1.0  
            else:
                times[k] += duration*1.0
       
    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    print("written")    
    


# Idee: Benutze den Jaccard Koeff als positiv definiten Kernel um zwei Intervalle miteinander zu vergleichen.
# https://en.wikipedia.org/wiki/Jaccard_index

def kernIntervalMult(x1,x2):
    i1,n1 = x1
    i2,n2 = x2
    return jaccard(i1,i2)*kernNgram(n1,n2)

def kernIntervalAdd(x1,x2):
    i1,n1 = x1
    i2,n2 = x2
    alpha = 0.750
    return alpha*(jaccard(i1,i2))+(1-alpha)*kernNgram(n1,n2)

def kernInterval(x1,x2):
    return kernIntervalAdd(x1,x2)


def distInterval1(interval):
    return lambda x1,x2 : np.sqrt(2-2*kernInterval(interval[int(x1)],interval[int(x2)]))

def distNgram(interval,alphaPitch=0.25):
    return lambda x1,x2 : np.sqrt(2-2*kernNgram(interval[int(x1)],interval[int(x2)],alphaPitch))


def distInterval2(x1,x2):
    return np.sqrt(2*(1-kernInterval(x1,x2)))

def distKern(x,y,alphaPitch=0.25):
    #print(alphaPitch)
    return np.sqrt(2-2*kern(alphaPitch)(x,y))

def generateNotes(octave,durs=[0,1],alphaPitch=0.25):
    from itertools import product
    from music21 import pitch
    pitchlist = [p for p in list(range(60-1*octave*12,60+24-1*octave*12))]
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernPitch(x,y))) for x in pitchlist] for y in pitchlist]))
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #pitchlist = [pitchlist[permutation[k]] for k in range(len(pitchlist))]
    print([pitch.Pitch(midi=int(p)) for p in pitchlist])
    durationlist = [n for n in durs]
    #if len(durs)>2:
    #    distmat = np.array(matrix([[np.sqrt(2*(1.0-kernDuration(x,y))) for x in durationlist] for y in durationlist]))
    #    permutation,distance = tspWithDistanceMatrix(distmat)
    #    durationlist = [durationlist[permutation[k]] for k in range(len(durationlist))]
    print(durationlist)
    volumelist = vols = [(128//8)*(k+1) for k in range(8)] #[x*127 for x in [1.0/6.0,1.0/3.0,1.0/2.0,2.0/3.0 ]]
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernVolume(x,y))) for x in volumelist] for y in volumelist]))
    #permutation,distance = tspWithDistanceMatrix(distmat)
    #volumelist = [volumelist[permutation[k]] for k in range(len(volumelist))]
    print(volumelist)
    pauselist = [False]
    ll = list(product(durationlist,volumelist,pauselist,pitchlist))
    #distmat = np.array(matrix([[distKern(x,y,alphaPitch) for x in ll] for y in ll]))
    #np.random.seed(43)
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #ll = [ll[permutation[k]] for k in range(len(ll))]
    print(len(ll))
    #print(ll)
    pitches = [p[3] for p in ll]
    durations = [d[0] for d in ll]
    volumes = [v[1] for v in ll]
    isPauses = [p[2] for p in ll]
    return pitches,durations,volumes,isPauses    


def findBestMatches(nbrs,new_row,n_neighbors=3):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])))
    print(dx)
    indi = [d[1] for d in dx]
    print(indi)
    #print(distances)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    return indi


def findByRadius(nbrs,new_row,radius):
    distances,indices = nbrs.radius_neighbors([np.array(new_row)],radius=radius,return_distance=True,sort_results=True)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    #print(radius,list(zip(distances[0],indices[0])))
    return indices[0],distances[0]

def findBestMatch(nbrs,new_row,radius):
    res =  []
    while len(res)==0:
        res = findByRadius(nbrs,new_row,radius)
        radius *= 1.01
    return res[-1]    


def forecastWithAStar(fn,radius_numbers=[5],voices=[0,1],Nmaxnotes= [20,10],Nforecast=[20,10],Nseq=1,Ncopy=[3,1],durs=[[0,1],[-1,0]],alphaPitch=0.25,generative=True,seeds = []):
    from sklearn.neighbors import NearestNeighbors
    import networkx as nx
    from numpy.random import choice
    from networkx.algorithms import approximation
    np.random.seed(0)
    start = 0
    print("getting notes...")
        
    iinds = []
    for v in range(len(voices)):
        print("getting notes for ",v)
        if generative:
            pitches,durations,volumes,isPauses = generateNotes(octave=voices[v],durs=durs[v],alphaPitch=alphaPitch)  
        else:
            pitches,durations,volumes,isPauses = parseMidi(fn,part=v) 
        zz0 = ngrams(list(zip(pitches,durations,volumes,isPauses)),Nseq[v])
        #print(zz0)
        print("voice = ",v)
        intrange = np.array([[x] for x in range(len(zz0))])  
        print("fitting neighbors...")
        #nn = int(max([max(knn_numbers[vv]) for vv in voices]))
        nbrs = NearestNeighbors( algorithm='ball_tree',metric=distNgram(zz0,alphaPitch=alphaPitch)).fit(intrange)
        inds = []
        
        print("constructing graph(s)...")
        n_neighbors = 2
        connected = False
        radius = 0.25
        while not connected:
            print("constructing graph with ",radius," radius, until connected ..")
            #A_knn = nbrs.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
            A_radius = nbrs.radius_neighbors_graph(radius=radius, mode='distance',sort_results=True)
            G = nx.from_numpy_array(A_radius,create_using=nx.Graph)
            print((nx.number_connected_components(G)))
            connected = nx.is_connected(G)
            n_neighbors += 1
            radius = 1.1*radius
        print("graph connected with ",n_neighbors-1," neighbors")
        print("forecasting...")
        
        preds = []
        for j in range(len(seeds[v])-1):                
            si1,si2 = seeds[v][j],seeds[v][j+1]
            print("seeds = ",si1,si2)
            rr = range(len(radius_numbers[v]))
            #for xi in range(len(intrange)-1): 
            print("computing a* between",si1,si2)
            i = nx.astar_path(G, si1, si2, heuristic=distNgram(zz0,alphaPitch=alphaPitch), weight="weight")
            i = list(i)    
            print(i)
            for ii in i:
                lz = len(zz0[ii])
                inds.extend(zz0[ii][(lz//2-Ncopy[v]):(lz//2+Ncopy[v]+1)])    
        #print(inds)        
        iinds.append(inds)            
    return(iinds)

def forecastFromRadius(fn,radius_numbers=[5],voices=[0,1],Nmaxnotes= [20,10],Nforecast=[20,10],Nseq=1,Ncopy=[3,1],durs=[[0,1],[-1,0]],alphaPitch=0.25,generative=True,seeds = []):
    from sklearn.neighbors import NearestNeighbors
    import networkx as nx
    from numpy.random import choice
    from networkx.algorithms import approximation
    np.random.seed(0)
    start = 0
    print("getting notes...")
    #intervals,ZZ,lenVoices = generateIntervals(voices=voices,Nmaxnotes=Nmaxnotes,Nseq=Nseq,durs=durs) 
    
    iinds = []
    for v in range(len(voices)):
        print("getting notes for ",v)
        if generative:
            pitches,durations,volumes,isPauses = generateNotes(octave=voices[v],durs=durs[v],alphaPitch=alphaPitch)  
        else:
            pitches,durations,volumes,isPauses = parseMidi(fn,part=v) 
        zz0 = ngrams(list(zip(pitches,durations,volumes,isPauses)),Nseq[v])
        #print(zz0)
        print("voice = ",v)
        intrange = np.array([[x] for x in range(len(zz0))])  
        print("fitting neighbors...")
        #nn = int(max([max(knn_numbers[vv]) for vv in voices]))
        nbrs = NearestNeighbors( algorithm='ball_tree',metric=distNgram(zz0,alphaPitch=alphaPitch)).fit(intrange)
        inds = []        

        print("forecasting...")
        
        preds = []
        for seed in seeds[v]:          
            preds = [np.array([seed])] 
            print("seed = ",seed)
            print("preds = ",preds)
            rr = range(len(radius_numbers[v]))
            #for xi in range(len(intrange)-1): 
            for xi in rr:
                x = preds.pop(0)
                allids,dists = findByRadius(nbrs,x,sqrt(2.0))
                i = [allids[r] for r in radius_numbers[v][xi]]
                i = list(i)
                print(i,preds)
                for ii in i:
                    lz = len(zz0[ii])
                    #inds.extend(zz0[ii][(lz//2-Ncopy[v]):(lz//2+Ncopy[v]+1)])    
                    print(zz0[ii])
                    inds.append(zz0[ii][lz//2])    
                preds.append(intrange[ii])    
        #print(inds)        
        iinds.append(inds)            
    return(iinds)

def cycle_path(path,node_move_to_zero_index):
    n = node_move_to_zero_index
    if n in path:
        i = path.index(n)
        lp = len(path)
        newpath = []
        for j in range(lp):
            newpath.append( path[(j+i)%lp] )
        return newpath    
    else:
        return None

def forecastHamiltonian(fn,seeds=[0,-1],voices=[0,1],Nmaxnotes= [20,10],Nforecast=[20,10],Nseq=1,Ncopy=[3,1],durs=[[0,1],    [-1,0]],alphaPitch=0.25):
    import networkx as nx
    from networkx.algorithms import approximation
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    start = 0
    print("getting notes...")
    #intervals,ZZ,lenVoices = generateIntervals(voices=voices,Nmaxnotes=Nmaxnotes,Nseq=Nseq,durs=durs) 
    
    iinds = []
    for v in range(len(voices)):
        print("getting notes for ",v)
        pitches,durations,volumes,isPauses = generateNotes(octave=voices[v],durs=durs[v],alphaPitch=alphaPitch)  
        zz0 = ngrams(list(zip(pitches,durations,volumes,isPauses)),Nseq[v])
        #print(zz0)
        print("voice = ",v)
        intrange = np.array([[x] for x in range(len(zz0))])  
        print("fitting neighbors...")
        #nn = int(max([max(knn_numbers[vv]) for vv in voices]))
        nbrs = NearestNeighbors( algorithm='ball_tree',metric=distNgram(zz0,alphaPitch=alphaPitch)).fit(intrange)
        print("constructing graph...")
        G = nx.Graph(nodes = intrange)
        nodes = list(range(len(zz0)))
        for a in nodes:
            for b in nodes:
                G.add_edge(a,b,weight=distNgram(zz0,alphaPitch=alphaPitch)(a,b))                
        tsp = approximation.traveling_salesman_problem
        inds = []
        
        
        #seeds = [intrange[k] for k in [0,len(zz0)//3*1,0,len(zz0)//3*2]]
        print("forecasting...")
        methods = [approximation.traveling_salesman.christofides, approximation.traveling_salesman.simulated_annealing_tsp]
        cache = dict([])
        preds = []
        si = 0
        for i in range(len(seeds[v])):
            si1 = seeds[v][i]
            # bridge between last note and next seed:
            print("si,si1 = ",si,si1)
            radius = distNgram(zz0,alphaPitch=alphaPitch)(si,si1)
            print("finding nodes by radius..")
            visit = findByRadius(nbrs,np.array([si]),radius)
            print(si,visit,si1)
            if si1 in cache.keys():
                i = cache[si1]
            else:    
                print("doing tsp... this can take some time")
                path = tsp(G, nodes=visit, cycle=True,method=methods[0])
                print("cycle path ...")
                i = cycle_path(path,si) #[ids[r] for r in radius_numbers[v][xi]]
                cache[si1] = i
            #k = i.index(si1)
            print(i)
            #print(i[0:(k+1)])
            #print(path)
            for ii in i:
                lz = len(zz0[ii])
                inds.extend(zz0[ii][(lz//2-Ncopy[v]):(lz//2+Ncopy[v]+1)])
            si = ii    
        #print(inds)        
        iinds.append(inds)            
    return(iinds)


def sum_seq(seq):
    s = seq[0]
    sum_seq = [s]
    for i in range(1,len(seq)):
        s+= seq[i]
        sum_seq.append(s)
    return sum_seq    


def diff_seq_radius_nr(Nforecast,seq):       
    radius_numbers = []
    for v in range(len(Nforecast)):
        rr =  []
        for n in range(Nforecast[v]-1):
            nthprime  = seq[v][n] #nth_prime(n+1)
            nextprime = seq[v][n+1] #next_prime(nthprime)
            print(nthprime,nextprime)
            #ll = list(range(1,nextprime-nthprime+1)) # consonant
            ll = []
            ll.extend(list(range(nextprime-nthprime+1,-1,-1))) # disonant
            rr.append(ll)
            #rr.append(list(range(4+n%8,0,-1)))
        radius_numbers.append(rr)   
    return radius_numbers    
    


def midiWithRadius(fn1,voices=[0,1],tempo=80,instruments=[0,0],Nmaxnotes=[20,10],Nforecast=[20,10],Nseq=1,Ncopy=[3,1],durs=[[0,1],[-1,0]],start_at=[0,0],alphaPitch=0.25,generative=False,seeds=[]):
    
    seq = [[nth_prime(n+1) for n in range(Nforecast[v])] for v in range(len(Nforecast))]#primes
    #seq = [sum_seq([euler_phi(n+1) for n in range(Nforecast[v])]) for v in range(len(Nforecast))] # euler_phi
    #seq = [sum_seq([valuation(2*(n+1),2) for n in range(Nforecast[v])]) for v in range(len(Nforecast))] # euler_phi
    radius_numbers = diff_seq_radius_nr(
             Nforecast,
             seq=seq)
    
    print(radius_numbers)    
    iinds = forecastFromRadius(fn1,radius_numbers=radius_numbers,voices=voices,Nmaxnotes= Nmaxnotes,Nforecast=Nforecast,Nseq=Nseq,Ncopy=Ncopy,durs=durs,alphaPitch=alphaPitch,generative=generative,seeds=seeds)
    
    #iinds = forecastWithAStar(fn1,radius_numbers=radius_numbers,voices=voices,Nmaxnotes= Nmaxnotes,Nforecast=Nforecast,Nseq=Nseq,Ncopy=Ncopy,durs=durs,alphaPitch=alphaPitch,generative=generative,seeds=seeds)
    
    
    conf = {
        "type": "radius_based_knn",
        "radius_numbers": radius_numbers,
        "voices" : voices,
        "Nmaxnotes": Nmaxnotes,
        "filename": fn1,
        "Nforecast": Nforecast,
        "Nseq":Nseq,
        "Ncopy": Ncopy,
        "durs": durs,
        "alphaPitch": alphaPitch,
        "generative": generative,
        "tempo": tempo,
        "instruments":instruments,
        "alphaPitch": alphaPitch,
        "start_at": start_at,
        "seeds": seeds
    }
    
    writeConfiguration("conf-"+conf["type"]+".yaml",conf)
    print(iinds)
    writePitches(fn1,iinds,tempo=tempo,instrument=instruments,add21=1-generative,start_at = start_at,durationsInQuarterNotes=False)

def midiWithHamiltonian(fn1,voices=[0,1],seeds= [[0,-1],[0,-1]],tempo=80,instruments=[0,0],Nmaxnotes=[20,10],Nforecast=[20,10],Nseq=1,Ncopy=[3,1],durs=[[0,1],[-1,0]],start_at=[0,0],alphaPitch=0.25):
                                    
    iinds = forecastHamiltonian(fn = fn1,seeds=seeds,voices=voices,Nmaxnotes= Nmaxnotes,Nforecast=Nforecast,Nseq=Nseq,Ncopy=Ncopy,durs=durs,alphaPitch=alphaPitch)                                   
    fn1 = fn1.split("/")[-1]
    fn = "./midi/"+fn1+".hamiltonian.mid" 
    
    conf = {
        "type": "tsp_based_loops",
        "seeds": seeds,
        "voices" : voices,
        "Nmaxnotes": Nmaxnotes,
        "filename": fn,
        "Nforecast": Nforecast,
        "Nseq":Nseq,
        "Ncopy": Ncopy,
        "durs": durs,
        "alphaPitch": alphaPitch,
        "generative": generative,
        "tempo": tempo,
        "instruments":instruments,
        "alphaPitch": alphaPitch,
        "start_at": start_at
    }
    
    writeConfiguration("conf-"+conf["type"]+".yaml",conf)
    
    writePitches(fn,iinds,tempo=tempo,instrument=instruments,add21=0,start_at = start_at,durationsInQuarterNotes=False)


def writeConfiguration(filename,configuration):    
    import yaml
    import io

    # Write YAML file
    with io.open(filename, 'w', encoding='utf8') as outfile:
        yaml.dump(configuration, outfile, default_flow_style=False, allow_unicode=True)

def readConfiguration(filename):        
    # Read YAML file
    import yaml
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded    

def getPitchVectors():
    import pandas as pd
    M = matrix([[kernPitch(p1,p2) for p1 in range(128)] for p2 in range(128)],ring=RDF)
    C = M.cholesky()
    from sklearn.decomposition import PCA
    nDim =128
    pca = PCA(n_components=nDim)
    from sklearn.preprocessing import StandardScaler
    stdScaler = StandardScaler()   
    X = pca.fit_transform(stdScaler.fit_transform(C))
    XX = []
    for k in range(nDim):
        x = [k]
        x.extend([z for z in X[k]])
        XX.append(x)
    cols = ["midi_pitch"]
    cols.extend([ "pca_"+str(i) for i in range(nDim) ])
    df = pd.DataFrame(columns = cols, data = XX)
    return df

def getDynamicsVectors():
    import pandas as pd
    dynamics = ["ppp","pp","p","mp","mf","f","ff","fff"]
    allvolume = list(range(1,128))
    vols = [(128//8)*(k+1) for k in range(8)]
    
    M = matrix([[kernVolume(p1,p2) for p1 in vols] for p2 in vols],ring=RDF)
    C = M.cholesky()
    from sklearn.decomposition import PCA
    nDim =8
    pca = PCA(n_components=nDim)
    from sklearn.preprocessing import StandardScaler
    stdScaler = StandardScaler()   
    X = pca.fit_transform(stdScaler.fit_transform(C))
    XX = []
    for k in range(nDim):
        x = [dynamics[k]]
        x.extend([z for z in X[k]])
        XX.append(x)
    cols = ["dynamics"]
    cols.extend([ "pca_"+str(i) for i in range(nDim) ])
    df = pd.DataFrame(columns = cols, data = XX)
    return df

def getDurationVectors():
    import pandas as pd
    durations = get_notevalues()
    print(durations)
    numerators = [d.numerator() for d in durations]
    denominators = [d.denominator() for d in durations]
    nDim = len(durations)
        
    M = matrix([[kernDuration(p1,p2) for p1 in durations] for p2 in durations],ring=RDF)
    C = M.cholesky()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=nDim)
    from sklearn.preprocessing import StandardScaler
    stdScaler = StandardScaler()   
    X = pca.fit_transform(stdScaler.fit_transform(C))
    XX = []
    for k in range(nDim):
        x = [durations[k]*1.0, numerators[k], denominators[k]]
        x.extend([z for z in X[k]])
        XX.append(x)
    cols = ["note_duration","note_duration_numerator","note_duration_denominator"]
    cols.extend([ "pca_"+str(i) for i in range(nDim) ])
    df = pd.DataFrame(columns = cols, data = XX)
    return df


def getRestVectors():
    import pandas as pd
    rests = [True,False]
    names = ["is_rest","is_no_rest"]
    nDim = len(rests)
        
    M = matrix([[kernPause(p1,p2) for p1 in rests] for p2 in rests],ring=RDF)
    C = M.cholesky()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=nDim)
    from sklearn.preprocessing import StandardScaler
    stdScaler = StandardScaler()   
    X = pca.fit_transform(stdScaler.fit_transform(C))
    XX = []
    for k in range(nDim):
        x = [names[k]]
        x.extend([z for z in X[k]])
        XX.append(x)
    cols = ["flag_is_rest"]
    cols.extend([ "pca_"+str(i) for i in range(nDim) ])
    df = pd.DataFrame(columns = cols, data = XX)
    return df
    
def generate_with_weights(weights=[4,1,3,2],conf=None):
    fn1 = "./midi/knn_"+"_".join([str(x) for  x in weights])+".mid"
    print(fn1)
    import os
    #fn1 = sorted([os.path.join("./midi/",fn) for fn in os.listdir("./midi/") if fn.endswith(".mid") 
    #          and "gymnopedie" in fn.lower() and not "mix" in fn.lower()])[0]  
    
    if not conf is None and conf["type"]=="radius_based_knn":
        alphaPitch = conf["alphaPitch"]
        voices = conf["voices"]
        tempo = conf["tempo"]
        instruments = conf["instruments"]
        Nmaxnotes = conf["Nmaxnotes"]
        Nforecast = conf["Nforecast"]
        Nseq = conf["Nseq"]
        Ncopy = conf["Ncopy"]
        durs = conf["durs"]
        start_at = conf["start_at"]
        generative = conf["generative"]
        fn1 = conf["filename"]
        seeds = conf["seeds"]
    else: 
        voices=[0,1]
        tempo=80
        instruments=[0,0]
        Nmaxnotes=[200,200]
        Nforecast=[40,80]
        Nseq=[1,1]
        Ncopy=[0,0]
        durs = [[1/2,1/4],[1/16,1/8]]
        start_at= [0,0.0]
        sW = sum(weights)
        probs = [w*1.0/sW for w in weights]
        alphaPitch = dict(zip(["pitch","duration","volume","pause"],probs))
        generative=True
        nSeeds = 50
        seeds = []
        for v in range(len(voices)):
            pitches,durations,volumes,isPauses = generateNotes(octave=voices[v],durs=durs[v],alphaPitch=alphaPitch)  
            zz0 = ngrams(list(zip(pitches,durations,volumes,isPauses)),Nseq[v])
            #print(zz0)
            #seeds = [intrange[k] for k in [0,len(zz0)//3*1,0,len(zz0)//3*2]]
            import random
            random.seed(11)
            s1 = random.randint(0,len(zz0))
            s2 = random.randint(0,len(zz0))
            s3 = random.randint(0,len(zz0))
            #seeds.append([s1,len(zz0)//3*1,len(zz0)//3*2])
            aims = [n for n in range(1,3)]
            random.shuffle(aims)
            seeds.append(aims) #,s2,s3])
            #print("-->",zz0[0])
    
    gamma = (1+sqrt(5.0))/2
    
    
    midiWithRadius(fn1,
                voices=voices,
                tempo=tempo,
                instruments=instruments,
                Nmaxnotes=Nmaxnotes,
                Nforecast=Nforecast,
                Nseq=Nseq,
                Ncopy=Ncopy,durs = durs,
                start_at= start_at,
                alphaPitch=alphaPitch,
                generative=generative,seeds=seeds)

def generate_hamiltonian(weights=[4,1,2,3]):
    weights = weights
    fn1 = "./midi/knn_"+"_".join([str(x) for  x in weights])+".mid"
    print(fn1)
    import os
    #fn1 = sorted([os.path.join("./midi/",fn) for fn in os.listdir("./midi/") if fn.endswith(".mid") 
    #          and "gymnopedie" in fn.lower() and not "mix" in fn.lower()])[0]  
    gamma = (1+sqrt(5.0))/2
    
    sW = sum(weights)
    probs = [w*1.0/sW for w in weights]
    alphaPitch = dict(zip(["pause","pitch","duration","volume"],probs))
    Nnotes = []
    durs = [[1/2,1/4,1/8,1/16],[1/16,1/8,1/4,1/2]]
    voices = [0,1]
    for v in range(len(voices)):
        voice=voices[v]
        Ns = len(generateNotes(octave=voice,durs=durs[v],alphaPitch=alphaPitch)[0])-1
        Nnotes.append(Ns)
    nrSeeds= 4  
    seeds = [[k*Nnotes[v]//nrSeeds for k in [1,2,4]] for v in voices]
    
    seed = [nth_prime(n+1)-nth_prime(n) for n in range(1,210)]
    seed = [euler_phi(n) for n in range(1,210)]
    seeds = [seed,seed]
    print(seeds)
    
    midiWithHamiltonian(fn1,
                voices=voices,
                seeds = seeds,       
                tempo=80,
                instruments=[0,0],
                Nmaxnotes=[200,200],
                Nforecast=[20,5],
                Nseq=[1,1],
                Ncopy=[0,0],durs = durs,
                start_at= [0,0.0],
                alphaPitch=alphaPitch)
    
    
def permutations():
    for p in Permutations([2,3,4]):
        weight = list(p)
        weight.insert(1,1)
        print(weight)
        generate_with_weights(weights=weight)
        
def gen1():
    gamma = (1+sqrt(5.0))/2
    x = 1/gamma**5
    y = 1/3.0*(1-x)
    weights = [y,x,y,y]
    weights = [4,1,2,3]
    generate_with_weights(weights = weights)

def func(list_of_sims):
    var("a1,a2,a3,a4")
    return sum([np.sqrt(2*(1-(a1*sim[0]+a2*sim[1]+a3*sim[2]+a4*sim[3]))) for sim in list_of_sims])

def transform_to_eps(x,eps):
    if 0.0<eps<10**-3:
        return x*(1-2*eps)+eps
    
def get_seq_dist_from_midi(fn,alphaPitch,voice=0):    
    import pandas as pd
    v = voice
    # beta regression:
    # https://cran.r-project.org/web/packages/betareg/vignettes/betareg.pdf
    pitches,durations,volumes,isPauses = parseMidi(fn,part=v) 
    durations = [d*1/4.0 for d in durations]
    zz0 = list(zip(pitches,durations,volumes,isPauses))
    list_of_dists = []
    for i in range(len(zz0)-1):
        list_of_dists.append(distKern(kern(alphaPitch))(zz0[i],zz0[i+1])*1.0)
    return list_of_dists

def gen_dataset(fn,alphaPitch,Npast=10,voice=0):
    import pandas as pd
    v = voice
    # beta regression:
    # https://cran.r-project.org/web/packages/betareg/vignettes/betareg.pdf
    pitches,durations,volumes,isPauses = parseMidi(fn,part=v) 
    durations = [d*1/4.0 for d in durations]
    zz0 = list(zip(pitches,durations,volumes,isPauses))
    list_of_sims = []
    for i in range(len(zz0)-1):
        sim = [kern(alphaPitch)(zz0[i],zz0[i+1])*1.0]
        sim = [transform_to_eps(s,10**-4) for s in sim]
        list_of_sims.append(sim)
    #print(list_of_sims) 
    ngs = ngrams(list_of_sims,Npast+1)
    #order = ["pitch","duration","volume","rest"]
    #
    ll = []
    for ng in ngs:
        #print(ng)
        row = []
        cols = []
        counter = 0
        for notes in ng:
            #print(notes)
            counter += 1
            row.extend(notes)
            j = 0
            for note in notes:
                cols.append("note_"+str(counter))
                j += 1
        ll.append(row)
    return pd.DataFrame(data=ll,columns=cols),zz0        
            
        
    #weights = minimize_distance(list_of_sims)
    #print(weights)
    
def predict_results(results,one_row):
    from statsmodels.api import add_constant
    preds = []
    for res in results:
        #print((res.predict.__doc__))
        pred = res.predict(one_row)
        #print(one_row)
        #print(len(pred))
        preds.append(pred[0])
    return np.array(preds)    

def get_notevalues():
    durationslist = [[sum([(QQ(2)**(n-i)) for i in range(d+1)]) for n in range(-8,3+1)] for d in range(0,3+1)]
    notevals = []
    for i in range(len(durationslist)):
        notevals.extend(durationslist[i])
    notevals = sorted(notevals)    
    return notevals

def distKern(kern):
    return lambda a,b : np.sqrt(2*(1-kern(a,b)))

def get_knn_model_durations(durations):
    #notevals = np.array([[x*1.0] for x in durations])
    #print(notevals)
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernDuration)).fit(durations)
    return nbrs,durations
                   
def get_knn_model_pitches(pitches):
    #pitches = np.array([[x*1.0] for x in pitches])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernPitch)).fit(pitches)
    return nbrs,pitches                    

def get_knn_model_volumes(volumes):
    #volumes = np.array([[x*1.0] for x in volumes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernVolume)).fit(volumes)
    return nbrs,volumes                    

def get_knn_model_rests(rests):
    #rests = np.array([[x*1.0] for x in rests])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernPause)).fit(rests)
    return nbrs,rests     

def get_knn_model_notes(notes,alphaPitch):
    #notes = np.array([[x*1.0 for x in n] for n in notes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kern(alphaPitch))).fit(notes)
    return nbrs,notes
                        
def get_nearest_note_by_radius(notes_nbrs,notes_list,note,radius):
    note = notes_list[findBestMatch(notes_nbrs,np.array([1.0*x for x in note]),radius)]
    return note


def getProbsFromWeights(weights):
    sW = sum(weights)
    probs = [w*1.0/sW for w in weights]
    alphaPitch = dict(zip(["pitch","duration","volume","pause"],probs))
    return alphaPitch

def make_np_array(ll):
    return np.array([[x*1.0] for x in ll])

def plot_graph(typeOfGraph,startRadius=0.25,plotInFile = True):
    import numpy as np, networkx as nx
    dd = {"pitches": (make_np_array(range(128)), get_knn_model_pitches),
          "durations": (make_np_array(get_notevalues()), get_knn_model_durations),
          "volumes":  (make_np_array([(128//8)*(k+1) for k in range(8)]), get_knn_model_volumes),
          "rests": (make_np_array([True,False]),get_knn_model_rests),
          }
    ll = dd[typeOfGraph][0]
    func = dd[typeOfGraph][1]
    nbrs,lls = func(ll)
    print("constructing graph(s)...for type = ",typeOfGraph)
    n_neighbors = 2
    connected = False
    radius = startRadius
    while not connected:
        print("constructing graph with ",radius," radius, until connected ..")
        #A_knn = nbrs.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
        A_radius = nbrs.radius_neighbors_graph(radius=radius, mode='distance',sort_results=True)
        G = nx.from_numpy_array(A_radius,create_using=nx.Graph)
        print((nx.number_connected_components(G)))
        connected = nx.is_connected(G)
        n_neighbors += 1
        radius = 1.1*radius
    print("graph connected with ",radius/1.1," radius")
    if plotInFile:
        Gr = Graph(G,loops=True)
        Gr.plot().save("./plots/"+typeOfGraph+"_graph_radius_"+str(np.round(radius/1.1,2))+".png")
    return G,ll
    

   
def get_nearest_note(pitches_nbrs,pitches_list,
                     durations_nbrs,durations_list,
                     volumes_nbrs,volumes_list,
                     rests_nbrs,rests_list,
                     current_note, new_row):
    cpitch, cduration, cvolume, crest = current_note
    print(cpitch)
    radius_pitch,radius_duration,radius_volume,radius_rest = new_row            
    pitch = pitches_list[findBestMatch(pitches_nbrs,np.array([cpitch]),radius_pitch)][0]
    volume = volumes_list[findBestMatch(volumes_nbrs,np.array([cvolume]),radius_volume)][0]                                    
    duration = durations_list[findBestMatch(durations_nbrs,np.array([cduration]),radius_duration)][0]                           
    rest = rests_list[findBestMatch(rests_nbrs,np.array([crest]),radius_rest)][0]                                        
    return (pitch,duration, volume,rest)                      


def getForecastForVoice(fn,Npast,voice,Nforecast=200):
    import statsmodels.api as sm
    
    fn1 = fn    
    print(fn1)
    
    weights = [1,1,2,2]
    sW = sum(weights)
    probs = [w*1.0/sW for w in weights]
    alphaPitch = dict(zip(["pitch","duration","volume","pause"],probs))
    
    df,notes = gen_dataset(fn1,alphaPitch,Npast=Npast,voice=voice)

    notes_nbrs,notes_list = get_knn_model_notes(notes,alphaPitch)
    
    print(df)
    print(df.shape)   
    
    results = []
    #X_cols_list = ["note_"+str(counter)+"_"+order[j] for counter in range(1,Npast+1) for j in range(len(order))]
    X_cols_list = ["note_"+str(counter) for counter in range(1,Npast+1)]
    y_col = "note_"+str(Npast+1)
    print(X_cols_list)
    print(y_col)
    y = np.array(df[[y_col]])
    X = np.array(df[X_cols_list])
    print(X.shape,y.shape)
    #logit, probit, cauchy, log, loglog, and cloglog
    binom_glm = sm.GLM(y, X, family=sm.families.Binomial(link=sm.genmod.families.links.logit))      
    rslt = binom_glm.fit()
    print(rslt.summary())
    results.append(rslt)    
    one_row = np.array(df.iloc[0][X_cols_list])
    print(one_row)
    current_note = notes[1]
    print(current_note)
    inds = [current_note]
    for k in range(Nforecast):
        new_row = (predict_results(results,one_row))  
        current_note = get_nearest_note_by_radius(notes_nbrs,notes_list,current_note,[np.sqrt(2*(1-x)) for x in new_row][0]) 
        inds.append(current_note)
        #print(new_row)
        #print(current_note)
        ll = list(one_row)
        ll.extend(list(new_row))
        one_row = ll[-len(X_cols_list):] 
    return inds

def experiment():
    fn1 = sorted([os.path.join("./midi/",fn) for fn in os.listdir("./midi/") if fn.endswith(".mid") 
              and "gymnopedie" in fn.lower() and not "mix" in fn.lower()])[0]  
    Npast = 5
    print(fn1)
    inds0 = getForecastForVoice(fn1,Npast=Npast,voice=0)
    inds1 = getForecastForVoice(fn1,Npast=Npast,voice=1)
    iinds = [inds0,inds1]
    fn = "./midi/"+fn1.split("./midi/")[1]+"-beta.mid"
    tempo = 80
    instruments = [0,0]
    writePitches(fn,iinds,tempo=tempo,instrument=instruments,add21=0,start_at = [0,0],durationsInQuarterNotes=False)

def writeVectors(): 
    import pandas as pd
    from itertools import product
    pitchVectors = getPitchVectors()
    pitchVectors.to_csv("midi_pitch_pca_vectors.csv",sep=";",index=False,header=True)
    
    dynamicVectors = getDynamicsVectors()
    dynamicVectors.to_csv("dynamics_pca_vectors.csv",sep=";",index=False,header=True)
    
    durationVectors = getDurationVectors()
    durationVectors.to_csv("note_values_pca_vectors.csv",sep=";",index=False,header=True)
    
    restVectors = getRestVectors()
    restVectors.to_csv("rest_pca_vectors.csv",sep=";",index=False,header=True)
    
    noteNames = list(product(
                       pitchVectors[[ c for c in pitchVectors.columns if not "pca" in c]].values,
           durationVectors[[ c for c in durationVectors.columns if not "pca" in c]].values,
           dynamicVectors[[ c for c in dynamicVectors.columns if not "pca" in c]].values,
           restVectors[[ c for c in restVectors.columns if not "pca" in c]].values
                 )
                )
    
    colNames = []
    colNames.extend(pitchVectors[[ c for c in pitchVectors.columns if not "pca" in c]].columns)
    colNames.extend(durationVectors[[ c for c in durationVectors.columns if not "pca" in c]].columns)
    colNames.extend(dynamicVectors[[ c for c in dynamicVectors.columns if not "pca" in c]].columns)
    colNames.extend(restVectors[[ c for c in restVectors.columns if not "pca" in c]].columns)
                 
    
    noteVectors = list(product(
                       pitchVectors[[ c for c in pitchVectors.columns if "pca" in c]].values,
           durationVectors[[ c for c in durationVectors.columns if "pca" in c]].values,
           dynamicVectors[[ c for c in dynamicVectors.columns if "pca" in c]].values,
           restVectors[[ c for c in restVectors.columns if "pca" in c]].values
                 )
                )
    notevec = []
    #notenames = [ "_".join([str(xx[0]) for xx in x]) for x in list(noteNames)]
    c = 0
    for nv in noteVectors:
        xx = []
        for nn in noteNames[c]: 
            xx.extend(nn)
        for y in nv:
            xx.extend(list(y))
        notevec.append(xx)
        c+=1
    print(notevec[0])    
    Ndim = sum([len(list(y)) for y in noteVectors[0]])
    cols = colNames
    cols.extend(["vec_"+str(i) for i in range(1,Ndim+1)])
    
    df = pd.DataFrame(data=notevec,columns=cols)
    
    df.to_csv("note_vectors.csv",sep=";",index=False,header=True)
    
    print(df.head())
    
    
    #print(notes[0:10])
    #print(cols)
    

    
def findMidiFile(search):
    import os
    fn = sorted([os.path.join("./midi/",fn) for fn in os.listdir("./midi/") if fn.endswith(".mid") 
              and search in fn.lower() and not "mix" in fn.lower()])[0]
    return fn

def overview_dists():
    for midifile in ["knn_1_3_2_4","for_elise","gymnopedie","einaudi","bach_minor","beethoven_9th"]:
        fn = findMidiFile(midifile)
        print(fn)
        alphaPitch = getProbsFromWeights([1,3,2,4])
        dists = get_seq_dist_from_midi(fn,alphaPitch,voice=1)
        #print(dists)
        print(min(dists),median(dists),max(dists))
        
def dump_graphs():
    import networkx as nx
    import pickle
    G_pitches,ll_pitches = plot_graph("pitches",startRadius=0.05)
    G_durations,ll_durations = plot_graph("durations",startRadius=0.05)
    G_volumes,ll_volumes = plot_graph("volumes",startRadius=0.05)
    G_rests,ll_rests = plot_graph("rests",startRadius=0.05) 
    
    ll = [(G_pitches,ll_pitches),
          (G_durations,ll_durations),
          (G_volumes,ll_volumes),
          (G_rests,ll_rests)
         ]
    
    with open("./data/graphs.pkl","wb") as f:
        pickle.dump(ll,f)
    
    
def load_graphs():
    import pickle
    with open("./data/graphs.pkl","rb") as f:
        ll = pickle.load(f)
    return ll    

def get_paths_between_two_notes(ll_graphs, n0,n1):
    import networkx as nx
    p0,d0,v0,r0 = n0
    p1,d1,v1,r1 = n1
    t_pitches, t_durations,t_volumes,t_rests = ll_graphs
    G_pitches, ll_pitches = t_pitches
    G_durations, ll_durations = t_durations
    G_volumes, ll_volumes = t_volumes
    G_rests, ll_rests = t_rests
    ll_pitches = list(ll_pitches)
    ll_durations = list(ll_durations)
    ll_volumes = list(ll_volumes)
    ll_rests = list(ll_rests)
    path_pitches = nx.astar_path(G_pitches, ll_pitches.index(p0), ll_pitches.index(p1), heuristic=distKern(kernPitch), weight="weight")
    path_durations = nx.astar_path(G_durations, ll_durations.index(d0), ll_durations.index(d1), heuristic=distKern(kernDuration), weight="weight")
    path_volumes = nx.astar_path(G_volumes, ll_volumes.index(v0), ll_volumes.index(v1), heuristic=distKern(kernVolume), weight="weight")
    path_rests = nx.astar_path(G_rests, ll_rests.index(r0), ll_rests.index(r1), heuristic=distKern(kernPause), weight="weight")
    
    pp = [ll_pitches[x] for x in path_pitches]
    pd = [ll_durations[x] for x in path_durations]
    pv = [ll_volumes[x] for x in path_volumes]
    pr = [ll_rests[x] for x in path_rests]
    return pp,pd,pv,pr



def experiment2():
    from itertools import product
    #dump_graphs()
    ll = load_graphs()
    t_pitches, t_durations,t_volumes,t_rests = ll
    G_pitches, dd_pitches = t_pitches
    G_durations, dd_durations = t_durations
    G_volumes, dd_volumes = t_volumes
    G_rests, dd_rests = t_rests
    #print(G_pitches.nodes)
    dd = {"pitches": (make_np_array(range(128)), get_knn_model_pitches),
          "durations": (make_np_array(get_notevalues()), get_knn_model_durations),
          "volumes":  (make_np_array([(128//8)*(k+1) for k in range(8)]), get_knn_model_volumes),
          "rests": (make_np_array([True,False]),get_knn_model_rests),
          }
    notes = list(product(dd["pitches"][0],dd["durations"][0],dd["volumes"][0],dd["rests"][0]))
    n0 = notes[0]
    n1 = notes[100]
    print(n0,n1)
    pp,pd,pv,pr = get_paths_between_two_notes(ll,n0,n1)
    print(list(product(pp,pd,pv,pr)))


def plot_graphs_connected():
    import networkx as nx
    import pickle
    G_pitches,ll_pitches = plot_graph("pitches",startRadius=0.05)
    G_durations,ll_durations = plot_graph("durations",startRadius=0.05)
    G_volumes,ll_volumes = plot_graph("volumes",startRadius=0.05)
    G_rests,ll_rests = plot_graph("rests",startRadius=0.05) 
    
    si1,si2 = 3,2
    path = nx.astar_path(G_pitches, si1, si2, heuristic=distKern(kernPitch), weight="weight")
    print("pitches>",path)
        
if __name__=="__main__":
    #writeVectors(
    #overview_dists()
    #random.seed(0)
    #plot_graphs_connected()
    #experiment2()
    import sys
    if len(sys.argv)==2:
        conf = readConfiguration(sys.argv[1])
    else:
        conf = None
    print(conf)
    #generate_hamiltonian(weights=[4,1,2,3],conf=conf)
    generate_with_weights(weights=[3,4,2,1],conf=conf)
    