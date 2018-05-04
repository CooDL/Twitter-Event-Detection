# coding=utf-8

__version__ = '1.0'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2017-May-11'
__url__     = 'https://engineering.purdue.edu/kak/distLSH/LocalitySensitiveHashing-1.0.html'
__copyright__ = "(C) 2017 Avinash Kak. Python Software Foundation."

__doc__ = '''
Version: '''+ __version__ + '''
Author: Avinash Kak (kak@purdue.edu)
Date: '''+ __date__ + '''
@endofdocs
'''


import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *

#-----------------------------------  Utility Functions  ------------------------------------

def sample_index(sample_name):
    '''
    We assume that the raw data is stored in the following form:

       sample0_0,0.951,-0.134,-0.102,0.079,0.12,0.123,-0.03,-0.078,0.036,0.138
       sample0_1,1.041,0.057,0.095,0.026,-0.154,0.231,-0.074,0.005,0.055,0.14
       ...
       ...
       sample1_8,-0.153,1.083,0.041,0.086,-0.059,0.042,-0.172,0.014,-0.153,0.091
       sample1_9,0.051,1.122,-0.014,-0.117,0.015,-0.044,0.011,0.008,-0.121,-0.017
       ...
       ...

    This function returns the second integer in the name of each data record.
    It is useful for sorting the samples and for visualizing whether or not
    the final clustering step is working correctly.
    '''
    m = re.search(r'_(.+)$', sample_name)
    return int(m.group(1))

def sample_group_index(sample_group_name):
    '''
    As the comment block for the previous function explains, the data sample
    for LSH are supposed to have a symbolic name at the beginning of the 
    comma separated string.  These symbolic names look like 'sample0_0', 
    'sample3_4', etc., where the first element of the name, such as 'sample0',
    indicates the group affiliation of the sample.  The purpose of this
    function is to return just the integer part of the group name.
    '''
    m = re.search(r'^.*(\d+)', sample_group_name)
    return int(m.group(1))

def band_hash_group_index(block_name):
    '''
    The keys of the final output that is stored in the hash self.coalesced_band_hash
    are strings that look like:

         "block3 10110"

    This function returns the block index, which is the integer that follows the 
    word "block" in the first substring in the string that you see above.
    '''
    firstitem = block_name.split()[0]
    m = re.search(r'(\d+)$', firstitem)
    return int(m.group(1))

def deep_copy_array(array_in):
    '''
    Meant only for an array of scalars (no nesting):
    '''
    array_out = []
    for i in range(len(array_in)):
        array_out.append( array_in[i] )
    return array_out

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def cleanup_csv(line):
    line = line.translate(bytes.maketrans(b":?/()[]{}'",b"          ")) \
           if sys.version_info[0] == 3 else line.translate(string.maketrans(":?/()[]{}'","          "))
    double_quoted = re.findall(r'"[^\"]*"', line[line.find(',') : ])         
    for item in double_quoted:
        clean = re.sub(r',', r'', item[1:-1].strip())
        parts = re.split(r'\s+', clean.strip())
        line = str.replace(line, item, '_'.join(parts))
    white_spaced = re.findall(r',(\s*[^,]+)(?=,|$)', line)
    for item in white_spaced:
        litem = item
        litem = re.sub(r'\s+', '_', litem)
        litem = re.sub(r'^\s*_|_\s*$', '', litem) 
        line = str.replace(line, "," + item, "," + litem) if line.endswith(item) else str.replace(line, "," + item + ",", "," + litem + ",") 
    fields = re.split(r',', line)
    newfields = []
    for field in fields:
        newfield = field.strip()
        if newfield == '':
            newfields.append('NA')
        else:
            newfields.append(newfield)
    line = ','.join(newfields)
    return line

# Needed for cleanly terminating the interactive method lsh_basic_for_nearest_neighbors():
def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)

#----------------------------------- LSH Class Definition ------------------------------------

class LocalitySensitiveHashing(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception(  
                   '''LocalitySensitiveHashing constructor can only be called with keyword arguments for the 
                      following keywords: datafile,csv_cleanup_needed,how_many_hashes,r,b,
                      similarity_group_min_size_threshold,debug,
                      similarity_group_merging_dist_threshold,expected_num_of_clusters''') 
        allowed_keys = 'datafile','dim','csv_cleanup_needed','how_many_hashes','r','b','similarity_group_min_size_threshold','similarity_group_merging_dist_threshold','expected_num_of_clusters','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        datafile=dim=debug=csv_cleanup_needed=how_many_hashes=r=b=similarity_group_min_size_threshold=None
        similarity_group_merging_dist_threshold=expected_num_of_clusters=None
        if kwargs and not args:
            if 'csv_cleanup_needed' in kwargs : csv_cleanup_needed = kwargs.pop('csv_cleanup_needed')
            if 'datafile' in kwargs : datafile = kwargs.pop('datafile')
            if 'dim' in kwargs :  dim = kwargs.pop('dim')
            if 'r' in kwargs  :  r = kwargs.pop('r')
            if 'b' in kwargs  :  b = kwargs.pop('b')
            if 'similarity_group_min_size_threshold' in kwargs  :  
                similarity_group_min_size_threshold = kwargs.pop('similarity_group_min_size_threshold')
            if 'similarity_group_merging_dist_threshold' in kwargs  :  
                similarity_group_merging_dist_threshold = kwargs.pop('similarity_group_merging_dist_threshold')
            if 'expected_num_of_clusters' in kwargs  :  
                expected_num_of_clusters = kwargs.pop('expected_num_of_clusters')
            if 'debug' in kwargs  :  debug = kwargs.pop('debug')
        if datafile:
            self.datafile = datafile
        else:
            raise Exception("You must supply a datafile")
        self._csv_cleanup_needed = csv_cleanup_needed
        self.similarity_group_min_size_threshold = similarity_group_min_size_threshold
        self.similarity_group_merging_dist_threshold = similarity_group_merging_dist_threshold
        self.expected_num_of_clusters = expected_num_of_clusters
        if dim:
            self.dim = dim
        else:
            raise Exception("You must supply a value for 'dim' which stand for data dimensionality")
        self.r = r                               # Number of rows in each band (each row is for one hash func)
        self.b = b                               # Number of bands.
        self.how_many_hashes =  r * b
        self._debug = debug
        self._data_dict = {}                     # sample_name =>  vector_of_floats extracted from CSV stored here
        self.how_many_data_samples = 0
        self.hash_store = {}                     # hyperplane =>  {'plus' => set(), 'minus'=> set()}
        self.htable_rows  = {}
        self.index_to_hplane_mapping = {}
        self.band_hash = {}                      # BitVector column =>  bucket for samples  (for the AND action)
        self.band_hash_mean_values = {}          # Store the mean of the bucket contents in band_hash dictionary
        self.similarity_group_mean_values = {}
        self.coalesced_band_hash = {}            # Coalesce those keys of self.band_hash that have data samples in common
        self.similarity_groups = []
        self.coalescence_merged_similarity_groups = []  # Is a list of sets
        self.l2norm_merged_similarity_groups = []  # Is a list of sets
        self.merged_similarity_groups = None
        self.pruned_similarity_groups = []
        self.evaluation_classes = {}             # Used for evaluation of clustering quality if data in particular format

    def get_data_from_csv(self):
        if not self.datafile.endswith('.csv'): 
            Exception("Aborted. get_training_data_from_csv() is only for CSV files")
        data_dict = {}
        with open(self.datafile) as f:
            for i,line in enumerate(f):
                if line.startswith("#"): continue      
                record = cleanup_csv(line) if self._csv_cleanup_needed else line
                parts = record.rstrip().split(r',')
                data_dict[parts[0].strip('"')] = list(map(lambda x: convert(x), parts[1:]))
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        self.how_many_data_samples = i + 1
        self._data_dict = data_dict

    def show_data_for_lsh(self):
        print("\n\nData Samples:\n\n")
        for item in sorted(self._data_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item)

    def initialize_hash_store(self):
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

    def hash_all_data_with_one_hyperplane(self):
        hyperplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
        print( "hyperplane: %s" % str(hyperplane) )
        hyperplane = hyperplane / numpy.linalg.norm(hyperplane)
        for sample in self._data_dict:
            bin_val = numpy.dot( hyperplane, self._data_dict[sample])
            bin_val = 1 if bin_val>= 0 else -1      
            print( "%s: %s" % (sample, str(bin_val)) )

    def hash_all_data(self):
        for hplane in self.hash_store:
            for sample in self._data_dict:
                hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  ")) \
                       if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
                bin_val = numpy.dot(list(map(convert, hplane_vals.split())), self._data_dict[sample])
                bin_val = 1 if bin_val>= 0 else -1      
                if bin_val>= 0:
                    self.hash_store[hplane]['plus'].add(sample)
                else:
                    self.hash_store[hplane]['minus'].add(sample)

    def lsh_basic_for_nearest_neighbors(self):
        '''
        Regarding this implementation of LSH, note that each row of self.htable_rows corresponds to 
        one hash function.  So if you have 3000 hash functions for 3000 different randomly chosen 
        orientations of a hyperplane passing through the origin of the vector space in which the
        numerical data is defined, this table has 3000 rows.  Each column of self.htable_rows is for
        one data sample in the vector space.  So if you have 80 samples, then the table has 80 columns.
        The output of this method consists of an interactive session in which the user is asked to
        enter the symbolic name of a data record in the dataset processed by the LSH algorithm. The
        method then returns the names (some if not all) of the nearest neighbors of that data point.
        '''
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        for (i,_) in enumerate(sorted(self.hash_store)):
            if i % self.r == 0: print()
            print( str(self.htable_rows[i]) )
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):                
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)
        if self._debug:
            print( "\n\nPre-Coalescence results:" )
            for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):
                print()
                print( "%s    =>   %s" % (key, str(self.band_hash[key])) )
        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        while True:
            sample_name = None
            if sys.version_info[0] == 3:
                sample_name =  input('''\nEnter the symbolic name for a data sample '''
                                     '''(must match names used in your datafile): ''')
            else:
                sample_name = raw_input('''\nEnter the symbolic name for a data sample '''
                                        '''(must match names used in your datafile): ''')
            if sample_name in similarity_neighborhoods:
                print( "\nThe nearest neighbors of the sample: %s" % str(similarity_neighborhoods[sample_name]) )
            else:
                print( "\nThe name you entered does not match any names in the database.  Try again." )
        return similarity_neighborhoods

    def lsh_basic_for_neighborhood_clusters(self):
        '''
        This method is a variation on the method lsh_basic_for_nearest_neighbors() in the following
        sense: Whereas the previous method outputs a hash table whose keys are the data sample names
        and whose values are the immediate neighbors of the key sample names, this method merges
        the keys with the values to create neighborhood clusters.  These clusters are returned as 
        a list of similarity groups, with each group being a set.
        '''
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        for (i,_) in enumerate(sorted(self.hash_store)):
            if i % self.r == 0: print
            print( str(self.htable_rows[i]) ) 
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):                
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)
        if self._debug:
            print("\n\nPre-Coalescence results:")
            for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):
                print()
                print( "%s    =>    %s" % (key, str(self.band_hash[key])) )
        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        print("\n\nSimilarity neighborhoods calculated by the basic LSH algo:")
        for key in sorted(similarity_neighborhoods, key=lambda x: sample_index(x)):
            print( "\n  %s   =>  %s" % (key, str(sorted(similarity_neighborhoods[key], key=lambda x: sample_index(x)))) )
            simgroup = set(similarity_neighborhoods[key])
            simgroup.add(key)
            self.similarity_groups.append(simgroup)
        print( "\n\nSimilarity groups calculated by the basic LSH algo:\n" )
        for group in self.similarity_groups:
            print(str(group))
            print()
        print( "\nTotal number of similarity groups found by the basic LSH algo: %d" % len(self.similarity_groups) )
        return self.similarity_groups

    def merge_similarity_groups_with_coalescence(self, similarity_groups):
        '''
        The purpose of this method is to do something that, strictly speaking, is not the right thing to do
        with an implementation of LSH.  We take the clusters produced by the method 
        lsh_basic_for_neighborhood_clusters() and we coalesce them based on the basis of shared data samples.
        That is, if two neighborhood clusters represented by the sets A and B have any data elements in 
        common, we merge A and B by forming the union of the two sets.
        '''
        merged_similarity_groups = []
        for group in similarity_groups:
            if len(merged_similarity_groups) == 0:
                merged_similarity_groups.append(group)
            else:
                new_merged_similarity_groups = []
                merge_flag = 0
                for mgroup in merged_similarity_groups:
                    if len(set.intersection(group, mgroup)) > 0:
                        new_merged_similarity_groups.append(mgroup.union(group))
                        merge_flag = 1
                    else:
                       new_merged_similarity_groups.append(mgroup)
                if merge_flag == 0:
                    new_merged_similarity_groups.append(group)     
                merged_similarity_groups = list(map(set, new_merged_similarity_groups))
        for group in merged_similarity_groups:
            print( str(group) )
            print()
        print( "\n\nTotal number of MERGED similarity groups using coalescence: %d" % len(merged_similarity_groups) )
        self.coalescence_merged_similarity_groups = merged_similarity_groups
        return merged_similarity_groups

    def merge_similarity_groups_with_l2norm_sample_based(self, similarity_groups):
        '''
        The neighborhood set coalescence as carried out by the previous method will generally result
        in a clustering structure that is likely to have more clusters than you may be expecting to
        find in your data. This method first orders the clusters (called 'similarity groups') according 
        to their size.  It then pools together the data samples in the trailing excess similarity groups.  
        Subsequently, for each data sample in the pool, it merges that sample with the closest larger 
        group.
        '''
        similarity_group_mean_values = {}
        for group in similarity_groups:            #  A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
            if self._debug:
                print( "\n\nCLUSTER MEAN: %f" % group_mean )
        new_similarity_groups = []
        key_to_small_group_mapping = {}
        key_to_large_group_mapping = {}
        stringified_list = [str(item) for item in similarity_groups]
        small_group_pool_for_a_given_large_group = {x : [] for x in stringified_list}
        if len(similarity_groups) > self.expected_num_of_clusters:
            ordered_sim_groups_by_size = sorted(similarity_groups, key=lambda x: len(x), reverse=True)
            retained_similarity_groups = ordered_sim_groups_by_size[:self.expected_num_of_clusters]
            straggler_groups = ordered_sim_groups_by_size[self.expected_num_of_clusters :]
            print( "\n\nStraggler groups: %s" % str(straggler_groups) )
            samples_in_stragglers =  sum([list(group) for group in straggler_groups], [])
            print( "\n\nSamples in stragglers: %s" %  str(samples_in_stragglers) )
            straggler_sample_to_closest_retained_group_mapping = {sample : None for sample in samples_in_stragglers}
            for sample in samples_in_stragglers:
                dist_to_closest_retained_group_mean, closest_retained_group = None, None
                for group in retained_similarity_groups:
                        dist = l2norm(similarity_group_mean_values[str(group)], self._data_dict[sample])
                        if dist_to_closest_retained_group_mean is None:
                            dist_to_closest_retained_group_mean = dist
                            closest_retained_group = group
                        elif dist < dist_to_closest_retained_group_mean:
                            dist_to_closest_retained_group_mean = dist
                            closest_retained_group = group
                        else:
                            pass
                straggler_sample_to_closest_retained_group_mapping[sample] = closest_retained_group
            for sample in samples_in_stragglers:
                straggler_sample_to_closest_retained_group_mapping[sample].add(sample)
            print( "\n\nDisplaying sample based l2 norm merged similarity groups:" )
            self.merged_similarity_groups_with_l2norm = retained_similarity_groups
            for group in self.merged_similarity_groups_with_l2norm:
                print( str(group) )
            return self.merged_similarity_groups_with_l2norm
        else:
            print('''\n\nNo sample based merging carried out since the number of clusters yielded by coalescence '''
                  '''is fewer than the expected number of clusters.''')
            return similarity_groups

    def merge_similarity_groups_with_l2norm_set_based(self, similarity_groups):
        '''
        The overall goal of this method is the same as that of 
        merge_similarity_groups_with_l2norm_sample_based(), except for the difference that
        we now merge the excess similarity groups wholesale with the retained similarity 
        groups.  For each excess similarity group, we find the closest retained similarity group,
        closest in terms of the l2 norm distance between the mean values of the two groups.
        '''    
        similarity_group_mean_values = {}
        for group in similarity_groups:                # A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
            if self._debug:
                print( "\n\nCLUSTER MEAN: %f" % group_mean )
        if len(similarity_groups) > self.expected_num_of_clusters:
            new_similarity_groups = []
            key_to_small_group_mapping = {}
            key_to_large_group_mapping = {}
            ordered_sim_groups_by_size = sorted(similarity_groups, key=lambda x: len(x), reverse=True)
            retained_similarity_groups = ordered_sim_groups_by_size[:self.expected_num_of_clusters]
            straggler_groups = ordered_sim_groups_by_size[self.expected_num_of_clusters :]
            print( "\n\nStraggler groups: %s" % str(straggler_groups) )
            print( "\n\nNumber of samples in retained groups: %d" % len(list(set.union(*retained_similarity_groups))) )
            print( "\n\nNumber of samples in straggler groups: %d" % len(list(set.union(*straggler_groups))) )
            retained_stringified_list = [str(item) for item in retained_similarity_groups]
            small_group_pool_for_a_given_large_group = {x : [] for x in retained_stringified_list}
            for group1 in straggler_groups:
                key_to_small_group_mapping[str(group1)] = group1
                dist_to_closest_large_group_mean, closest_large_group = None, None
                for group2 in retained_similarity_groups:
                    key_to_large_group_mapping[str(group2)] = group2
                    dist = l2norm(similarity_group_mean_values[str(group2)], similarity_group_mean_values[str(group1)])
                    if dist_to_closest_large_group_mean is None:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    elif dist < dist_to_closest_large_group_mean:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    else:
                        pass
                small_group_pool_for_a_given_large_group[str(closest_large_group)].append(group1)
            if any(len(small_group_pool_for_a_given_large_group[x]) > 0 for x in small_group_pool_for_a_given_large_group):
                print( "\n\nTHERE IS NON-ZERO POOL FOR MERGING FOR AT LEAST ONE LARGER SIMILARITY GROUPS" )
                print( str(small_group_pool_for_a_given_large_group.values()) )
            for key in small_group_pool_for_a_given_large_group:
                lgroup = key_to_large_group_mapping[key]
                list_fo_small_groups = small_group_pool_for_a_given_large_group[key]
                print( "\n\nFor group %s, the pool of small groups for merging =====>  %s" % 
                                                                          (str(lgroup), str(list_fo_small_groups)) )
            for group in sorted(retained_similarity_groups, key=lambda x: len(x), reverse=True):
                group_copy = set(group)     # shallow copy
                if len(small_group_pool_for_a_given_large_group[str(group)]) > 0:
                    for setitem in small_group_pool_for_a_given_large_group[str(group)]:
                        group_copy.update(setitem)  
                    new_similarity_groups.append(group_copy)
                else:
                    new_similarity_groups.append(group_copy)
            self.merged_similarity_groups_with_l2norm = new_similarity_groups
            print( "\n\nDisplaying set based l2 norm merged similarity groups:")
            for group in new_similarity_groups:
                print( str(group) )
            return new_similarity_groups
        else:
            print('''\n\nNo set based merging carried out since the number of clusters yielded by coalescence '''
                  '''is fewer than the expected number of clusters.''')
            return similarity_groups

    def prune_similarity_groups(self):
        '''
        If your data produces too many similarity groups, you can get rid of the smallest with
        this method.  In order to use this method, you must specify a value for the parameter
        'similarity_group_min_size_threshold' in the call to the constructor of the LSH module.
        '''
        if self.merged_similarity_groups is not None:
            self.pruned_similarity_groups = [x for x in self.merged_similarity_groups if len(x) > 
                                                            self.similarity_group_min_size_threshold] 
        else:
            self.pruned_similarity_groups = [x for x in self.similarity_groups if len(x) > 
                                                        self.similarity_group_min_size_threshold]
        print( "\nNumber of similarity groups after pruning: %d" % len(self.pruned_similarity_groups) )      
        print( "\nPruned similarity groups: " )
        for group in self.pruned_similarity_groups:
            print( str(group) )
        return self.pruned_similarity_groups

    def evaluate_quality_of_similarity_groups(self, evaluation_similarity_groups):
        '''
        The argument to this method, evaluation_similarity_groups, is a list of sets, with each set being 
        a similarity group, which is the same thing as a cluster.

        If you plan to invoke this method to evaluate the quality of clustering achieved by the values
        used for the parameters r and b, you'd want the data records in the CSV datafile to look like:

            sample0_3,0.925,-0.008,0.009,0.058,0.092,0.117,-0.076,0.239,0.086,-0.149
        
        Note in particular the syntax used for naming a data record. The name 'sample0_3' means that this 
        is the 3rd sample generated randomly for data class 0.  The goal of this method is to example all 
        such  sample names and figure out how many classes exist in the data.
        '''
        print( '''\n\nWe measure the quality of a similarity group by taking stock of how many '''
               '''different different input similarity groups are in the same output similarity group.''')
        sample_classes = set()
        for item in sorted(self._data_dict.items(), key = lambda x: sample_index(x[0]) ):
            sample_classes.add(item[0][:item[0].find(r'_')])
        self.evaluation_classes = sample_classes
        if len(self.evaluation_classes) == 0:
            sys.exit('''\n\nUnable to figure out the number of data classes in the datafile processed by '''
                     '''this module --- aborting''')                     
        total_num_samples_in_all_similarity_groups = 0
        print( "\n\nTotal number of similarity groups tested: %d" % len(evaluation_similarity_groups) )
        m = re.search('^([a-zA-Z]+).+_', list(self._data_dict.keys())[0])
        sample_name_stem = m.group(1)
        for group in sorted(evaluation_similarity_groups, key=lambda x: len(x), reverse=True):
            total_num_samples_in_all_similarity_groups += len(group)
            set_for_sample_ids_in_group = set()
            how_many_uniques_in_each_group = {g : 0 for g in self.evaluation_classes}
            for sample_name in group:
                m = re.search('^[\w]*(.+)_', sample_name)
                group_index_for_sample = int(m.group(1))
                set_for_sample_ids_in_group.add(group_index_for_sample)
                how_many_uniques_in_each_group[sample_name_stem + str(group_index_for_sample)] += 1
            print( "\n\nSample group ID's in this similarity group: %s" % str(set_for_sample_ids_in_group) )
            print( "    Distribution of sample group ID's in similarity group: " )
            for key in sorted(how_many_uniques_in_each_group, key=lambda x: sample_group_index(x)):
                if how_many_uniques_in_each_group[key] > 0:
                    print("        Number of samples with Group ID " + str(sample_group_index(key)) + 
                                                           " => " + str(how_many_uniques_in_each_group[key]))
            if len(set_for_sample_ids_in_group) == 1:
                print("    Group purity level: ", 'pure')
            else:
                print("    Group purity level: ", 'impure')
        print( "\n\nTotal number of samples in the different clusters: %d" % total_num_samples_in_all_similarity_groups )

    def write_clusters_to_file(self, clusters, filename):
        FILEOUT = open(filename, 'w')
        for cluster in clusters:
            FILEOUT.write( str(cluster) + "\n\n" )
        FILEOUT.close()

    def show_sample_to_initial_similarity_group_mapping(self):
        self.sample_to_similarity_group_mapping = {sample : [] for sample in self._data_dict}
        for sample in sorted(self._data_dict.keys(), key=lambda x: sample_index(x)):        
            for key in sorted(self.coalesced_band_hash, key=lambda x: band_hash_group_index(x)):            
                if (self.coalesced_band_hash[key] is not None) and (sample in self.coalesced_band_hash[key]):
                    self.sample_to_similarity_group_mapping[sample].append(key)
        print( "\n\nShowing sample to initial similarity group mappings:" )
        for sample in sorted(self.sample_to_similarity_group_mapping.keys(), key=lambda x: sample_index(x)):
            print( "\n %s     =>    %s" % (sample, str(self.sample_to_similarity_group_mapping[sample])) )

    def display_contents_of_all_hash_bins_pre_lsh(self):
        for hplane in self.hash_store:
            print( "\n\n hyperplane: %s" % str(hplane) )
            print( "\n samples in plus bin: %s" % str(self.hash_store[hplane]['plus']) )
            print( "\n samples in minus bin: %s" % str(self.hash_store[hplane]['minus']) )
#-----------------------------  End of Definition for Class LSH --------------------------------


#----------------------  Generate Your Own Data For Experimenting with LSH ------------------------

class DataGenerator(object):
    def __init__(self, *args, **kwargs ):
        if args:
            raise SyntaxError('''DataGenerator can only be called with keyword arguments '''
                              '''for the following keywords: output_csv_file, how_many_similarity_groups '''
                              '''dim, number_of_samples_per_group, and debug''') 
        allowed_keys = 'output_csv_file','dim','covariance','number_of_samples_per_group','how_many_similarity_groups','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise Exception("Wrong keyword used --- check spelling") 
        output_csv_file=dim=covariance=number_of_samples_per_group=debug=None
        if 'output_csv_file' in kwargs :       output_csv_file = kwargs.pop('output_csv_file')
        if 'dim' in kwargs:             dim = kwargs.pop('dim')
        if 'covariance' in kwargs:             covariance = kwargs.pop('covariance')
        if 'debug' in kwargs:                  debug = kwargs.pop('debug')
        if 'how_many_similarity_groups' in kwargs:  
                         how_many_similarity_groups = kwargs.pop('how_many_similarity_groups')
        if 'number_of_samples_per_group' in kwargs:      
                                  number_of_samples_per_group = kwargs.pop('number_of_samples_per_group')
        if output_csv_file:
            self._output_csv_file = output_csv_file
        else:
            raise Exception('''You must specify the name for a csv file for the training data''')
        if dim:
            self.dim = dim
        else:
            raise Exception('''You must specify the dimensionality of your problem''')        
        if covariance is not None: 
            self.covariance = covariance
        else:
            self.covariance = numpy.diag([1] * dim)
        if number_of_samples_per_group:
            self.number_of_samples_per_group = number_of_samples_per_group
        else:
            raise Exception('''You forgot to specify the number of samples per similarity group''')
        if how_many_similarity_groups:
            self.how_many_similarity_groups = how_many_similarity_groups
        else:
            self.how_many_similarity_groups = dim
        if debug:
            self._debug = debug
        else:
            self._debug = 0


    def gen_data_and_write_to_csv(self):
        '''
        Note that a unit cube in N dimensions has 2^N corner points.  The coordinates of all these
        corner points are given by the bit patterns of the integers 0, 1, 2, ...., 2^N - 1.
        For example, in a vector 3-space, a unit cube has 8 corners whose coordinates are given by
        the bit patterns for the integers 0, 1, 2, 3, 4, 5, 6, 7.  These bit patterns would be
        000, 001, 010, 011, 100, 101, 110, 111.

        This script uses only N of the 2^N vertices of a unit cube as the mean vectors for N 
        similarity groups.   These N vertices correspond to the far points on the cube edges that
        emanate at the origin.  For example, when N=3, it uses only 001,010,100 as the three mean
        vectors for the AT MOST 3 similarity groups.  If needed, we can add additional similarity 
        groups by selecting additional coordinate bit patterns from the integers 0 through 2^N - 1.
        '''
        mean_coords = numpy.diag([1] * self.how_many_similarity_groups)
        if self.how_many_similarity_groups < self.dim:
            mean_coords = list(map(lambda x: x + [0] * (self.dim - self.how_many_similarity_groups),
                          [mean_coords[i,].tolist() for i in range(self.how_many_similarity_groups)]))
        else:
            Exception('''The logic for the case when number of similarity groups exceeds '''
                      '''the number of dimensions has not yet been coded''')
        print( "\nShowing the mean vector used for each cluster:" )
        print( str(mean_coords) )
        sample_records = []
        for i in range(self.how_many_similarity_groups):
            k = len(sample_records)
            new_samples = numpy.random.multivariate_normal(mean_coords[i], 
                                                 self.covariance, self.number_of_samples_per_group)
            new_samples = [list(map(float, map(lambda x: "%.3f" % x, sample_coords))) for sample_coords in new_samples]
            for j in range(len(new_samples)):
                sample_records.append('sample' + str(i) + '_' + str(j+k) + ',' 
                                                     + ','.join(list(map(lambda x: str(x), new_samples[j]))) + "\n")
        print("Writing data to the file %s" % self._output_csv_file)
        FILE = open(self._output_csv_file, 'w') 
        list(map(FILE.write, sample_records))
        FILE.close()    
#------------------------  End of Definition for Class DataGenerator ---------------------------


#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':

    '''
    dim = 10
    covar = numpy.diag([0.01] * dim)
    output_file = 'data_for_lsh.csv'
    data_gen = DataGenerator( 
                              output_csv_file   = output_file,
                              how_many_similarity_groups = 10,
                              dim = dim,
                              number_of_samples_per_group = 8,
                              covariance = covar,
                            )

    data_gen.gen_data_and_write_to_csv()
    '''


    lsh = LocalitySensitiveHashing( datafile = "data_for_lsh.csv",  
                                    dim = 10,
                                    r = 5,                              # number of rows in each band
                                    b = 20,                 # number of bands.   IMPORTANT: Total number of hash fns:  r * b
                                  )
    lsh.get_data_from_csv()
    lsh.show_data_for_lsh()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    lsh.display_contents_of_all_hash_bins_pre_lsh()
    lsh.lsh_basic_for_neighborhood_clusters()
    lsh.show_sample_to_similarity_group_mapping()

