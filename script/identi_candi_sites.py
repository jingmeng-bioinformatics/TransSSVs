#!/usr/bin/env python

'''Indentify candidate somatic small variant sites in a mixed mpileup file from tumor
   and normal bam files generated by smatools for evaluation with deep convolutional neural network models.
'''

import argparse
import re
import itertools
import time
import math
from xphyle import xopen

parser = argparse.ArgumentParser()

parser.add_argument('--Tumor_Normal_mpileup', required=True, metavar='pileup', help='a mixed mpileup file from tumor and normal bam files generated by samtools')
parser.add_argument('--Candidate_somatic_sites', required=True, metavar='file', help='identified candidate somatic sites')

number_of_columns=7
args = parser.parse_args()

def pairwise(iterable, number_of_columns = number_of_columns):
    # n = number_of_columns
    '''Return every pair of items that are separated by n items from iterable.
       s --> (s[0], s[n]), (s[1], s[n+1]), (s[2], s[n+2]), ...'''
    a, b = itertools.tee(iterable)
    next(itertools.islice(a, number_of_columns, number_of_columns), None)
    next(itertools.islice(b, 2*number_of_columns, 2*number_of_columns), None)
    return zip(a, b)

def find_indel(pattern, source):
    # find all the lengths of inserted or deleted sequences 
    indel = re.findall(pattern, source)
    if len(indel) == 0:
        return [], 0, [], 0, 0  # no indels found
    else:
        indel_new = [int(x[1:]) for x in indel]
        length_of_indel = max(indel_new, key = indel_new.count)  # the length of inserted or deleted bases
        
        if pattern[1] == '+':
            indel_forward = re.findall('\+' + str(length_of_indel) + '(' + '[ACGTN]' + '{' + str(length_of_indel) + '}' + ')', source)
        else:
            indel_forward = re.findall('\-' + str(length_of_indel) + '(' + '[ACGTN]' + '{' + str(length_of_indel) + '}' + ')', source)
        
        # the length of indels in forward strand
        indel_forward_count = len(indel_forward) 
        # the occurences of indels with such length
        indel_count = indel_new.count(length_of_indel) 
        return indel_new, length_of_indel, indel_forward, indel_forward_count, indel_count 

def small_variant_count(source, pattern = '[\+\-][0-9]+'):
    # the occurences of all the indels
    allele_indel = re.findall(pattern, source)
    allele_indel_count = len(allele_indel)
    
    # the number of all the bases that are different from reference base
    allele_total = len(re.findall('[ACGTNacgtn]', source))
    allele_indel_sum = sum([int(x[1:]) for x in allele_indel])
    return allele_total-allele_indel_sum+allele_indel_count   # the number of indels and non-reference base

def process_line(line_content, line, Cs):
    pattern = ['A', 'a', 'T', 't', 'G', 'g', 'C', 'c', '\+[0-9]+', '\-[0-9]+']
    k = 10   # depth in tumor and normal
    m = 0.05 # variant allele frequency in tumor  

    # keep just the genomics sites that meet the criteria
    if line_content[2] not in 'Nn' and int(line_content[3]) >= k and int(line_content[8]) >= k and sum([len(re.findall(base, line_content[4])) for base in pattern]) > 0:
        
        # generate the information about indels of mapped reads in tumor and normal
        allele_insertion_tumor_new, length_of_inserted_bases, inserted_bases_forward, insertion_forward_count, insertion_count = find_indel('\+[0-9]+', line_content[4])
        allele_deletion_tumor_new, length_of_deleted_bases, deleted_bases_forward, deletion_forward_count, deletion_count = find_indel('\-[0-9]+', line_content[4])
        
        # replace indels of pattern '-[0-9]+[ACGTNacgtn]+' or '\+[0-9]+[ACGTNacgtn]+' with X
        if allele_insertion_tumor_new == [] and allele_deletion_tumor_new != []:
            allele_deletion_tumor_set = list(set(allele_deletion_tumor_new))
            field_holder = line_content[4]
            
            # replace deletions in mapped reads in tumor
            for number_deletion in allele_deletion_tumor_set:
                create_pattern = '\-' + str(number_deletion) + '[ACGTNacgtn]' + '{' + str(number_deletion) + '}'
                field_holder = re.sub(create_pattern, 'x', field_holder)
        
        elif allele_deletion_tumor_new == [] and allele_insertion_tumor_new != []:
            allele_insertion_tumor_set = list(set(allele_insertion_tumor_new))
            field_holder = line_content[4]
            
            # replace insertions in mapped reads in tumor
            for number in allele_insertion_tumor_set:
                create_pattern = '\+' + str(number) + '[ACGTNacgtn]' + '{' + str(number) + '}'
                field_holder = re.sub(create_pattern, 'x', field_holder)
        
        elif allele_insertion_tumor_new == [] and allele_deletion_tumor_new == []:
            field_holder = line_content[4]   # no indels found in mapped reads in tumor
        
        else:
            # both insertions and deletions found in mapped reads in tumor
            allele_insertion_tumor_set = list(set(allele_insertion_tumor_new))
            allele_deletion_tumor_set = list(set(allele_deletion_tumor_new))
            field_holder = line_content[4]
            
            for number in allele_insertion_tumor_set:
                create_pattern = '\+' + str(number) + '[ACGTNacgtn]' + '{' + str(number) + '}'
                field_holder = re.sub(create_pattern, 'x', field_holder)
            
            for number_deletion in allele_deletion_tumor_set:
                create_pattern = '\-' + str(number_deletion) + '[ACGTNacgtn]' + '{' + str(number_deletion) + '}'
                field_holder = re.sub(create_pattern, 'x', field_holder)
        
        # the number of non-reference bases in mapped reads in tumor
        allele_mismatch_count_iter = iter([len(re.findall(base, field_holder)) for base in 'AaTtGgCcNn'])
        allele_forward_count = []
        allele_count = []
        
        # create a list with the occurences of mismatches in mapped reads in tumor
        for number in allele_mismatch_count_iter:
            allele_forward_count.append(number)   # the occurences of mismatches in the forward strand
            allele_count.append(number + next(itertools.islice(allele_mismatch_count_iter, 0, 1))) # the occurences of mismatches in total
        
        # a list with the occurences of mismatches and indels
        allele_count.extend([insertion_count, deletion_count])
        allele_forward_count.extend([insertion_forward_count, deletion_forward_count])

        # a list of non-reference allele types
        if allele_insertion_tumor_new == [] and allele_deletion_tumor_new != []:
            if deletion_forward_count == 0:
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '', ''] # no deletion found in the forward strand
            else:
                # add the information about the deletion
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '', '-' + str(length_of_deleted_bases) + str(deleted_bases_forward[0])]
        
        elif allele_deletion_tumor_new == [] and allele_insertion_tumor_new != []:
            if insertion_forward_count == 0:
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '', '']     # no insertion found in the forward strand
            else:
                # add the information about the insertion 
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '+' + str(length_of_inserted_bases) + str(inserted_bases_forward[0]), '']
        
        elif allele_insertion_tumor_new == [] and allele_deletion_tumor_new == []:
            allele_types_total = ['A', 'T', 'G', 'C', 'N', '', '']  # no indels found
        
        else:
            if insertion_forward_count == 0 and deletion_forward_count == 0:
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '', '']
            elif insertion_forward_count != 0 and deletion_forward_count == 0:
                # add the information about the insertion in the forward strand
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '+' + str(length_of_inserted_bases) + str(inserted_bases_forward[0]), '']
            elif insertion_forward_count == 0 and deletion_forward_count != 0:
                # add the information about the deletion in the forward strand
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '', '-' + str(length_of_deleted_bases) + str(deleted_bases_forward[0])]
            else:
                allele_types_total = ['A', 'T', 'G', 'C', 'N', '+' + str(length_of_inserted_bases) + str(inserted_bases_forward[0]), '-' + str(length_of_deleted_bases) + str(deleted_bases_forward[0])]
        
        # the potential allele type
        max_index = allele_count.index(max(allele_count))
        
        # ignore the indels of length larger than 50 and move forward
        if max_index == 5 and length_of_inserted_bases > 50:
            pass
        
        elif max_index == 4:
            pass 
       
        elif max_index == 6 and length_of_deleted_bases > 50:
            pass
        
        else:
            if (0.1 <= (allele_forward_count[max_index]/max(allele_count)) <= 0.90 and (max(allele_count)/int(line_content[3])) >= m and
               small_variant_count(line_content[9]) < 2 and small_variant_count(line_content[9])/int(line_content[8]) < 0.03):
                # keep only the genomic sites with no strand bias, allele frequency larger than 0.075 and weak evidence in normal
                Cs.write(str(line[0]) + '\t' + str(line_content[0]) + '\t' + str(line_content[1]) + '\t' + str(line_content[2]) + '\t'  
                         + str(allele_types_total[max_index]) + '\t' + str(line_content[3]) + '\t' + str(max(allele_count)) + '\n')                        
            else:
                pass                              
    else:
        pass

def main(args):
    with xopen(args.Tumor_Normal_mpileup, 'rt') as TN, open(args.Candidate_somatic_sites, 'wt') as Cs:
        combi = pairwise(enumerate(TN))
        
        for line, line_pair in combi:
            # look at a genomic site and its next number_of_columns site
            line_content = line[1].rstrip('\n').split('\t')
            line_pair_content = line_pair[1].rstrip('\n').split('\t')
            
            if line_content[0] == line_pair_content[0] and (int(line_pair_content[1]) - int(line_content[1]) == number_of_columns):
                process_line(line_content, line, Cs)    
            else:
                # ignore the genomic site if its next number_of_columns site does not exist
                next(itertools.islice(combi, 2*number_of_columns-1, 2*number_of_columns-1), None)         

if __name__ == '__main__':
    start_time = time.time()
    main(args)
    print('--- %s seconds ---' %(time.time() - start_time))
