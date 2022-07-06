

import argparse
import re
import time

parser = argparse.ArgumentParser()

parser.add_argument('--vcf_file', required=True, metavar='file', help='vcf file')
parser.add_argument('--pred_class', required=True, metavar='file', help='a file that called y_pred_all')
parser.add_argument('--Candidate_somatic_sites', required=True, metavar='file', help='identified candidate somatic sites')

args = parser.parse_args()


def main(args):
    with open(args.Candidate_somatic_sites, 'rt') as Cs, open(args.pred_class, 'rt') as pc, open(args.vcf_file, 'wt') as vcf:
        # create a vcf file for predictions
        enume_Cs = enumerate(Cs)
        enume_pc = enumerate(pc)          
        # the threshold used to decide if a candiate site is a somatic site
        t = 0.5
        
        # write meta-information lines
        vcf.write(str('##fileformat=VCFv4.2') + '\n')
        vcf.write(str('##phasing=none') + '\n')
        vcf.write(str('##ALT=<ID=INS,Description="Insertion">') + '\n')
        vcf.write(str('##ALT=<ID=DEL,Description="Deletion">') + '\n')
        # INFO field
        vcf.write(str('##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth in tumor; some reads may have been filtered">') + '\n')
        vcf.write(str('##INFO=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">') + '\n')
        vcf.write(str('##INFO=<ID=AD,Number=1,Type=Integer,Description="Depth of variant allele in tumor">') + '\n')
        # FORMAT field
        vcf.write(str('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">') + '\n')

        # write header line
        vcf.write(str('#CHROM') + '\t' + str('POS') + '\t' + str('ID') + '\t' + str('REF') + '\t' + str('ALT') +
            '\t' + str('QUAL') + '\t' + str('FILTER') + '\t' + str('INFO') + '\t' + str('FORMAT') + '\n')
        
        # write data lines
        for line, line_pair in zip(enume_Cs, enume_pc):
            line_content = line[1].rstrip('\n').split('\t')
            line_pair_content = line_pair[0].rstrip('\n').split('\t')
            
            if float(line_pair_content[1]) >= t: 

                # write deletions
                if re.search('\-[0-9]+', line_content[4]):
                    line_content[4] = re.sub('\-[0-9]+', '', line_content[4])
                    vcf.write(line_content[1] + '\t' + line_content[2] + '\t' + str('.') + '\t' + str(line_content[3].upper()) + 
                        str(line_content[4].upper()) + '\t' + str(line_content[3].upper()) + '\t' + str(line_pair_content[1]) + '\t' + str('PASS') + 
                        '\t' + str('DP') + str('=') + str(line_content[5]) + str(';') + str('VAF') + str('=') + str(int(line_content[6])/int(line_content[5])) +
                        str(';') + str('AD') + str('=') + str(line_content[6]) + '\t')
                    
                    # write genotype information
                    if int(line_content[6])/int(line_content[5]) <= 0.5:
                        vcf.write(str('GT') + '\t' + str('0/1') + '\n')
                    else:
                        vcf.write(str('GT') + '\t' + str('1/0') + '\n')
                
                # write insertions
                elif re.search('\+[0-9]+', line_content[4]):
                    line_content[4] = re.sub('\+[0-9]+', '', line_content[4])
                    vcf.write(line_content[1] + '\t' + line_content[2] + '\t' + str('.') + '\t' + str(line_content[3].upper()) + 
                        '\t' + str(line_content[3].upper()) + str(line_content[4].upper()) + '\t' + str(line_pair_content[1]) + '\t' + str('PASS') + 
                        '\t' + str('DP') + str('=') + str(line_content[5]) + str(';') + str('VAF') + str('=') + str(int(line_content[6])/int(line_content[5])) +
                        str(';') + str('AD') + str('=') + str(line_content[6]) + '\t')
                    
                    # write genotype information
                    if int(line_content[6])/int(line_content[5]) <= 0.5:
                        vcf.write(str('GT') + '\t' + str('0/1') + '\n')
                    else:
                        vcf.write(str('GT') + '\t' + str('1/0') + '\n')

                # write SNVs
                else:
                    vcf.write(line_content[1] + '\t' + line_content[2] + '\t' + str('.') + '\t' + str(line_content[3].upper()) + 
                        '\t' + str(line_content[4].upper()) + '\t' + str(line_pair_content[1]) + '\t' + str('PASS') + '\t' + 
                        str('DP') + str('=') + str(line_content[5]) + str(';') + str('VAF') + str('=') + str(int(line_content[6])/int(line_content[5])) +
                        str(';') + str('AD') + str('=') + str(line_content[6]) + '\t')
                    
                    # write genotype information
                    if int(line_content[6])/int(line_content[5]) <= 0.5:
                        vcf.write(str('GT') + '\t' + str('0/1') + '\n')
                    else:
                        vcf.write(str('GT') + '\t' + str('1/0') + '\n')

            else:
                pass

if __name__ == '__main__':
    main(args)