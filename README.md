# TransSSVs

Accurate detection of somatic small variants with attention-based deep neural networks

The BAM files of tumor and normal samples are first converted into a mixed pileup file where the candidate somatic sites are identified. Then, the mapping information that corresponds to the reference and variant alleles in the tumor and normal samples is extracted and encoded. Next, the intersite interactions in the context sequence are captured by the multi-head attention-based neural network to obtain a new feature representation of the genomic context. Then, the new feature representation is used to predict the somatic state of the candidate somatic sites. Finally, potential somatic small variants determined by the TransSSVs model are generated in the variant call format (VCF).

TransSSVs was tested on Debian GNU/Linux 11 (bullseye) and requires Python 3.

Prerequisites
----------
+ samtools 1.8
```
wget https://github.com/samtools/samtools/releases/download/1.8/samtools-1.8.tar.bz2
tar jxvf samtools-1.8.tar.bz2
cd samtools-1.8
./configure --prefix=/where/to/install
make
make install
export PATH=/where/to/install/bin:$PATH

```
 <br>

Python 3.7 and the following Python packages must be installed:
+ TensorFlow 2.7.0 
+ sklearn 0.19.2
+ pandas 1.3.5
+ numpy 1.21.6
+ keras 2.7.0

You can install these packages using anaconda/miniconda :
```
conda install tensorflow=2.7
conda install scikit-learn=0.19.2
conda install pandas=1.3.5
conda install numpy=1.21.6
conda install keras=2.7
```

Working demonstration
----------
<br>

![image](https://github.com/jingmeng-bioinformatics/TransSSVs/assets/35085665/d08371a3-9d2f-4185-b17a-8fc3d986a61c)




Getting started
----------

<br>
1. Run samtools (tested version: 1.8) to convert tumor and normal BAM files to a mixed pileup file required by TransSSVs:

```
samtools mpileup -B -d 100 \
-f /path/to/ref.fasta [-l] [-r] -q 10 -O -s \
-a /path/to/tumor.bam /path/to/normal.bam | \
gzip [-9] > /path/to/mixed_pileup_file
```
+ It spent about __23h__ to generate the mileup-file(__310.10GB__) of __MB__ dataset (tumor: __104.08GB__, normal: __98.14GB__) on the centos 7 with __1 cpu__ and __2G memory__.
+ To reduce the storage, you can set the compression ratio by gzip -9.

<br>
Note: For the case of applying TransSSVs on a part of the whole genome, increase the BED entry by n (the number of flanking genomic sites to the left or right of the candidate somatic site) base pairs in each direction, and specify the genomic region via the option -l or -r.
<br>

<br>
2. Run identi_candi_sites.py to identify candidate somatic small variants from the mixed pileup file: 

```
python3 identi_candi_sites.py \
 --Tumor_Normal_mpileup /path/to/mixed_pileup_file \
 --Candidate_somatic_sites /path/to/candidate_sites 
```

+ It spent about __13h__ to generate the candidate-file(__3MB__) of __MB__ genome on the centos 7 with __1 cpu__ and __2G memory__.



<br>
3. Run mapping_infor_candi_sites.py to create a file with mapping information for candidate somatic small variant sites as input for trained TransSSVs, or to create a file with mapping information for validated somatic sites:

```
python3 mapping_infor_candi_sites.py \
--Tumor_Normal_mpileup /path/to/mixed_pileup_file \
--Candidate_somatic_sites /path/to/candidate_sites \
--number_of_columns N \ #(defult=7)
--path /path/to/save \
--filename filename
```
number_of_columns: the number of flanking genomic sites to the left or right of the candidate somatic site(defult=7)

+ The resource consumption of generating a file with mapping information for __MB__ genome:

```
number_of_columns      time(h)      disk_usage(MB)    running_mem(GB)    number_of_running_cpu      number_of_candaidate_site
3                       3.97         121.83              2                     1                          87736
4                       4.38         156.63              2                     1                          87736
5                       4.84         191.44              2                     1                          87736
6                       4.99         226.25              2                     1                          87736
7                       5.30         261.06              2                     1                          87736
10                      6.05         365.48              2                     1                          87736
15                      8.40         539.51              2                     1                          87736
20                      10.42        713.55              4                     1                          87736
30                      13.87        1064.96             4                     1                          87736
40                      17.60        1413.12             4                     1                          87736
50                      22.20        1761.28             4                     1                          87736
60                      41.87        2109.44             4                     1                          87736
```
<br>
4.1 Run model_train.py to train your own TransSSVs model:

```
CUDA_VISIBLE_DEVICES='' python3 model_train.py \
--input_dir /path/to/input \
--filename filename \
--vaild_dir /path/to/vaild_dir
```
<br>
4.2 Run model_train.py to fine-tune your own TransSSVs model based on our pre-trained-model:

```
CUDA_VISIBLE_DEVICES='' python3 model_train.py \
--input_dir /path/to/input \
--filename filename \
--vaild_dir /path/to/vaild_dir \
--indicator fine_tune \
--weights /TransSSVs/pre_trained_model/pre_trained_model 
```


<br>
5. Run model_infer.py to predict somatic small variants using our pre-trained model:

```
CUDA_VISIBLE_DEVICES='' python3 model_infer.py \
--weights /TransSSVs/pre_trained_model/pre_trained_model \
--input_dir /path/to/input \
--filename filename \
--save_dir /results/TransSSVs
```

```
number_of_columns      time(s)          running_mem(GB)    number_of_running_cpu      number_of_candaidate_site
3                       66                   4                     1                          87736
4                       58                   4                     1                          87736
5                       64                   4                     1                          87736
6                       70                   4                     1                          87736
7                       74                   4                     1                          87736
10                      86                   4                     1                          87736
15                      111                  4                     1                          87736
20                      137                  4                     1                          87736
30                      186                  4                     1                          87736
40                      253                  4                     1                          87736
50                      296                  4                     1                          87736
60                      449                  4                     1                          87736
```





<br>
6. Run write_vcf.py to generate the vcf file:

```
python3 write_vcf.py \
--vcf_file /path/to/vcf_file \
--pred_class /results/TransSSVs/y_pred_all.txt \
--Candidate_somatic_sites /path/to/candidate_sites
```


<br>

Example of the Validated_labels file for validated sites with labels (1: somatic site, 0: non-somatic site):

```
chr1    790265    0
chr1    1595272   1
chr1    2312314   1
chr1    5006153   0
```

Testing time
----------






Please help us improve TransSSVs by reporting bugs or ideas on how to make things better. You can submit an issue or send us an email.<br>

Jing Meng, Jiangyuan Wang<br>

jing.mengrabbit@outlook.com<br>
wjy_bazi@hotmail.com<br>

