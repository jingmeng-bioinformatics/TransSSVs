##
##
##caller:deepssv
##name='AML'
##file_path='figure/roc/data/deepssv/AML.vcf'
##candidate_path='../../../vcf/true_from_paper/filter_candidate/AML_v0_snp_indel.txt'


setwd('D:/method/transformer/figure_data/figure2')
file_deepssv <- read.table(file_path,sep='\t')
file_deepssv <- select(file_deepssv,V1,V2,V6)
colnames(file_deepssv) <- c('chr','site','value')
file_deepssv$name <- paste(file_deepssv$chr,file_deepssv$site,sep='+')
candidate <- read.table(candidate_path,sep='\t')

candidate$name <- paste(candidate$V2,candidate$V3,sep='+')
candidate <- select(candidate,V4,V7,name)
colnames(candidate) <- c('label','state','name')

all <- merge(file_deepssv,candidate,by='name')

all_1 <- select(all,value,label,state)

all_indel <- filter(all_1,state=='i')
all_2 <- select(all_indel,value,label)
write.table(all_2,paste('figure/roc/data/data/deepssv_',name,'_indel.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)

all_snp <- filter(all_1,state=='s')
all_3 <- select(all_snp,value,label)
write.table(all_3,paste('figure/roc/data/data/deepssv_',name,'_snp.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)


all_4 <- select(all_1,value,label)
write.table(all_4,paste('figure/roc/data/data/deepssv_',name,'_all.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)