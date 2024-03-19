##
##
##caller:transssv
##name='AML'
##file_path='figure/roc/data/transssv/AML.vcf'
##candidate_path='../../../vcf/true_from_paper/filter_candidate/'
##number='07'




setwd('D:/method/transformer/figure_data/figure2')

candidate <- read.table(candidate_path,sep='\t')



file_trans <- read.table(paste(file_path,name,'_',number,'.vcf',sep=''),sep='\t')

file_trans <- select(file_trans,V1,V2,V6)
colnames(file_trans) <- c('chr','site','value')
file_trans$name <- paste(file_trans$chr,file_trans$site,sep='+')


candidate$name <- paste(candidate$V2,candidate$V3,sep='+')
candidate <- select(candidate,V4,V7,name)
colnames(candidate) <- c('label','state','name')

all <- merge(file_trans,candidate,by='name')

all_1 <- select(all,value,label,state)

all_indel <- filter(all_1,state=='i')
all_2 <- select(all_indel,value,label)

write.table(all_2,paste('figure/roc/data/data/transssv_',name,'_',number,'_indel.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)


all_snp <- filter(all_1,state=='s')
all_3 <- select(all_snp,value,label)
write.table(all_3,paste('figure/roc/data/data/transssv_',name,'_',number,'_snp.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)


all_4 <- select(all_1,value,label)

write.table(all_4,paste('figure/roc/data/data/transssv_',name,'_',number,'_all.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)


cat('\n','indel\n',dim(all_indel)[1])
cat('\n','snp\n',dim(all_snp)[1])
cat('\n','all\n',dim(all_data)[1])
cat('\n','indel_tp\n',dim(filter(all_indel,label=='1'))[1])
cat('\n','snp_tp\n',dim(filter(all_snp,label=='1'))[1])