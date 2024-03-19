##
##
##caller:mutect2
##name='AML'
##file_path='figure/roc/data/other/'
##candidate_path='../../../vcf/true_from_paper/filter_candidate/AML_v0_snp_indel.txt'




setwd('D:/method/transformer/figure_data/figure2')
file_mutect2_snp <- read.table(paste(file_path,name,'_snp_2.vcf',sep=''),sep='\t')

file_mutect2_snp[,12]=gsub(".*TLOD=","", file_mutect2_snp[, 8])

file_mutect2_snp <- select(file_mutect2_snp,V1,V2,V12)


file_mutect2_snp$xuhao <- row.names(file_mutect2_snp)
colnames(file_mutect2_snp) <- c('chr','site','value','xuhao')
file_mutect2_snp$name <- paste(file_mutect2_snp$chr,file_mutect2_snp$site,sep='+')
candidate <- read.table(candidate_path,sep='\t')

candidate$name <- paste(candidate$V2,candidate$V3,sep='+')
candidate <- select(candidate,V4,V7,name)
colnames(candidate) <- c('label','state','name')

candidate_tp <- filter(candidate,label=='1')

file_mutect2_snp$label <-'0'

file_mutect2_snp[filter(file_mutect2_snp,name %in% candidate_tp$name)$xuhao,]$label <- '1'

snp_data <- select(file_mutect2_snp,value,label)

write.table(snp_data,paste('figure/roc/data/data/mutect2_',name,'_snp.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)

file_mutect2_indel <- read.table(paste(file_path,name,'_indel_2.vcf',sep=''),sep='\t')

file_mutect2_indel[,12]=gsub(".*TLOD=","", file_mutect2_indel[, 8])

file_mutect2_indel <- select(file_mutect2_indel,V1,V2,V12)


file_mutect2_indel$xuhao <- row.names(file_mutect2_indel)
colnames(file_mutect2_indel) <- c('chr','site','value','xuhao')
file_mutect2_indel$name <- paste(file_mutect2_indel$chr,file_mutect2_indel$site,sep='+')
candidate <- read.table(candidate_path,sep='\t')

candidate$name <- paste(candidate$V2,candidate$V3,sep='+')
candidate <- select(candidate,V4,V7,name)
colnames(candidate) <- c('label','state','name')

candidate_tp <- filter(candidate,label=='1')

file_mutect2_indel$label <-'0'

file_mutect2_indel[filter(file_mutect2_indel,name %in% candidate_tp$name)$xuhao,]$label <- '1'

indel_data <- select(file_mutect2_indel,value,label)


write.table(indel_data,paste('figure/roc/data/data/mutect2_',name,'_indel.txt',sep=''),row.names=F,col.names=F,sep='\t',quote=F)


cat('\n','indel\n',dim(indel_data)[1])
cat('\n','snp\n',dim(snp_data)[1])

