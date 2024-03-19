
##name='AML'
##type='all' 'snp' 'indel'
##caller='deepssv'
library(PRROC)

save_path=paste("D:/method/transformer/figure_data/figure2/figure/roc/figure/",name,'_',type,'.pdf',sep='')
pdf(file=save_path, height=5, width=5)
par(mfrow=c(1,1), mar=c(3.2,3.2,1.3,0.5))

path='D:/method/transformer/figure_data/figure2/figure/roc/data/data/'




pile <- read.delim(paste(path,'deepssv/deepssv_',name,'_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr <- pr.curve(scores.class0= pile[,1], weights.class0=pile[,2], curve =TRUE)
plot(pr, auc.main=FALSE, legend =FALSE, xlab='', ylab='', main=paste(name,type,sep=''), colo='#7e2065', add = FALSE)
pr

pile1 <- read.delim(paste(path,'mutect2/mutect2_',name,'_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr1 <- pr.curve(scores.class0= pile1[,1], weights.class0=pile1[,2], curve =TRUE)
plot(pr1, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#F6BBC6', add = TRUE)
pr1

pile2 <- read.delim(paste(path,'neusomatic/neusomatic_',name,'_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr2 <- pr.curve(scores.class0= pile2[,1], weights.class0=pile2[,2], curve =TRUE)
plot(pr2, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#C9A77C', add = TRUE)
pr2

pile3 <- read.delim(paste(path,'strelka2/strelka2_',name,'_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr3 <- pr.curve(scores.class0= pile3[,1], weights.class0=pile3[,2], curve =TRUE)
plot(pr3, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#ffff00', add = TRUE)
pr3




pile63 <- read.delim(paste(path,'varscan2/varscan2_',name,'_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr63 <- pr.curve(scores.class0= pile63[,1], weights.class0=pile63[,2], curve =TRUE)
plot(pr63, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#41ae3c', add = TRUE, cex.axis=1.5)
pr63


pile645 <- read.delim(paste(path,'transssv/transssv_',name,'_05_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr645 <- pr.curve(scores.class0= pile645[,1], weights.class0=pile645[,2], curve =TRUE)
plot(pr645, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#4169E1', add = TRUE, cex.axis=1.5)
pr645


pile6415 <- read.delim(paste(path,'transssv/transssv_',name,'_15_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr6415 <- pr.curve(scores.class0= pile6415[,1], weights.class0=pile6415[,2], curve =TRUE)
plot(pr6415, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#708090', add = TRUE, cex.axis=1.5)
pr6415

pile64 <- read.delim(paste(path,'transssv/transssv_',name,'_03_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr64 <- pr.curve(scores.class0= pile64[,1], weights.class0=pile64[,2], curve =TRUE)
plot(pr64, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#008B8B', add = TRUE, cex.axis=1.5)
pr64


pile62 <- read.delim(paste(path,'transssv/transssv_',name,'_07_',type,'.txt',sep=''), header=FALSE, sep="\t", quote="", dec=NULL)
pr62 <- pr.curve(scores.class0= pile62[,1], weights.class0=pile62[,2], curve =TRUE)
plot(pr62, auc.main=FALSE, legend =FALSE, xlab='', ylab='', colo='#E71F19', add = TRUE, cex.axis=1.5)
pr62



mtext("Recall", side =1, line =2)
mtext("Precision", side =2, line =2)


legend(0.4,0.47,col=c('#008B8B','#4169E1','#E71F19', '#708090','#7e2065', '#F6BBC6', '#C9A77C', '#41ae3c','#28A9A1'),lty=c(1, 1), c("TransSSV-03","TransSSV-05","TransSSV-07","TransSSV-15","DeepSSV","Mutect2","NeuSomatic", "VarScan2",'Strelka2'),cex=1)


dev.off()
