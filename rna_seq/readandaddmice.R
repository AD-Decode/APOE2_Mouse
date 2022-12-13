a=read.delim('/Users/ali/Desktop/may/rnarpackage/Mouse_RNASEQNormalizedCounts.txt',sep="\t", header = T, row.names = NULL)

library(biomaRt)
mart <- useMart('ensembl', dataset = 'mmusculus_gene_ensembl')

colnames(a)[1]='gene_id'
lookup=a$gene_id

tabtemp=getBM(
  attributes = c(
    'ensembl_gene_id',
    'ensembl_transcript_id', 
    'entrezgene_id', 'mgi_symbol'),
  filters = 'ensembl_gene_id',
  values = lookup,
  mart = mart)


a$sym=NA
notfound=0
k=0
l=0
for (i in 1: length(a$sym) ) {
 tempindex=which(a[i,1] ==tabtemp[,1] )
 tempindex=tempindex[1]
 a$sym[i]=tabtemp[tempindex,4]
 if (is.na(tabtemp[tempindex,3])) { 
   a$sym[i]=a[i,1] ;
   notfound=c(notfound,a[i,1]);
   cat(i,'th out of',length(a$sym),'with gene',a$sym[i], a[i,1]  ,'\n'  ); k=k+1}
 else if ( nchar(tabtemp[tempindex,3])==0 ) {  
   a$sym[i]=a[i,1] ;
   foundbutempty=c(foundbutempty,a[i,1]);
   cat(i,'th out of',length(a$sym),'with gene',a$sym[i], a[i,1]  ,'\n'  ); l=l+1
 }
  # cat(i,'th out of',length(a$sym),'with gene',a$sym[i] ,'\n' )
}

sum(is.na(a))
sum(nchar(a)==0 )

write.table(a, file = "micewithsymbol.txt", sep = "\t", col.names = TRUE)
b=read.delim('/Users/ali/Desktop/may/rnarpackage/micewithsymbol.txt',sep="\t", header = T)
sum(is.na(b$sym))
