## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval = FALSE-------------------------------------------------------------
#  library(SpaCOAP)

## ----eval = FALSE-------------------------------------------------------------
#  width <- 20; height <- 30
#  n <- width*height
#  p=500
#  q = 5; d <- 40; k <- 3; r <- 3
#  bandwidth <- 1
#  rho<-   c(8,0.6)
#  sigma2_eps=1
#  datlist <- gendata_spacoap(seed=1, width=width, height = height,
#                             p=p, q=q, d=d, k=k, rank0 = r, bandwidth=1,
#                             eta0 = 0.5, rho=rho, sigma2_eps=sigma2_eps)
#  X_count <- datlist$X; H <- datlist$H; Z <- datlist$Z
#  F0 <- datlist$F0; B0 <- datlist$B0
#  bbeta0 <- datlist$bbeta0; alpha0 <- datlist$alpha0
#  Adj_sp <- SpaCOAP:::getneighbor_weightmat(datlist$pos, 1.1,  bandwidth)

## ----eval = FALSE-------------------------------------------------------------
#  reslist <- SpaCOAP(X_count,Adj_sp, H, Z = Z, rank_use = r, q=q)
#  str(reslist)

## ----eval = FALSE-------------------------------------------------------------
#  library(ggplot2)
#  dat_iter <- data.frame(iter=1:length(reslist$ELBO_seq[-1]), ELBO=reslist$ELBO_seq[-1])
#  ggplot(data=dat_iter, aes(x=iter, y=ELBO)) + geom_line() + geom_point() + theme_bw(base_size = 20)
#  

## ----eval = FALSE-------------------------------------------------------------
#  norm1_vec <- function(x) mean(abs(x))
#  trace_statistic_fun <- function(H, H0){
#  
#    tr_fun <- function(x) sum(diag(x))
#    mat1 <- t(H0) %*% H %*% qr.solve(t(H) %*% H) %*% t(H) %*% H0
#  
#    tr_fun(mat1) / tr_fun(t(H0) %*% H0)
#  
#  }

## ----eval = FALSE-------------------------------------------------------------
#  metricList <- list()
#  metricList$SpaCOAP <- list()
#  metricList$SpaCOAP$F_tr <- trace_statistic_fun(reslist$F, F0)
#  metricList$SpaCOAP$B_tr <- trace_statistic_fun(reslist$B, B0)
#  metricList$SpaCOAP$alpha_norm1 <- norm1_vec(reslist$alpha- alpha0)/mean(abs(alpha0))
#  metricList$SpaCOAP$beta_norm1<- norm1_vec(reslist$bbeta- bbeta0)/mean(abs(bbeta0))
#  metricList$SpaCOAP$time <- reslist$time_use

## ----eval = FALSE-------------------------------------------------------------
#  library(COAP)
#  tic <- proc.time()
#  res_coap <- RR_COAP(X_count, Z = cbind(Z, H), rank_use= k+r, q=5, epsELBO = 1e-9)
#  toc <- proc.time()
#  time_coap <- toc[3] - tic[3]
#  metricList$COAP$F_tr <- trace_statistic_fun(res_coap$H, F0)
#  metricList$COAP$B_tr <- trace_statistic_fun(res_coap$B, B0)
#  alpha_coap <- res_coap$bbeta[,1:k]
#  beta_coap <- res_coap$bbeta[,(k+1):(k+d)]
#  metricList$COAP$alpha_norm1 <- norm1_vec(alpha_coap- alpha0)/mean(abs(alpha0))
#  metricList$COAP$beta_norm1 <- norm1_vec(beta_coap- bbeta0)/mean(abs(bbeta0))
#  metricList$COAP$time <- time_coap

## ----eval = FALSE-------------------------------------------------------------
#  
#  PLNPCA_run <- function(X_count, covariates, q,  Offset=rep(1, nrow(X_count)), workers=NULL,
#                         maxIter=10000,ftol_rel=1e-8, xtol_rel= 1e-4){
#    require(PLNmodels)
#    if(!is.null(workers)){
#      future::plan("multisession", workers = workers)
#    }
#    if(!is.character(Offset)){
#      dat_plnpca <- prepare_data(X_count, covariates)
#      dat_plnpca$Offset <- Offset
#    }else{
#      dat_plnpca <- prepare_data(X_count, covariates, offset = Offset)
#    }
#  
#    d <- ncol(covariates)
#    #  offset(log(Offset))+
#    formu <- paste0("Abundance ~ 1 + offset(log(Offset))+",paste(paste0("V",1:d), collapse = '+'))
#    control_use  <- list(maxeval=maxIter, ftol_rel=ftol_rel, xtol_rel= ftol_rel)
#    control_main <- PLNPCA_param(
#      backend = "nlopt",
#      trace = 1,
#      config_optim = control_use,
#      inception = NULL
#    )
#  
#    myPCA <- PLNPCA(as.formula(formu), data = dat_plnpca, ranks = q,  control = control_main)
#  
#    myPCA1 <- getBestModel(myPCA)
#    myPCA1$scores
#  
#    res_plnpca <- list(PCs= myPCA1$scores, bbeta= myPCA1$model_par$B,
#                       loadings=myPCA1$model_par$C)
#  
#    return(res_plnpca)
#  }
#  
#  tic <- proc.time()
#  res_plnpca <- PLNPCA_run(X_count, cbind(Z[,-1],H), q=q)
#  toc <- proc.time()
#  time_plnpca <- toc[3] - tic[3]
#  
#  metricList$PLNPCA$F_tr <- trace_statistic_fun(res_plnpca$PCs, F0)
#  metricList$PLNPCA$B_tr <- trace_statistic_fun(res_plnpca$loadings, B0)
#  alpha_plnpca <- t(res_plnpca$bbeta[1:k,])
#  beta_plnpca <- t(res_plnpca$bbeta[(k+1):(k+d),])
#  metricList$PLNPCA$alpha_norm1 <- norm1_vec(alpha_plnpca- alpha0)/mean(abs(alpha0))
#  metricList$PLNPCA$beta_norm1 <- norm1_vec(beta_plnpca- bbeta0)/mean(abs(bbeta0))
#  metricList$PLNPCA$time <- time_plnpca

## ----eval = FALSE-------------------------------------------------------------
#  ## MRRR
#  ## Compare with MRRR
#  mrrr_run <- function(Y, X, rank0, q=NULL, family=list(poisson()),
#                       familygroup=rep(1,ncol(Y)), epsilon = 1e-4, sv.tol = 1e-2,
#                       maxIter = 2000, trace=TRUE, truncflag=FALSE, trunc=500){
#    # epsilon = 1e-4; sv.tol = 1e-2; maxIter = 30; trace=TRUE
#    # Y <- X_count; X <- cbind(Z, H); rank0 = r + ncol(Z)
#  
#    require(rrpack)
#  
#    n <- nrow(Y); p <- ncol(Y)
#  
#    if(!is.null(q)){
#      rank0 <- rank0+q
#      X <- cbind(X, diag(n))
#    }
#    if(truncflag){
#      ## Trunction
#      Y[Y>trunc] <- trunc
#  
#    }
#  
#    svdX0d1 <- svd(X)$d[1]
#    init1 = list(kappaC0 = svdX0d1 * 5)
#    offset = NULL
#    control = list(epsilon = epsilon, sv.tol = sv.tol, maxit = maxIter,
#                   trace = trace, gammaC0 = 1.1, plot.cv = TRUE,
#                   conv.obj = TRUE)
#    fit.mrrr <- mrrr(Y=Y, X=X[,-1], family = family, familygroup = familygroup,
#                     penstr = list(penaltySVD = "rankCon", lambdaSVD = 1),
#                     control = control, init = init1, maxrank = rank0)
#  
#    return(fit.mrrr)
#  }
#  tic <- proc.time()
#  res_mrrr <- mrrr_run(X_count, cbind(Z,H), r+ncol(Z), q=q, truncflag= TRUE, trunc=1e4)
#  toc <- proc.time()
#  time_mrrr <- toc[3] - tic[3]
#  

## ----eval = FALSE-------------------------------------------------------------
#  hbbeta_mrrr <-t(res_mrrr$coef[1:ncol(cbind(Z,H)), ])
#  Theta_hb <- (res_mrrr$coef[(ncol(cbind(Z,H))+1): (nrow(cbind(Z,H))+ncol(cbind(Z,H))), ])
#  svdTheta <- svd(Theta_hb, nu=q, nv=q)
#  metricList$MRRR$F_tr <- trace_statistic_fun(svdTheta$u, F0)
#  metricList$MRRR$B_tr <- trace_statistic_fun(svdTheta$v, B0)
#  alpha_mrrr <- hbbeta_mrrr[,1:k]
#  beta_mrrr <- hbbeta_mrrr[,(k+1):(k+d)]
#  metricList$MRRR$alpha_norm1 <- norm1_vec(alpha_mrrr- alpha0)/mean(abs(alpha0))
#  metricList$MRRR$beta_norm1 <- norm1_vec(beta_mrrr- bbeta0)/mean(abs(bbeta0))
#  metricList$MRRR$time <- time_mrrr

## ----eval =FALSE--------------------------------------------------------------
#  ## FAST
#  fast_run <- function(X_count, Adj_sp, q, verbose=TRUE, epsELBO=1e-8){
#    require(ProFAST)
#  
#    reslist <- FAST_run(XList = list(X_count),
#                        AdjList = list(Adj_sp), q = q, fit.model = 'poisson',
#                        verbose=verbose, epsLogLik=epsELBO)
#    reslist$hV <- reslist$hV[[1]]
#    return(reslist)
#  }
#  tic <- proc.time()
#  res_fast <- fast_run(X_count, Adj_sp, q=q, verbose=TRUE, epsELBO=1e-8)
#  toc <- proc.time()
#  time_fast <- toc[3] - tic[3]
#  metricList$FAST$F_tr <- trace_statistic_fun(res_fast$hV, F0)
#  metricList$FAST$B_tr <- trace_statistic_fun(res_fast$W, B0)
#  metricList$FAST$time <- time_fast
#  

## ----eval = FALSE-------------------------------------------------------------
#  list2vec <- function(xlist){
#    nn <- length(xlist)
#    me <- rep(NA, nn)
#    idx_noNA <- which(sapply(xlist, function(x) !is.null(x)))
#    for(r in idx_noNA) me[r] <- xlist[[r]]
#    return(me)
#  }
#  
#  dat_metric <- data.frame(Tr_F = sapply(metricList, function(x) x$F_tr),
#                           Tr_B = sapply(metricList, function(x) x$B_tr),
#                           err_alpha =list2vec(lapply(metricList, function(x) x$alpha_norm1)),
#                           err_beta = list2vec(lapply(metricList, function(x) x$beta_norm1)),
#                           time = sapply(metricList, function(x) x$time),
#                           Method = names(metricList))
#  dat_metric$Method <- factor(dat_metric$Method, levels=dat_metric$Method)

## ----eval = FALSE, fig.width=9, fig.height=6----------------------------------
#  library(cowplot)
#  p1 <- ggplot(data=subset(dat_metric, !is.na(Tr_B)), aes(x= Method, y=Tr_B, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL) + theme_bw(base_size = 16)
#  p2 <- ggplot(data=subset(dat_metric, !is.na(Tr_F)), aes(x= Method, y=Tr_F, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  p3 <- ggplot(data=subset(dat_metric, !is.na(err_alpha)), aes(x= Method, y=err_alpha, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  p4 <- ggplot(data=subset(dat_metric, !is.na(err_beta)), aes(x= Method, y=err_beta, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  plot_grid(p1,p2,p3, p4, nrow=2, ncol=2)

## ----eval = FALSE-------------------------------------------------------------
#  
#  res1 <- chooseParams(X_count, Adj_sp, H, Z, verbose=FALSE)
#  
#  print(c(q_true=q, q_est=res1['hq']))
#  print(c(r_true=r, r_est=res1['hr']))

## -----------------------------------------------------------------------------
sessionInfo()

