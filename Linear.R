rm(list=ls())
library(MASS)
library(caret)
library(glmnet)
library(mvtnorm)

n <- 1e5 # N=1e5
p <- 60 # 60, 80, 100 correspond to pn=60, 80, 100 in Section 4.1
xtype <- 1 # 1, 2, 3 correspond to setting (i), (ii), (iii) in Section 4.1
itr.max <- 500
r1 <- 400 # n0=400
seq_r=c(400,600,800,1000) # n=400, 600, 800, 1000
rho=0.1 # coefficient to protect the probabilities

X_AR1_covmatrix<- function(rho,p){
  #---------------------------Arguments----------------------------------------#
  # Purpose: This function is to produce the covariance matrix of the predictors.
  #
  # Input:
  #      rho: A parameter
  #      p: The dimension of the predictor vector. 
  #----------------------------------------------------------------------------#
  x_covmatrix<- matrix(rep(0,p^2),p,p)
  for (ii in 1:p)
  {
    for (jj in 1:p){
      x_covmatrix[ii,jj]<- rho^(abs(ii-jj))
    }
  }
  return(x_covmatrix)
}
X_CS_covmatrix<- function(rho,p){
  #---------------------------Arguments----------------------------------------#
  # Purpose: This function is to produce the covariance matrix of the predictors.
  #
  # Input:
  #      rho: A parameter
  #      p: The dimension of the predictor vector. 
  #----------------------------------------------------------------------------#
  x_covmatrix<- matrix(rep(rho,p^2),p,p)
  diag(x_covmatrix) <- 1
  return(x_covmatrix)
}
bt <- rep(1, p) #true value of beta_0

A1_final_result <- A1_another_result <- vector()
A2_final_result <- A2_another_result <- vector()
A3_final_result <- A3_another_result <- vector()


for (r2 in seq_r){
  #record the results for current n
  A1_beta_total.opt <-A1_beta_total.unif <- A1_beta_total.uw <- A1_beta_total.lev <- vector()
  A1_sd_total.opt <- A1_sd_total.unif <- A1_sd_total.uw <- A1_sd_total.lev <- vector()
  A1_cp_total.opt <- A1_cp_total.unif <- A1_cp_total.uw <- A1_cp_total.lev <-vector()
  
  A2_beta_total.opt <-A2_beta_total.unif <- A2_beta_total.uw <- A2_beta_total.lev <- vector()
  A2_sd_total.opt <- A2_sd_total.unif <- A2_sd_total.uw <- A2_sd_total.lev <- vector()
  A2_cp_total.opt <- A2_cp_total.unif <- A2_cp_total.uw <- A2_cp_total.lev <-vector()
  
  A3_beta_total.opt <-A3_beta_total.unif <- A3_beta_total.uw <- A3_beta_total.lev <- vector()
  A3_sd_total.opt <- A3_sd_total.unif <- A3_sd_total.uw <- A3_sd_total.lev <- vector()
  A3_cp_total.opt <- A3_cp_total.unif <- A3_cp_total.uw <- A3_cp_total.lev <-vector()
  
  for (k in 1:itr.max){
    cat(k,"\n")
    set.seed(k+2024)
    if (xtype==1){ #Setting (i)
      X <- rmvnorm(n,rep(0,p),X_CS_covmatrix(0.5,p))
    }
    if (xtype==2){ #Setting (ii)
      X <- rmvt(n,X_AR1_covmatrix(0.5,p),df=4)
    }
    if (xtype==3){ #Setting (iii)
      X <- rmvt(n,X_CS_covmatrix(0.5,p),df=10)
    }
    Y <- X %*% bt + rnorm(n, sd=2)
    
    #pilot estimator
    idx <- 1:n
    pi <- rep((r1/n),n)
    decision <- rbinom(n,rep(1,n),prob=pi)
    idx.simp <- idx[decision==1]
    X.simp <- X[idx.simp,]
    Y.simp <- Y[idx.simp]
    D.simp <- t(X.simp) %*% X.simp
    for (a in 1:3){
      if (a==1) { #Example (1)
        A <- diag(rep(1,p))
      }
      if (a==2) { #Example (2)
        A <- matrix(1,1,p)/sqrt(p)
      }
      if (a==3) { #Example (3)
        A <- matrix(0,1,p)
        A[1,1] <- 1
      }
      
      #optimal probabilities --- response-free
      Pi.L.tmp <- sqrt(rowSums((X %*% ginv(D.simp) %*% t(A))^2))
      Pi.L <- Pi.L.tmp/sum(Pi.L.tmp)
      Pi.L <- r2*Pi.L*(1-rho)+r2/n*rho
      Pi.L[which(Pi.L >1)]=1
      
      #####optimal subsampling --- weighted
      set.seed(k+2024)
      decision.opt <- rbinom(n,rep(1,n),prob=Pi.L)
      idx.opt <- idx[decision.opt==1]
      X.opt <- X[idx.opt,]
      Y.opt <- Y[idx.opt]
      Pi.L.opt <- Pi.L[idx.opt]
      pinv.opt <- 1/Pi.L.opt
      beta.opt <- as.vector(ginv(t(X.opt) %*% (pinv.opt *X.opt)) %*% t(X.opt) %*% (pinv.opt *Y.opt))
      
      #inference --- weighted
      D.opt <- t(X.opt) %*% (X.opt/Pi.L.opt)
      Vc.opt <- t(X.opt) %*% (X.opt/Pi.L.opt^2)
      V.opt <- A %*% ginv(D.opt) %*% (Vc.opt) %*% t(ginv(D.opt)) %*% t(A) * mean(c((Y.opt- X.opt %*% beta.opt)^2, (Y.simp- X.simp %*% beta.opt)^2))
      sd.opt <- sqrt(diag(V.opt))
      
      
      #####optimal subsampling --- unweighted
      X.uw <- X.opt
      Y.uw <- Y.opt
      Pi.L.uw <- Pi.L.opt
      beta.uw <- glm.fit(X.uw, Y.uw, family=gaussian(link=identity))$coefficients
      
      #inference --- unweighted
      gamma.uw <- t(X.uw) %*% X.uw
      Vc.uw <- ginv(gamma.uw)
      V.uw <- A %*% Vc.uw %*% t(A) * mean(c((Y.uw- X.uw %*% beta.uw)^2, (Y.simp- X.simp %*% beta.uw)^2))
      sd.uw <- sqrt(diag(V.uw))
      
      ######leverage score subsampling
      data <- data.frame(X, Y)
      fit <- lm(Y ~ X+0, data = data)
      leverage <- hatvalues(fit)
      Pi.lev <- leverage/p
      Pi.lev <- r2 * Pi.lev *(1-rho) + rho * r2/n
      Pi.lev[which(Pi.lev>1)] <- 1
      
      set.seed(k+2024)
      idx <- 1:n
      decision.lev <- rbinom(n,rep(1,n),prob=Pi.lev)
      idx.lev <- idx[decision.lev==1]
      X.lev <- X[idx.lev,]
      Y.lev <- Y[idx.lev]
      Pi.lev <- Pi.lev[idx.lev]
      pinv.lev <- 1/Pi.lev
      beta.lev <- as.vector(ginv(t(X.lev) %*% (pinv.lev *X.lev)) %*% t(X.lev) %*% (pinv.lev *Y.lev))
      
      #inference --- leverage
      D.lev <- t(X.lev) %*% (X.lev/Pi.lev) / n
      Vc.lev <- t(X.lev) %*% (X.lev/Pi.lev^2)/n^2
      V.lev <- A %*% ginv(D.lev) %*% Vc.lev %*% t(ginv(D.lev)) %*% t(A) * mean((Y.lev- X.lev %*% beta.lev)^2)
      sd.lev <- sqrt(diag(V.lev))
      
      #####uniform subsampling
      set.seed(k+2024)
      idx <- 1:n
      pi <- rep((r2/n),n)
      decision.unif <- rbinom(n,rep(1,n),prob=pi)
      idx.unif <- idx[decision.unif==1]
      X.unif <- X[idx.unif,]
      Y.unif <- Y[idx.unif]
      beta.unif <- glm.fit(X.unif, Y.unif, family=gaussian(link=identity))$coefficients
      #inference
      V.unif <- A %*% ginv(t(X.unif) %*% (X.unif)) %*% t(A) * mean((Y.unif-X.unif %*% beta.unif)^2)
      sd.unif <- sqrt(diag(V.unif))
      
      
      #results of current loop
      if (a==1){
        #optimal
        cp.opt <- as.vector(abs(A %*% beta.opt- A %*% bt) <= 1.96*sd.opt)
        A1_beta_total.opt <- rbind(A1_beta_total.opt,as.vector(A %*% beta.opt))
        A1_sd_total.opt <- rbind(A1_sd_total.opt,sd.opt)
        A1_cp_total.opt <- rbind(A1_cp_total.opt,cp.opt)
        
        #uniform
        cp.unif <- as.vector(abs(A %*% beta.unif- A %*% bt) <= 1.96*sd.unif)
        A1_beta_total.unif <- rbind(A1_beta_total.unif,as.vector(A %*% beta.unif))
        A1_sd_total.unif <- rbind(A1_sd_total.unif,sd.unif)
        A1_cp_total.unif <- rbind(A1_cp_total.unif,cp.unif)
        
        #uw
        cp.uw <- as.vector(abs(A %*% beta.uw- A %*% bt) <= 1.96*sd.uw)
        A1_beta_total.uw <- rbind(A1_beta_total.uw,as.vector(A %*% beta.uw))
        A1_sd_total.uw <- rbind(A1_sd_total.uw,sd.uw)
        A1_cp_total.uw <- rbind(A1_cp_total.uw,cp.uw)
        
        #leverage
        cp.lev <- as.vector(abs(A %*% beta.lev- A %*% bt) <= 1.96*sd.lev)
        A1_beta_total.lev <- rbind(A1_beta_total.lev,as.vector(A %*% beta.lev))
        A1_sd_total.lev <- rbind(A1_sd_total.lev,sd.lev)
        A1_cp_total.lev <- rbind(A1_cp_total.lev,cp.lev)
      }
      
      if (a==2){
        #optimal
        cp.opt <- as.vector(abs(A %*% beta.opt- A %*% bt) <= 1.96*sd.opt)
        A2_beta_total.opt <- rbind(A2_beta_total.opt,as.vector(A %*% beta.opt))
        A2_sd_total.opt <- rbind(A2_sd_total.opt,sd.opt)
        A2_cp_total.opt <- rbind(A2_cp_total.opt,cp.opt)
        
        #uniform
        cp.unif <- as.vector(abs(A %*% beta.unif- A %*% bt) <= 1.96*sd.unif)
        A2_beta_total.unif <- rbind(A2_beta_total.unif,as.vector(A %*% beta.unif))
        A2_sd_total.unif <- rbind(A2_sd_total.unif,sd.unif)
        A2_cp_total.unif <- rbind(A2_cp_total.unif,cp.unif)
        
        #uw
        cp.uw <- as.vector(abs(A %*% beta.uw- A %*% bt) <= 1.96*sd.uw)
        A2_beta_total.uw <- rbind(A2_beta_total.uw,as.vector(A %*% beta.uw))
        A2_sd_total.uw <- rbind(A2_sd_total.uw,sd.uw)
        A2_cp_total.uw <- rbind(A2_cp_total.uw,cp.uw)
        
        #leverage
        cp.lev <- as.vector(abs(A %*% beta.lev- A %*% bt) <= 1.96*sd.lev)
        A2_beta_total.lev <- rbind(A2_beta_total.lev,as.vector(A %*% beta.lev))
        A2_sd_total.lev <- rbind(A2_sd_total.lev,sd.lev)
        A2_cp_total.lev <- rbind(A2_cp_total.lev,cp.lev)
      }
      
      if (a==3){
        #optimal
        cp.opt <- as.vector(abs(A %*% beta.opt- A %*% bt) <= 1.96*sd.opt)
        A3_beta_total.opt <- rbind(A3_beta_total.opt,as.vector(A %*% beta.opt))
        A3_sd_total.opt <- rbind(A3_sd_total.opt,sd.opt)
        A3_cp_total.opt <- rbind(A3_cp_total.opt,cp.opt)
        
        #uniform
        cp.unif <- as.vector(abs(A %*% beta.unif- A %*% bt) <= 1.96*sd.unif)
        A3_beta_total.unif <- rbind(A3_beta_total.unif,as.vector(A %*% beta.unif))
        A3_sd_total.unif <- rbind(A3_sd_total.unif,sd.unif)
        A3_cp_total.unif <- rbind(A3_cp_total.unif,cp.unif)
        
        #uw
        cp.uw <- as.vector(abs(A %*% beta.uw- A %*% bt) <= 1.96*sd.uw)
        A3_beta_total.uw <- rbind(A3_beta_total.uw,as.vector(A %*% beta.uw))
        A3_sd_total.uw <- rbind(A3_sd_total.uw,sd.uw)
        A3_cp_total.uw <- rbind(A3_cp_total.uw,cp.uw)
        
        #leverage
        cp.lev <- as.vector(abs(A %*% beta.lev- A %*% bt) <= 1.96*sd.lev)
        A3_beta_total.lev <- rbind(A3_beta_total.lev,as.vector(A %*% beta.lev))
        A3_sd_total.lev <- rbind(A3_sd_total.lev,sd.lev)
        A3_cp_total.lev <- rbind(A3_cp_total.lev,cp.lev)
      }
    }
  }
  A <- diag(rep(1,p))
  #Bias
  A1_Bias.opt <- mean(colMeans(A1_beta_total.opt)-as.vector(A %*% bt))
  A1_Bias.unif <- mean(colMeans(A1_beta_total.unif)-as.vector(A %*% bt))
  A1_Bias.uw <- mean(colMeans(A1_beta_total.uw)-as.vector(A %*% bt))
  A1_Bias.lev <- mean(colMeans(A1_beta_total.lev)-as.vector(A %*% bt))

  #MSE
  A1_MSE.opt <- sum((colMeans(A1_beta_total.opt)-as.vector(A %*% bt))^2+apply(A1_beta_total.opt,2,sd)^2)
  A1_MSE.unif <- sum((colMeans(A1_beta_total.unif)-as.vector(A %*% bt))^2+apply(A1_beta_total.unif,2,sd)^2)
  A1_MSE.uw <- sum((colMeans(A1_beta_total.uw)-as.vector(A %*% bt))^2+apply(A1_beta_total.uw,2,sd)^2)
  A1_MSE.lev <- sum((colMeans(A1_beta_total.lev)-as.vector(A %*% bt))^2+apply(A1_beta_total.lev,2,sd)^2)

  #ACP
  A1_acp.opt <- mean(colMeans(A1_cp_total.opt))
  A1_acp.unif <- mean(colMeans(A1_cp_total.unif))
  A1_acp.uw <- mean(colMeans(A1_cp_total.uw))
  A1_acp.lev <- mean(colMeans(A1_cp_total.lev))

  #AL
  A1_AL.opt <- mean(2*1.96*colMeans(A1_sd_total.opt))
  A1_AL.unif <- mean(2*1.96*colMeans(A1_sd_total.unif))
  A1_AL.uw <- mean(2*1.96*colMeans(A1_sd_total.uw))
  A1_AL.lev <- mean(2*1.96*colMeans(A1_sd_total.lev))

  #SD 
  A1_emp.sd.opt <- mean(apply(A1_beta_total.opt,2,sd))
  A1_emp.sd.uw <- mean(apply(A1_beta_total.uw,2,sd))

  #SE
  A1_est.sd.opt <- mean(colMeans(A1_sd_total.opt))
  A1_est.sd.uw <- mean(colMeans(A1_sd_total.uw))


  A1_final_result <- rbind(A1_final_result,c(r2,A1_Bias.uw,A1_acp.uw,A1_AL.uw,A1_Bias.opt,A1_acp.opt,A1_AL.opt,A1_Bias.lev,A1_acp.lev,A1_AL.lev,A1_Bias.unif,A1_acp.unif,A1_AL.unif))
  A1_another_result <- rbind(A1_another_result,c(A1_MSE.uw,A1_MSE.opt,A1_MSE.lev,A1_MSE.unif,A1_emp.sd.uw,A1_est.sd.uw,A1_emp.sd.opt,A1_est.sd.opt))
   
  A <- matrix(1,1,p)/sqrt(p)
  A2_Bias.opt <- mean(colMeans(A2_beta_total.opt)-as.vector(A %*% bt))
  A2_Bias.unif <- mean(colMeans(A2_beta_total.unif)-as.vector(A %*% bt))
  A2_Bias.uw <- mean(colMeans(A2_beta_total.uw)-as.vector(A %*% bt))
  A2_Bias.lev <- mean(colMeans(A2_beta_total.lev)-as.vector(A %*% bt))
  
  A2_MSE.opt <- sum((colMeans(A2_beta_total.opt)-as.vector(A %*% bt))^2+apply(A2_beta_total.opt,2,sd)^2)
  A2_MSE.unif <- sum((colMeans(A2_beta_total.unif)-as.vector(A %*% bt))^2+apply(A2_beta_total.unif,2,sd)^2)
  A2_MSE.uw <- sum((colMeans(A2_beta_total.uw)-as.vector(A %*% bt))^2+apply(A2_beta_total.uw,2,sd)^2)
  A2_MSE.lev <- sum((colMeans(A2_beta_total.lev)-as.vector(A %*% bt))^2+apply(A2_beta_total.lev,2,sd)^2)
  
  
  A2_acp.opt <- mean(colMeans(A2_cp_total.opt))
  A2_acp.unif <- mean(colMeans(A2_cp_total.unif))
  A2_acp.uw <- mean(colMeans(A2_cp_total.uw))
  A2_acp.lev <- mean(colMeans(A2_cp_total.lev))
  
  
  A2_AL.opt <- mean(2*1.96*colMeans(A2_sd_total.opt))
  A2_AL.unif <- mean(2*1.96*colMeans(A2_sd_total.unif))
  A2_AL.uw <- mean(2*1.96*colMeans(A2_sd_total.uw))
  A2_AL.lev <- mean(2*1.96*colMeans(A2_sd_total.lev))
  
  A2_emp.sd.opt <- mean(apply(A2_beta_total.opt,2,sd))
  A2_emp.sd.uw <- mean(apply(A2_beta_total.uw,2,sd))
  
  
  A2_est.sd.opt <- mean(colMeans(A2_sd_total.opt))
  A2_est.sd.uw <- mean(colMeans(A2_sd_total.uw))
  
  
  A2_final_result <- rbind(A2_final_result,c(r2,A2_Bias.uw,A2_acp.uw,A2_AL.uw,A2_Bias.opt,A2_acp.opt,A2_AL.opt,A2_Bias.lev,A2_acp.lev,A2_AL.lev,A2_Bias.unif,A2_acp.unif,A2_AL.unif))
  A2_another_result <- rbind(A2_another_result,c(A2_MSE.uw,A2_MSE.opt,A2_MSE.lev,A2_MSE.unif,A2_emp.sd.uw,A2_est.sd.uw,A2_emp.sd.opt,A2_est.sd.opt))
  
  A <- matrix(0,1,p)
  A[1,1] <- 1
  A3_Bias.opt <- mean(colMeans(A3_beta_total.opt)-as.vector(A %*% bt))
  A3_Bias.unif <- mean(colMeans(A3_beta_total.unif)-as.vector(A %*% bt))
  A3_Bias.uw <- mean(colMeans(A3_beta_total.uw)-as.vector(A %*% bt))
  A3_Bias.lev <- mean(colMeans(A3_beta_total.lev)-as.vector(A %*% bt))

  A3_MSE.opt <- sum((colMeans(A3_beta_total.opt)-as.vector(A %*% bt))^2+apply(A3_beta_total.opt,2,sd)^2)
  A3_MSE.unif <- sum((colMeans(A3_beta_total.unif)-as.vector(A %*% bt))^2+apply(A3_beta_total.unif,2,sd)^2)
  A3_MSE.uw <- sum((colMeans(A3_beta_total.uw)-as.vector(A %*% bt))^2+apply(A3_beta_total.uw,2,sd)^2)
  A3_MSE.lev <- sum((colMeans(A3_beta_total.lev)-as.vector(A %*% bt))^2+apply(A3_beta_total.lev,2,sd)^2)

  A3_acp.opt <- mean(colMeans(A3_cp_total.opt))
  A3_acp.unif <- mean(colMeans(A3_cp_total.unif))
  A3_acp.uw <- mean(colMeans(A3_cp_total.uw))
  A3_acp.lev <- mean(colMeans(A3_cp_total.lev))

  A3_AL.opt <- mean(2*1.96*colMeans(A3_sd_total.opt))
  A3_AL.unif <- mean(2*1.96*colMeans(A3_sd_total.unif))
  A3_AL.uw <- mean(2*1.96*colMeans(A3_sd_total.uw))
  A3_AL.lev <- mean(2*1.96*colMeans(A3_sd_total.lev))

  A3_emp.sd.opt <- mean(apply(A3_beta_total.opt,2,sd))
  A3_emp.sd.uw <- mean(apply(A3_beta_total.uw,2,sd))


  A3_est.sd.opt <- mean(colMeans(A3_sd_total.opt))
  A3_est.sd.uw <- mean(colMeans(A3_sd_total.uw))


  A3_final_result <- rbind(A3_final_result,c(r2,A3_Bias.uw,A3_acp.uw,A3_AL.uw,A3_Bias.opt,A3_acp.opt,A3_AL.opt,A3_Bias.lev,A3_acp.lev,A3_AL.lev,A3_Bias.unif,A3_acp.unif,A3_AL.unif))
  A3_another_result <- rbind(A3_another_result,c(A3_MSE.uw,A3_MSE.opt,A3_MSE.lev,A3_MSE.unif,A3_emp.sd.uw,A3_est.sd.uw,A3_emp.sd.opt,A3_est.sd.opt))
}

colnames(A1_another_result) <- c("MSE_uw","MSE_opt","MSE_lev","MSE_unif","EMP_uw","EST_uw","EMP_opt","EST_opt")
colnames(A2_another_result) <- c("MSE_uw","MSE_opt","MSE_lev","MSE_unif","EMP_uw","EST_uw","EMP_opt","EST_opt")
colnames(A3_another_result) <- c("MSE_uw","MSE_opt","MSE_lev","MSE_unif","EMP_uw","EST_uw","EMP_opt","EST_opt")

j.tmp <- vector()
i.tmp <- matrix(0,3,4)
bias_index <- c(1,4,7,10)
for (i in 1:4)
{
  for (j in 1:4){
    j.tmp <- cbind(j.tmp,A1_final_result[i,(j*3-1):(j*3+1)])
  }
  i.tmp <- rbind(i.tmp, j.tmp)
  j.tmp <- vector()
}
f1 <- i.tmp[-(1:3),]
f1[bias_index,] <- f1[bias_index,]*100

j.tmp <- vector()
i.tmp <- matrix(0,3,4)
for (i in 1:4)
{
  for (j in 1:4){
    j.tmp <- cbind(j.tmp,A2_final_result[i,(j*3-1):(j*3+1)])
  }
  i.tmp <- rbind(i.tmp, j.tmp)
  j.tmp <- vector()
}
f2 <- i.tmp[-(1:3),]
f2[bias_index,] <- f2[bias_index,]*100

j.tmp <- vector()
i.tmp <- matrix(0,3,4)
for (i in 1:4)
{
  for (j in 1:4){
    j.tmp <- cbind(j.tmp,A3_final_result[i,(j*3-1):(j*3+1)])
  }
  i.tmp <- rbind(i.tmp, j.tmp)
  j.tmp <- vector()
}
f3 <- i.tmp[-(1:3),]
f3[bias_index,] <- f3[bias_index,]*100

gap <- matrix(9,1,4) #a seperated line
MSE <- rbind(A1_another_result,A2_another_result,A3_another_result)[,1:4]
sdse <- rbind(A1_another_result,A2_another_result,A3_another_result)[,5:8]
print(round(rbind(f1,gap,f2,gap,f3),3)) #Bias, ACP and AL for three examples
print(log(MSE)) #MSE for three examples
print(sdse) #SD and SE for three examples