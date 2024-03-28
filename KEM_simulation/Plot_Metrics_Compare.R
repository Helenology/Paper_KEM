### read in the data
Ch_df = read.csv("/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/Ch-2023-10-31.csv")
random_info = read.csv("/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/random100_info.csv")
data = merge(random_info, Ch_df, by = "patient", all.x = T)
rownames(data) = data$patient
data = data[random_info$patient, ]
rownames(data) = 1:100


### Ch (Bandwidth Constant Raw Compare)
boxplot(data$Ch_CV, data$Ch_REG, 
        names = c("CV", "REG"),
        ylab = "The Optimal Bandwidth Constant",
        cex.axis = 1.5, cex.lab = 1.5)

median(data$Ch_CV, na.rm = T)
median(data$Ch_REG, na.rm = T)
a = quantile(data$Ch_CV, na.rm = T)
b = quantile(data$Ch_REG, na.rm = T)
a[4] - a[2]
b[4] - b[2]


### Ch (Bandwidth Constant Time Compare)
boxplot(log(data$time_CV), log(data$time_REG),
        names = c("CV", "REG"),
        ylab = "Logarithm of Time Duration in Seconds",
        cex.axis = 1.5, cex.lab = 1.5)


### Metrics Compare
metrics_KEM = read.csv("/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/compare_metrics_KEM-2023-11-01.csv")
metrics_others = read.csv("/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/compare_metrics_Kmeans_GMM-2023-10-31.csv")
metrics = merge(metrics_KEM, metrics_others, by = "patient")


### ACC
boxplot(metrics$CV_ACC, metrics$REG_ACC, metrics$Kmeans_acc, metrics$GMM_acc,
        names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
        ylab = "ACC",
        cex.axis = 1.5, cex.lab = 1.5)
median(metrics$CV_ACC)
median(metrics$REG_ACC)
median(metrics$Kmeans_acc)
median(metrics$GMM_acc)
median(metrics$CV_ACC) - max(median(metrics$Kmeans_acc), median(metrics$GMM_acc))
median(metrics$REG_ACC) - max(median(metrics$Kmeans_acc), median(metrics$GMM_acc))


mean(metrics$CV_ACC)
mean(metrics$REG_ACC)
mean(metrics$Kmeans_acc)
mean(metrics$GMM_acc)
mean(metrics$CV_ACC) - max(mean(metrics$Kmeans_acc), mean(metrics$GMM_acc))
mean(metrics$REG_ACC) - max(mean(metrics$Kmeans_acc), mean(metrics$GMM_acc))


# ### RMSE
# boxplot(metrics$CV_pi_rmse, metrics$REG_pi_rmse, metrics$Kmeans_pi_rmse, metrics$GMM_pi_rmse,
#         names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
#         ylab = "Pi RMSE")
# boxplot(metrics$CV_mu_rmse, metrics$REG_mu_rmse, metrics$Kmeans_mu_rmse, metrics$GMM_mu_rmse,
#         names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
#         ylab = "mu RMSE")
# boxplot(metrics$CV_sigma_rmse, metrics$REG_sigma_rmse, metrics$Kmeans_sigma_rmse, metrics$GMM_sigma_rmse,
#         names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
#         ylab = "sigma RMSE")

### log RMSE
## pi
boxplot(log(metrics$CV_pi_rmse),
        log(metrics$REG_pi_rmse),
        log(metrics$Kmeans_pi_rmse),
        log(metrics$GMM_pi_rmse),
        names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
        ylab = "log(RMSE)",
        cex.axis = 1.5, cex.lab = 1.5)
## mu
boxplot(log(metrics$CV_mu_rmse),
        log(metrics$REG_mu_rmse),
        log(metrics$Kmeans_mu_rmse),
        log(metrics$GMM_mu_rmse),
        names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
        ylab = "log(RMSE)",
        cex.axis = 1.5, cex.lab = 1.5)
## sigma
boxplot(log(metrics$CV_sigma_rmse),
        log(metrics$REG_sigma_rmse),
        log(metrics$Kmeans_sigma_rmse),
        log(metrics$GMM_sigma_rmse),
        names = c("KEM(CV)", "KEM(REG)", "Kmeans", "GMM"),
        ylab = "log(RMSE)",
        cex.axis = 1.5, cex.lab = 1.5)

### consistency table
consistency = read.csv("/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/consistency_KEM-2023-11-02.csv")
library(plyr)
table1 = ddply(consistency, .(seed), nrow)
table2 = ddply(consistency, .(training_ratio), function(x){
        c(mean(x$pi_rmse), mean(x$mu_rmse), mean(x$sigma_rmse))
})
names(table2)[2:4] = c("pi_rmse", "mu_rmse", "sigma_rmse")
table2
write.csv(table2, "/Users/helenology/Documents/GitHub/Paper_KEM/KEM_simulation/【results】/consistency_table.csv")
