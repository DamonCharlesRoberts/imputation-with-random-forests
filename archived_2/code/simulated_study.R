# Title: Comparison of RF-MICE with other procedures w/ simulated data

# Notes:
    #* Description: RScript producing simulated data, introducing MAR via amputation, imputing the data with different procedures, and running models to generate estimates
    #* Updated: 2022-07-28
    #* Updated by: dcr

# Setup
    #* Set seed
set.seed(90210)
    #* Modularly load required functions
box::use(
    reticulate = reticulate[use_condaenv, source_python],
    fabricatr = fabricatr[...],
    DeclareDesign = DeclareDesign[...],
    #feather = feather[write_feather],
    arrow = arrow[read_feather],
    mice = mice[ampute, pool, complete, mice],
    Amelia = Amelia[amelia],
    dplyr = dplyr[select, filter, bind_rows],
    tibble = tibble[tibble],
    ggplot2 = ggplot2[ggplot, geom_density, aes, theme_minimal, labs, geom_boxplot, ggsave],
    patchwork = patchwork[...]
)
    #* Source python scripts for amputation
use_condaenv('snek')

# Establish loop parameters
data_sets = 1000

# Construct loops
    #* Simulate the data
        #** Simple DGP dataset
#simple_dgp_list = list()
#for(i in 1:data_sets){
#    simple_dgp = fabricate(N = 500,
#        Var1 = rnorm(N, mean = 10),
#        Var2 = draw_binary(N = N, prob = 0.5),
#        Var3 = draw_ordered(x = rnorm(N, mean = 3.5), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        Var4 = rnorm(N, mean = 0),
#        Var5 = draw_binary(N = N, prob = 0.25),
#        Var6 = draw_ordered(x = rnorm(N, mean = 1.75), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        Var7 = rnorm(N, mean = -10),
#        Var8 = draw_binary(N = N, prob = 0.75),
#        Var9 = draw_ordered(x = rnorm(N, mean = 5.25), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        X = draw_binary(N = N, prob = ifelse(Var2 == 1, 0.4, 0.6)),
#        Z = draw_binary(N = N, prob = ifelse(Var5 == 1, 0.4, 0.6)),
#        Y = draw_ordered(x = (mean = 0.9 * X + 0.3 * Z + rnorm(N, 0)), breaks = c(1, 2, 3, 4, 5, 6, 7))
#    )
#    name = paste('dataset', i, sep = '_')
#    simple_dgp_list[[name]] = simple_dgp
#}
        #** Moderated dgp dataset
#moderated_dgp_list = list()
#for(i in 1:data_sets){
#    moderated_dgp = fabricate(N = 500,
#        Var1 = rnorm(N, mean = 10),
#        Var2 = draw_binary(N = N, prob = 0.5),
#        Var3 = draw_ordered(x = rnorm(N, mean = 3.5), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        Var4 = rnorm(N, mean = 0),
#        Var5 = draw_binary(N = N, prob = 0.25),
#        Var6 = draw_ordered(x = rnorm(N, mean = 1.75), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        Var7 = rnorm(N, mean = -10),
#        Var8 = draw_binary(N = N, prob = 0.75),
#        Var9 = draw_ordered(x = rnorm(N, mean = 5.25), breaks = c(1, 2, 3, 4, 5, 6, 7)),
#        X = draw_binary(N = N, prob = (0.5 * Var2)),
#        Z = draw_binary(N = N, prob = (0.5 * Var5)),
#        Y = draw_ordered(x = (mean = 0.3*X + 0.2*Z + 0.8*X*Z + rnorm(N, 0)), breaks = c(1,2,3,4,5,6,7))
#    )
#    name = paste('dataset', i, sep = '')
#    moderated_dgp_list[[name]] = moderated_dgp
#}
    #* Ampute the data
#ID = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var1 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var2 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var3 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var4 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var5 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var6 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var7 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var8 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#Var9 = c(0, 0, 0.072, 0, 0.072, 0, 0, 0, 0, 0, 0.072, 0.072, 0.072)
#X = c(0, 0, 0.076, 0, 0.076, 0, 0, 0, 0, 0, 0.076, 0.076, 0.076)
#Y = c(0, 0, 0.076, 0, 0.076, 0, 0, 0, 0, 0, 0.076, 0.076, 0.076)
#Z = c(0, 0, 0.076, 0, 0.076, 0, 0, 0, 0, 0, 0.076, 0.076, 0.076)
#weights = cbind(ID, Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, X, Y, Z)
        #** Simple dgp
simple_amputed_list = list()
simple_dgp_list = list()
for(d in 1:data_sets){
    for(p in c(0.1, 0.4, 0.9)){
        a= rbinom(n = 500, prob = p, size = 1)
        b = runif(n = 500)
        c = rbinom(n = 500, prob = p, size = 1)
        x = rbinom(n = 500, prob = 0.6, size = 1)
        z = rnorm(n = 500)
        y = 0.3 * x + 0.4 * z + rnorm(mean = 0, sd = 0.1, n = 500)
        simple_dgp = data.frame(cbind(x, z, y))
        name = paste('complete', d, sep = '_')
        simple_dgp_list[[name]] = list(simple_dgp)

        y_miss = a * b
        y[y_miss == 1] = NA
        #bmax = max(b)
        #bmin = min(b)
        #miss = ifelse(a == 1, bmax, bmin)
        #zmiss = ifelse(c == 1, bmax, bmin)
        #y_miss = rbinom(500, 1, miss)
        #y[y_miss == 1] = NA
        #z_miss = rbinom(500, 1, zmiss)
        #z[z_miss == 1] = NA
        simple_amputed = data.frame(cbind(x, z, y, a, b))
        assign(paste0('simple_mar_', p), simple_amputed)
    }
    name = paste('amputed', d, sep = '_')
    simple_amputed_list[[name]] = list('simple_mar_0.1' = simple_mar_0.1, 'simple_mar_0.4' = simple_mar_0.4, 'simple_mar_0.9' = simple_mar_0.9)
}


#for(d in 1:data_sets){
#    source_python('code/simulated_study_amputation.py')
#    print('successfully sourced it')
#    a = read.csv('data/temp_amputed.csv')
#    b = read.csv('data/temp_amputed_0_1.csv')
#    c = read.csv('data/temp_amputed_0_9.csv')
#    print('successfully read it again')
#    #for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
#        #x = ampute(simple_dgp_list[[d]], mech = 'MAR', prop = i,freq = c(0, 0, 0.275, 0, 0, 0.275, 0, 0, 0, 0, 0.15, 0.15, 0.15), weights = weights)
#        #x_1 = simple_dgp_list[[d]] |>
#        #    select(X, Y, Z)
#        #x_2 = simple_dgp_list[[d]] |>
#        #    select(-c(X,Y,Z))
#        #x_1 = ampute(x_1, mech = 'MAR', prop = i, freq = c(0.33, 0.33, 0.33))
#        #x_2 = ampute(x_2, mech = 'MAR', prop = i, freq =  c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
#        #x = data.frame(cbind(x_2$amp, x_1$amp))
#    #    assign(paste0('simple_mar_', i), x)
#    #}
#    name = paste('amputed', d, sep = '_')
#    simple_amputed_list[[name]] = list('simple_mar_0.1' = b, 'simple_mar_0.4' = a, 'simple_mar_0.9' = c)
#
#    e = read.csv('data/temp_complete.csv')
#    name = paste('complete', d, sep = '_')
#    simple_dgp_list[[name]] = list(e)
#}

        #** Moderated dgp
#moderated_amputed_list = list()
#for(d in 1:data_sets){
#    write.csv(moderated_dgp_list[[d]], 'data/temp_simulated.csv')
#    print('successfully wrote it')
#    source_python('code/simulated_study_amputation.py')
#    print('successfully sourced it')
#    a = read.csv('data/temp_amputed.csv')
#    b = read.csv('data/temp_amputed_0_1.csv') |>
#        select(-X.1)
#    c = read.csv('data/temp_amputed_0_9.csv') |>
#        select(-X.1)
#    print('successfully read it again')
#    #for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
#        #x = ampute(moderated_dgp_list[[d]], mech = 'MAR', prop = i,freq = c(0, 0, 0.275, 0, 0, 0.275, 0, 0, 0, 0, 0.15, 0.15, 0.15), weights = weights) 
#        #x_1 = moderated_dgp_list[[d]] |>
#        #    select(X,Y,Z)
#        #x_2 = moderated_dgp_list[[d]] |>
#        #    select(-c(X,Y,Z))
#        #x_1 = ampute(x_1, mech = 'MAR', prop = i, freq = c(0.33, 0.33, 0.33))
#        #x_2 = ampute(x_2, mech = 'MAR', prop = i, freq = c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
#        #x = cbind(x_2$amp, x_1$amp)
#        #assign(paste0('moderated_mar_', i), x)
#    #}
#    name = paste('amputed', d, sep = '_')
#    moderated_amputed_list[[name]] = list('moderated_mar_0.1' = b, 'moderated_mar_0.4' = a, 'moderated_mar_0.9' = c)
#}

        #** Reformat lists
            #*** Simple
simple_mar_0.1_list = list()
simple_mar_0.4_list = list()
simple_mar_0.9_list = list()
for(d in 1:data_sets){
    simple_mar_0.1_list[[d]] = simple_amputed_list[[d]][['simple_mar_0.1']]
    simple_mar_0.4_list[[d]] = simple_amputed_list[[d]][['simple_mar_0.4']]
    simple_mar_0.9_list[[d]] = simple_amputed_list[[d]][['simple_mar_0.9']]
}
            #*** Moderated
#moderated_mar_0.1_list = list()
#moderated_mar_0.4_list = list()
#moderated_mar_0.9_list = list()
#for(d in 1:data_sets){
#    moderated_mar_0.1_list[[d]] = moderated_amputed_list[[d]][['moderated_mar_0.1']]
#    moderated_mar_0.4_list[[d]] = moderated_amputed_list[[d]][['moderated_mar_0.4']]
#    moderated_mar_0.9_list[[d]] = moderated_amputed_list[[d]][['moderated_mar_0.9']]
#}
    #* Imputation
        #** Simple
            #*** Mean
simple_mean_list_0.4 = list()
for(d in 1:data_sets){
    simple_mean = mice(simple_mar_0.4_list[[d]], m = 10, meth = 'mean')
    simple_mean_list_0.4[[d]] = simple_mean
}
            #*** Amelia
simple_amelia_list_0.4 = list()
for(d in 1:data_sets){
    simple_amelia = amelia(data.frame(simple_mar_0.4_list[[d]]), m = 10)
    simple_amelia_list_0.4[[d]] = simple_amelia
}
            #*** Linear bayes
simple_linear_list_0.4 = list()
for(d in 1:data_sets){
    simple_linear = mice(simple_mar_0.4_list[[d]], m = 10, meth = 'norm')
    simple_linear_list_0.4[[d]] = simple_linear
}
            #*** RF-MICE
simple_rf_list_0.4 = list()
for(d in 1:data_sets){
    simple_rf = mice(simple_mar_0.4_list[[d]], m = 10, meth = 'rf')
    simple_rf_list_0.4[[d]] = simple_rf
}
#        #** Moderated
#            #*** Mean
#moderated_mean_list_0.4 = list()
#for(d in 1:data_sets){
#    moderated_mean = mice(moderated_mar_0.4_list[[d]], m = 10, meth = 'mean')
#    moderated_mean_list_0.4[[d]] = moderated_mean
#}
#            #*** Amelia
#moderated_amelia_list_0.4 = list()
#for(d in 1:data_sets){
#    moderated_amelia = amelia(moderated_mar_0.4_list[[d]], m = 10)
#    moderated_amelia_list_0.4[[d]] = moderated_amelia
#}
#            #*** Linear bayes
#moderated_linear_list_0.4 = list()
#for(d in 1:data_sets){
#    moderated_linear = mice(moderated_mar_0.4_list[[d]], m = 10, meth = 'norm')
#    moderated_linear_list_0.4[[d]] = moderated_linear
#}
#            #*** RF
#moderated_rf_list_0.4 = list()
#for(d in 1:data_sets){
#    moderated_rf = mice(moderated_mar_0.4_list[[d]], m = 10, meth = 'rf')
#    moderated_rf_list_0.4[[d]] = moderated_rf
#}
    #* Models
        #** Simple - Complete
simple_complete_list = list()
for(d in 1:data_sets){
    simple_complete = lm(formula = y ~ x + z, data = simple_dgp_list[[d]]) |>
        tidy()
    simple_complete_list[[d]] = simple_complete
}
        #** Simple - LWD
simple_lwd_list_0.4 = list()
for(d in 1:data_sets){
    simple_lwd = lm(y ~ x + z, data = simple_mar_0.4_list[[d]]) |>
        tidy()
    simple_lwd_list_0.4[[d]] = simple_lwd
}
        #* Simple - Mean
simple_mean_0.4_model_list = list()
for(d in 1:data_sets){
    simple_mean_0.4_model = pool(with(simple_mean_list_0.4[[d]], lm(y ~ x + z))) |>
        tidy()
    simple_mean_0.4_model_list[[d]] = simple_mean_0.4_model
}
        #* Simple - AMELIA
simple_amelia_0.4_model_list = list()
for(d in 1:data_sets){
    simple_amelia_0.4_model = pool(lapply(simple_amelia_list_0.4[[d]][['imputations']], function(x) lm(y ~ x + z, data = x))) |>
        tidy()
    simple_amelia_0.4_model_list[[d]] = simple_amelia_0.4_model
}
        #* Simple - Linear Bayes
simple_linear_0.4_model_list = list()
for(d in 1:data_sets){
    simple_linear_0.4_model = pool(with(simple_linear_list_0.4[[d]], lm(y ~ x + z))) |>
        tidy()
    simple_linear_0.4_model_list[[d]] = simple_linear_0.4_model
}
        #* Simple - RF-MICE
simple_rf_0.4_model_list = list()
for(d in 1:data_sets){
    simple_rf_0.4_model = pool(with(simple_rf_list_0.4[[d]], lm(y ~ x + z))) |>
        tidy()
    simple_rf_0.4_model_list[[d]] = simple_rf_0.4_model
}
        #* Moderated - Complete
#moderated_complete_list = list()
#for(d in 1:data_sets){
#    moderated_complete = lm(Y ~ X + Z + X*Z, data = moderated_dgp_list[[d]]) |>
#        tidy()
#    moderated_complete_list[[d]] = moderated_complete
#}
#        #* Moderated - LWD
#moderated_lwd_0.4_model_list = list()
#for(d in 1:data_sets){
#    moderated_lwd_0.4 = lm(Y ~ X + Z + X*Z, data = moderated_mar_0.4_list[[d]]) |>
#        tidy()
#    moderated_lwd_0.4_model_list[[d]] = moderated_lwd_0.4
#}
#        #* Moderated - Mean
#moderated_mean_0.4_model_list = list()
#for(d in 1:data_sets){
#    moderated_mean_0.4 = pool(with(moderated_mean_list_0.4[[d]], lm(Y ~ X + Z + X*Z))) |>
#        tidy()
#    moderated_mean_0.4_model_list[[d]] = moderated_mean_0.4
#}
#        #* Moderated - AMELIA
#moderated_amelia_0.4_model_list = list()
#for(d in 1:data_sets){
#    moderated_amelia_0.4 = pool(lapply(moderated_amelia_list_0.4[[d]][['imputations']], function(x) lm(Y ~ X + Z + X*Z, data = x))) |>
#        tidy()
#    moderated_amelia_0.4_model_list[[d]] = moderated_amelia_0.4
#}
#        #* Moderated - Linear Bayes
#moderated_linear_0.4_model_list = list()
#for(d in 1:data_sets){
#    moderated_linear_0.4 = pool(with(moderated_linear_list_0.4[[d]], lm(Y ~ X + Z + X*Z))) |>
#        tidy()
#    moderated_linear_0.4_model_list[[d]] = moderated_linear_0.4
#}
#        #* Moderated - RF-MICE
#moderated_rf_0.4_model_list = list()
#for(d in 1:data_sets){
#    moderated_rf_0.4 = pool(with(moderated_rf_list_0.4[[d]], lm(Y ~ X + Z + X*Z))) |>
#        tidy()
#    moderated_rf_0.4_model_list[[d]] = moderated_rf_0.4
#}

save.image(file = 'data/simulated_study.RData')

# Reload data once the simulations have been completed
load('data/simulated_study.RData')
#rm(list=setdiff(ls(), c('data_sets','simple_mar_0.4_list', 'simple_dgp_list', 'simple_lwd_list_0.4', 'simple_mean_list_0.4', 'simple_amelia_list_0.4', 'simple_linear_list_0.4', 'simple_rf_list_0.4','simple_complete_list', 'simple_lwd_0.4_model_list', 'simple_mean_0.4_model_list', 'simple_amelia_0.4_model_list', 'simple_linear_0.4_model_list', 'simple_rf_0.4_model_list', 'moderated_complete_list', 'moderated_lwd_0.4_model_list', 'moderated_mean_0.4_model_list', 'moderated_amelia_0.4_model_list', 'moderated_linear_0.4_model_list', 'moderated_rf_0.4_model_list')))
## Graphs
#    #* Distributions - Y
#        #** Mean
#simple_mean_plot = 
#    ggplot() +
#    theme_minimal() + 
#    labs(x = 'Y Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_mean_plot = simple_mean_plot + 
#        geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Y), color = 'black', linetype = 2) +
#        geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Y), color = 'black') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]])), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 2)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 3)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 4)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 5)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 6)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 7)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 8)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 9)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 10)), aes(x = Y), color = '#404040')
#}
#ggsave(simple_mean_plot, file = 'figures/simple_mean_plot_0.4_Y')
#
#        #** Amelia
#simple_amelia_plot = 
#    ggplot() +
#    theme_minimal() + 
#    labs(x = 'Y Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_amelia_plot = simple_amelia_plot +
#        geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Y), color = 'black', linetype = 2) +
#        geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Y), color = 'black') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Y), color = '#404040')
#}
#ggsave(simple_amelia_plot, file = 'figures/simple_amelia_plot_Y')
#
#        #** Linear
#simple_linear_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Y), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Y), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Y Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_linear_plot = simple_linear_plot +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]])), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 2)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 3)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 4)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 5)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 6)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 7)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 8)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 9)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 10)), aes(x = Y), color = '#404040')
#}
#ggsave(simple_linear_plot, file = 'figures/simple_linear_plot_Y')
#        #** RF-MICE
#simple_rf_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Y), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Y), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Y Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_rf_plot = simple_rf_plot +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]])), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 2)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 3)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 4)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 5)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 6)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 7)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 8)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 9)), aes(x = Y), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 10)), aes(x = Y), color = '#404040')
#}
#ggsave(simple_rf_plot, file = 'figures/simple_rf_plot_Y')
#    #* Distributions - X
#        #** Mean
#simple_mean_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = X), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = X), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'X Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_mean_plot = simple_mean_plot + 
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]])), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 2)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 3)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 4)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 5)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 6)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 7)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 8)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 9)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 10)), aes(x = X), color = '#404040')
#}
#ggsave(simple_mean_plot, file = 'figures/simple_mean_plot_0.4_X')
#
#        #** Amelia
#simple_amelia_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = X), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = X), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'X Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_amelia_plot = simple_amelia_plot +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = X), color = '#404040')
#}
#ggsave(simple_amelia_plot, file = 'figures/simple_amelia_plot_X')
#
#        #** Linear
#simple_linear_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = X), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = X), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'X Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_linear_plot = simple_linear_plot +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]])), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 2)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 3)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 4)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 5)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 6)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 7)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 8)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 9)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 10)), aes(x = X), color = '#404040')
#}
#ggsave(simple_linear_plot, file = 'figures/simple_linear_plot_X')
#        #** RF-MICE
#simple_rf_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = X), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = X), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'X Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_rf_plot = simple_rf_plot +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]])), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 2)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 3)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 4)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 5)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 6)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 7)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 8)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 9)), aes(x = X), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 10)), aes(x = X), color = '#404040')
#}
#ggsave(simple_rf_plot, file = 'figures/simple_rf_plot_X')
#    #* Distributions - Z
#        #** Mean
#simple_mean_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Z), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Z), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Z Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_mean_plot = simple_mean_plot + 
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]])), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 2)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 3)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 4)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 5)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 6)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 7)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 8)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 9)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_mean_list_0.4[[d]], 10)), aes(x = Z), color = '#404040')
#}
#ggsave(simple_mean_plot, file = 'figures/simple_mean_plot_0.4_Z')
#
#        #** Amelia
#simple_amelia_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Z), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Z), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Z Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_amelia_plot = simple_amelia_plot +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(simple_amelia_list_0.4[[d]][['imputations']]), aes(x = Z), color = '#404040')
#}
#ggsave(simple_amelia_plot, file = 'figures/simple_amelia_plot_Z')
#
#        #** Linear
#simple_linear_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Z), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Z), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Z Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_linear_plot = simple_linear_plot +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]])), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 2)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 3)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 4)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 5)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 6)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 7)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 8)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 9)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_linear_list_0.4[[d]], 10)), aes(x = Z), color = '#404040')
#}
#ggsave(simple_linear_plot, file = 'figures/simple_linear_plot_Z')
#        #** RF-MICE
#simple_rf_plot = 
#    ggplot() +
#    geom_density(data = as.data.frame(simple_mar_0.4_list[[d]]), aes(x = Z), color = 'black', linetype = 2) +
#    geom_density(data = as.data.frame(simple_dgp_list[[d]]), aes(x = Z), color = 'black') +
#    theme_minimal() + 
#    labs(x = 'Z Variable', y = 'Density')
#for(d in 1:data_sets){
#    simple_rf_plot = simple_rf_plot +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]])), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 2)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 3)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 4)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 5)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 6)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 7)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 8)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 9)), aes(x = Z), color = '#404040') +
#        geom_density(data = as.data.frame(complete(simple_rf_list_0.4[[d]], 10)), aes(x = Z), color = '#404040')
#}
#ggsave(simple_rf_plot, file = 'figures/simple_rf_plot_Z')

    #* Box plots of coefficients
        #** Convert to dataframe to make life easier
simple_complete_box_df = data.frame()
for(d in 1:data_sets){
    simple_complete = simple_complete_list[[d]] |>
        select(term, estimate)
    simple_complete_box_df = rbind(simple_complete_box_df, simple_complete) |> 
        filter(term != '(Intercept)')
}
simple_lwd_0.4_box_df = data.frame()
for(d in 1:data_sets){
    simple_lwd= simple_lwd_list_0.4[[d]] |>
        select(term, estimate)
    simple_lwd_0.4_box_df = rbind(simple_lwd_0.4_box_df, simple_lwd) |> 
        filter(term != '(Intercept)')
}
simple_mean_0.4_box_df = data.frame()
for(d in 1:data_sets){
    simple_mean = simple_mean_0.4_model_list[[d]] |>
        select(term, estimate)
    simple_mean_0.4_box_df = rbind(simple_mean_0.4_box_df, simple_mean) |> 
        filter(term != '(Intercept)')
}
simple_amelia_0.4_box_df = data.frame()
for(d in 1:data_sets){
    simple_amelia = simple_amelia_0.4_model_list[[d]] |>
        select(term, estimate)
    simple_amelia_0.4_box_df = rbind(simple_amelia_0.4_box_df, simple_amelia) |> 
        filter(term != '(Intercept)')
}
simple_linear_0.4_box_df = data.frame()
for(d in 1:data_sets){
    simple_linear = simple_linear_0.4_model_list[[d]] |>
        select(term, estimate)
    simple_linear_0.4_box_df = rbind(simple_linear_0.4_box_df, simple_linear) |> 
        filter(term != '(Intercept)')
}
simple_rf_0.4_box_df = data.frame()
for(d in 1:data_sets){
    simple_rf = simple_rf_0.4_model_list[[d]] |>
        select(term, estimate)
    simple_rf_0.4_box_df = rbind(simple_rf_0.4_box_df, simple_rf) |> 
        filter(term != '(Intercept)')
}
        #* Simple
simple_complete_box_df_x = simple_complete_box_df |>
    filter(term == 'x')
simple_lwd_0.4_box_df_x = simple_lwd_0.4_box_df |>
    filter(term == 'x')
simple_mean_0.4_box_df_x = simple_mean_0.4_box_df |>
    filter(term == 'x')
simple_amelia_0.4_box_df_x = simple_amelia_0.4_box_df |>
    filter(term == 'x')
simple_linear_0.4_box_df_x = simple_linear_0.4_box_df |>
    filter(term == 'x')
simple_rf_0.4_box_df_x = simple_rf_0.4_box_df |>
    filter(term == 'x')
simple_box_plot_x = ggplot() +
    geom_boxplot(aes(x = 'Complete', y = estimate), data = simple_complete_box_df_x) +
    geom_boxplot(aes(x = 'LWD', y = estimate), data = simple_lwd_0.4_box_df_x) +
    geom_boxplot(aes(x = 'Mean', y = estimate), data = simple_mean_0.4_box_df_x) +
    geom_boxplot(aes(x = 'Amelia', y = estimate), data = simple_amelia_0.4_box_df_x) +
    geom_boxplot(aes(x = 'Linear-Mice', y = estimate), data = simple_linear_0.4_box_df_x) +
    geom_boxplot(aes(x = 'RF-Mice', y = estimate), data = simple_rf_0.4_box_df_x) +
    theme_minimal() +
    labs(x = 'X', y = 'Estimates')

simple_complete_box_df_z = simple_complete_box_df |>
    filter(term == 'z')
simple_lwd_0.4_box_df_z = simple_lwd_0.4_box_df |>
    filter(term == 'z')
simple_mean_0.4_box_df_z = simple_mean_0.4_box_df |>
    filter(term == 'z')
simple_amelia_0.4_box_df_z = simple_amelia_0.4_box_df |>
    filter(term == 'z')
simple_linear_0.4_box_df_z = simple_linear_0.4_box_df |>
    filter(term == 'z')
simple_rf_0.4_box_df_z = simple_rf_0.4_box_df |>
    filter(term == 'z')
simple_box_plot_z = ggplot() +
    geom_boxplot(aes(x = 'Complete', y = estimate), data = simple_complete_box_df_z) + 
    geom_boxplot(aes(x = 'LWD', y = estimate), data = simple_lwd_0.4_box_df_z) +
    geom_boxplot(aes(x = 'Mean', y = estimate), data = simple_mean_0.4_box_df_z) +
    geom_boxplot(aes(x = 'Amelia', y = estimate), data = simple_amelia_0.4_box_df_z) +
    geom_boxplot(aes(x = 'Linear-Mice', y = estimate), data = simple_linear_0.4_box_df_z) +
    geom_boxplot(aes(x = 'RF-Mice', y = estimate), data = simple_rf_0.4_box_df_z) +
    theme_minimal() +
    labs(x = 'Z', y = 'Estimates')

simple_box_plot = simple_box_plot_x / simple_box_plot_z

ggplot2::ggsave(simple_box_plot, file = 'figures/simple_model_results.png')