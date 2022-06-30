# Title: Introducing MAR processes in simulated datasets

# Notes:
    #* Description: R script that generates missingness in simulated data
    #* Updated: 2022-05-23
    #* Updated by: dcr

# Setup 
    #* Set seed
set.seed(717)
    #* Modularly load functions
box::use(
    mice = mice[ampute],
    dplyr = dplyr[select, filter]
)


# Introducing MAR
    #* MAR in simple_dgp simulated dataset
        #** MVN
for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
    x = ampute(simple_dgp_mvn, mech = 'MAR', prop = i,freq = c(0, 0, 0.045, 0, 0, 0.045, 0, 0, 0, 0, 0.3, 0.3, 0.3))
    assign(paste0('simple_mvn_mar_', i), x)
}
        #** NMVN
for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
    x = ampute(simple_dgp_nmvn, mech = 'MAR', prop = i,freq = c(0, 0, 0.045, 0, 0, 0.045, 0, 0, 0, 0, 0.3, 0.3, 0.3))
    assign(paste0('simple_nmvn_mar_', i), x)
}
    #* MAR in moderated_dgp simulated dataset
        #** MVN
for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
    x = ampute(moderated_dgp_mvn, mech = 'MAR', prop = i, freq = c(0, 0, 0.045, 0, 0, 0.045, 0, 0, 0, 0, 0.3, 0.3, 0.3))
    assign(paste0('moderated_mvn_mar_', i), x) 
}
        #** NMVN
for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
    x = ampute(moderated_dgp_nmvn, mech = 'MAR', prop = i, freq = c(0, 0, 0.045, 0, 0, 0.045, 0, 0, 0, 0, 0.3, 0.3, 0.3))
    assign(paste0('moderated_nmvn_mar_', i), x) 
}
    #* MAR in hierarchical_dgp simulated dataset
#for(i in seq(from = 0.1, to = 0.9, by = 0.1)){
#    x = ampute(hierarchical_dgp, mech = 'MAR', prop = i)
#    assign(paste0('hierarchical_mar_', i), x)
#}