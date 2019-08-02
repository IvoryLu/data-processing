bleed = read.csv("H:/Juan Lu/AF/Outcome/bleed_new.csv")
ftime <- bleed$day_new
fstatus <- factor(bleed$status,levels = c(0,1,2),labels = c("Censored","Bleed","Mortality"))
cov <- factor(bleed$AnticoagcomboA, levels = c(0,1),labels = c("No Anticoag","Anticoag") )


print(z <- cuminc(ftime,fstatus,cov,cencode="Censored"))
summary(z)
plot(z,col = 1:8, lwd = 3, lty = 1, xlab = "time")
# legend("topleft",c("No anticoag Bleed","anticoag Bleed","No anticoag Death","anticoag Death"),fill=c("black","red","green","blue"))

stroke = read.csv("H:/Juan Lu/AF/Outcome/af_stroke.csv")
ftime <- stroke$day_new
fstatus <- factor(stroke$status,levels = c(0,1,2),labels = c("Censored","Stroke","Mortality"))
cov <- factor(stroke$AnticoagcomboA, levels = c(0,1),labels = c("No Anticoag","Anticoag") )
failcodes<-stroke$censoring

print(z <- cuminc(ftime,fstatus,cov,cencode = "Censored"))

# print(z <- crr(ftime,fstatus,cov,cencode = "Censored",failcode = "Stroke"))

summary(z)
# plot(z)
plot(z,col = 1:4, lwd = 3, lty = 1, xlab = "time",ylim=c(0.0,0.1))
# legend("topleft",c("No anticoag Stroke","anticoag Stroke","No anticoag Death","anticoag Death"),fill=c("black","red","green","blue"))
