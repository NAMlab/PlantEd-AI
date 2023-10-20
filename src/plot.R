library(stringr)
files <- list.files(path=".", pattern="*.csv", full.names=F, recursive=FALSE)
best.score = 0
best.episode = ""

for(f in files) {
  ep.name = str_remove(f, ".csv")
  l = read.csv(f)
  ep.score = sum(l[nrow(l),c("leaf_biomass", "stem_biomass", "root_biomass", "seed_biomass")])
  print(paste0(f, " score ", ep.score))
  if(ep.score > best.score) {
    best.score = ep.score
    best.episode = ep.name
  }
  l[8:13] = log(l[8:13])
  y.max = max(l[8:13])
  y.min = min(l[8:13])
  
  pdf(paste0(ep.name, ".pdf"), 7, 14)
  par(mfrow=c(9,1), mar=c(2, 4.5, 0.2, 0))
  plot(l$temperature ~ l$time, type="l", ylab="Temperature")
  plot(l$humidity ~ l$time, type="l", ylab="Humidity")
  plot(l$sun_intensity ~ l$time, type="l", ylab="Sun Intensity")
  plot(l$precipitation ~ l$time, type="l", ylab="Precipitation")
  plot(l$water_pool ~ l$time, type="l", ylab="Internal Water Pool")
  lines(l$max_water_pool ~ l$time, col="grey", lty=2)
  plot(l$accessible_water ~ l$time, type="l", ylab="Accessible Water")
  plot(l$accessible_nitrate ~ l$time, type="l", ylab="Accessible Nitrate")
  
  plot(l$leaf_percent ~ l$time, type="n", ylim=c(0, 100), ylab="Growth Percentages")
  lines(l$leaf_percent ~ l$time, col="darkgreen")
  lines(l$stem_percent ~ l$time, col="green")
  lines(l$root_percent ~ l$time, col="brown")
  lines(l$seed_percent ~ l$time, col="red")
  lines(l$starch_percent ~ l$time, col="black")
  
  plot(l$leaf_biomass ~ l$time, type="n", ylim=c(y.min, y.max), ylab="log(Biomass)")
  lines(l$leaf_biomass ~ l$time, col="darkgreen")
  lines(l$stem_biomass ~ l$time, col="green")
  lines(l$root_biomass ~ l$time, col="brown")
  lines(l$seed_biomass ~ l$time, col="red")
  lines(l$starch_pool ~ l$time, col="black")
  lines(l$max_starch_pool ~ l$time, col="grey", lty=2)
  legend("topleft", col=c("darkgreen", "green", "brown", "red", "black", "grey"),
         legend = c("Leaf", "Stem", "Root", "Seed", "Starch Pool", "Max Starch Pool"), lty=c(rep(1, 5), 2), lwd=2)
  dev.off()
}
print(paste0("Best episode: ", best.episode, " with score ", best.score))