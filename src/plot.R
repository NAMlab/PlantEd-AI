library(stringr)
library(zoo)
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
  l[8:11] = log(l[8:11])
  l[l == -Inf] <- NA
  y.max = max(l[8:11], na.rm = T)
  y.min = min(l[8:11], na.rm = T)
  
  pdf(paste0("pdfs/", ep.name, ".pdf"), 7, 14)
  par(mfrow=c(10,1), mar=c(2, 4.5, 0.2, 0))
  plot(l$temperature ~ l$time, type="l", ylab="Temperature")
  plot(l$humidity ~ l$time, type="l", ylab="Humidity")
  plot(l$sun_intensity ~ l$time, type="l", ylab="Sun Intensity")
  legend("topleft", pch=c(1,4),
         legend = c("Open Stomata", "Close Stomata"))
  # Mark stomata openings and closings
  actions = l[l$action == "OPEN_STOMATA",]
  points(actions$time, actions$sun_intensity, pch=1)
  actions = l[l$action == "CLOSE_STOMATA",]
  points(actions$time, actions$sun_intensity, pch=4)
  plot(l$precipitation ~ l$time, type="l", ylab="Precipitation")
  plot(l$water_pool ~ l$time, type="l", ylab="Internal Water Pool")
  lines(l$max_water_pool ~ l$time, col="grey", lty=2)
  plot(l$accessible_water ~ l$time, type="l", ylab="Accessible Water")
  plot(l$accessible_nitrate ~ l$time, type="l", ylab="Accessible Nitrate")
  
  plot(l$leaf_percent ~ l$time, type="n", ylim=c(0, 100), ylab="Growth Percentages")
  lines(rollmean(l$leaf_percent, k = 10, fill=0) ~ l$time, col="darkgreen")
  lines(rollmean(l$stem_percent, k = 10, fill=0) ~ l$time, col="green")
  lines(rollmean(l$root_percent, k = 10, fill=0) ~ l$time, col="brown")
  lines(rollmean(l$starch_percent, k = 10, fill=0) ~ l$time, col="black", lty=5)
  
  plot(l$leaf_biomass ~ l$time, type="n", ylim=c(y.min, y.max), ylab="log(Biomass)")
  lines(l$leaf_biomass ~ l$time, col="darkgreen")
  actions = l[l$action == "BUY_LEAF",]
  points(actions$time, actions$leaf_biomass, pch="*", col="darkgreen")
  lines(l$stem_biomass ~ l$time, col="green")
  actions = l[l$action == "BUY_STEM",]
  points(actions$time, actions$stem_biomass, pch="*", col="green")
  lines(l$root_biomass ~ l$time, col="brown")
  actions = l[l$action == "BUY_ROOT",]
  points(actions$time, actions$root_biomass, pch="*", col="brown")
  lines(l$seed_biomass ~ l$time, col="red")
  actions = l[l$action == "BUY_SEED",]
  points(actions$time, actions$seed_biomass, pch="*", col="red")
  legend("topleft", col=c("darkgreen", "green", "brown", "red", "black", "grey"),
         legend = c("Leaf", "Stem", "Root", "Seed", "Starch Pool", "Max Starch Pool"), lty=c(rep(1, 5), 2), lwd=2)
  plot(l$starch_pool ~ l$time, col="black", type="l", ylab="Starch Pool")
  lines(l$max_starch_pool ~ l$time, col="grey", lty=2)
  
  dev.off()
}
print(paste0("Best episode: ", best.episode, " with score ", best.score))
