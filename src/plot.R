library(stringr)
files <- list.files(path=".", pattern="*.csv", full.names=F, recursive=FALSE)
for(f in files) {
  print(f)
  ep.name = str_remove(f, ".csv")
  l = read.csv(f)
  l[6:11] = log(l[6:11])
  y.max = max(l[6:11])
  y.min = min(l[6:11])
  
  pdf(paste0(ep.name, ".pdf"), 7, 14)
  par(mfrow=c(6,1), mar=c(2, 4.5, 0.2, 0))
  plot(l$temperature ~ l$time, type="l", ylab="Temperature")
  plot(l$humidity ~ l$time, type="l", ylab="Humidity")
  plot(l$sun_intensity ~ l$time, type="l", ylab="Sun Intensity")
  plot(l$precipitation ~ l$time, type="l", ylab="Precipitation")
  
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
  dev.off()
  
}
