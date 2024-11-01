plot_exemplary_playthrough <- function(playthrough_id, x_axis=F, title=F) {
    l = read.csv(paste0("game_logs/human_summer_low_nitrate_run",playthrough_id,".csv"))
    l$days = l$time / (60*60*24)
    
    l[8:11] = log(l[8:11])
    l[l == -Inf] <- NA
    
    plot(l$leaf_biomass ~ l$days, type="n", ylab="", xlab="", axes=F, ylim=c(-5.2, 3.6))
    axis(side=2, las=2, lwd=0, at=seq(-5,3,2))
    abline(h=seq(-5,3,2), col="lightgray", lty="dotted")
    abline(v=seq(0, 25, 5), col="lightgray", lty="dotted")
    if(x_axis) {
      axis(side=1, lwd=0, line=-0.5)
      mtext("In-Game Days", side=1, line=1, cex=0.5)
    }
    if(title) {
      title("log(Biomass [g])", line=-1)
      legend("topright", legend=c("Leaf Tissue", "Stem Tissue", "Root Tissue", "Seed Tissue"), 
        col=c("#2E3532", "#84C0C6", "#BE7C4D", "#3F6696"), lty=1, cex=0.8, bty="n", lwd=2)
    }
    lines(l$leaf_biomass ~ l$days, lwd=1.5, col="#2E3532")
    lines(l$stem_biomass ~ l$days, lwd=1.5, col="#84C0C6")
    lines(l$root_biomass ~ l$days, lwd=1.5, col="#BE7C4D")
    lines(l$seed_biomass ~ l$days, lwd=1.5, col="#3F6696")
    text(-.8, 3.0, paste0("Playthrough ", playthrough_id), pos=4)
}

# for(f in c("game_logs/human_summer_low_nitrate_run2.csv",
#                      "game_logs/human_summer_low_nitrate_run301.csv",
#                      "game_logs/human_summer_low_nitrate_run528.csv")) {
#     plot_exemplary_playthrough(f)
# }
