source("plot_exemplary_playthroughs.R")
# Set up the plotting area
pdf("plots/first_figure.pdf", width = 6.5, height = 3.5)
par(cex=0.7)
par(fig=c(0, 0.5, 0.5, 1), mgp=c(2, 0.5, 0), mar=c(3,3,2,1))
source("plot_learning_curve.R")

par(fig=c(0.5, 1, 0, 0.4), new=T)
par(mar=c(2,2,1,0), cex=0.5, mgp=c(2, 0.5, 0))
plot_exemplary_playthrough(528, T)

par(mar=c(0,2.8,0.5,0))
par(fig=c(0.5, 1, 0.4, 0.7), new=T)
plot_exemplary_playthrough(301)
par(fig=c(0.5, 1, 0.7, 1), new=T)
par(mar=c(0.5,2.8,0,0))
plot_exemplary_playthrough(2, F, T)
dev.off()
