library(stringr)
library(zoo)
files <- list.files(path="game_logs", pattern="*.csv", full.names=F, recursive=FALSE)
episode.scores = data.frame(episode = numeric(), scenario = character(), score = numeric(), seed.biomass=numeric())

for(f in files) {
  ep.name = str_remove(f, ".csv")
  ep.number = as.numeric(str_remove(ep.name, ".*_run"))
  ep.scenario = paste(str_split(f, "_")[[1]][3:5], collapse="_")
  l = read.csv(paste0("game_logs/", f))
  ep.score = sum(l$reward)
  episode.scores = rbind(episode.scores,
                         list(episode = ep.number, scenario = ep.scenario, score = ep.score,
                              seed.biomass = max(l$seed_biomass)))
  print(paste0(f, " score ", ep.score))
}
# Order the scores by episode
episode.scores <- episode.scores[order(episode.scores$episode), ]
episode.scores <- episode.scores[episode.scores$episode != 1, ]
write.csv(episode.scores, "episode_scores.csv", row.names=F)

#episode.scores = read.csv("episode_scores.csv")

# Set up the plot with minimal ink
plot(episode.scores$score ~ episode.scores$episode, 
    xlab="Playthrough", 
    ylab="Summed Rewards", 
    type="n", 
    main="",
    xaxt='n', 
    yaxt='n', 
    bty='n')

title("AI Learning Curve", line=0.5)

# Add custom axis
axis(1, at=pretty(episode.scores$episode), cex.axis=0.8, lwd=0)
axis(2, at=pretty(episode.scores$score), cex.axis=0.8, lwd=0, las=2)

# Add grid lines
abline(h=pretty(episode.scores$score), col="lightgray", lty="dotted")
abline(v=pretty(episode.scores$episode), col="lightgray", lty="dotted")

# Plot the scores
points(episode.scores$episode, episode.scores$score, pch=19, col=rgb(0.2, 0.4, 0.6, 0.7))

# Add the moving average line
lines(rollmean(episode.scores$score, k=25, fill=NA), col="red", lwd=2)