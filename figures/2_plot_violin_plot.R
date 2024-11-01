# Load necessary libraries
library(ggplot2)

# Load the table "human_scores"
human_scores <- read.csv("human_scores.csv")
human_scores$score <- log(human_scores$score)
best_ai_score <- log(35.60908)

# Create a violin plot of the column "score" with individual scores as points with some jitter
# Save the plot to "human_scores.pdf"
pdf("plots/human_scores.pdf", width = 6.5, height = 3.5)
print(ggplot(human_scores, aes(x = factor(1), y = score)) +
    geom_violin(color = "gray50") +
    geom_jitter(width = 0.2, height = 0, alpha = 0.5, color = rgb(0.2, 0.4, 0.6, 0.7)) +
    geom_hline(yintercept = best_ai_score, color = "red", linetype = "dashed", linewidth = 0.5) +
    annotate("text", x = 0.4, y = best_ai_score, label = "best AI playthrough", color = "red", vjust = -1, hjust = 0) +
    labs(title = "Human Player Playthroughs", x = "", y = "log(Seed Mass [g])") +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.title.y = element_text(size = 12),
        axis.text.y = element_text(size = 10),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()
    ))

# Calculate the percentile of the best AI playthrough compared to the human scores
percentile <- ecdf(human_scores$score)(best_ai_score) * 100
print(paste("The best AI playthrough is in the", round(percentile, 2), "percentile compared to the human scores."))
dev.off()