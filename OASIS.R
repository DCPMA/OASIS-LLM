if (!require(nlme)) {install.packages("nlme"); require(nlme)}
if (!require(lme4)) {install.packages("lme4"); require(lme4)}
if (!require(psych)) {install.packages("psych"); require(psych)}
if (!require(lmerTest)) {install.packages("lmerTest"); require(lmerTest)}

# Descriptives ------------------------------------------------------------

data$Gender <- factor(data$Gender, levels = c(1,2), labels = c("Male", "Female"))
table(data$Gender)
barplot(table(data$Gender), col = "white", main = "Gender")

mean(data$Age, na.rm = TRUE)
sd(data$Age, na.rm = TRUE)
hist(data$Age, xlab = "Age", main = "Participant age")
range(data$Age, na.rm = TRUE)

data$Ethnicity <- factor(data$Ethnicity, levels = c(1, 2, 4),
                         labels = c("Hispanic or Latino", "Not Hispanic or Latino", "Unknown"))
table(data$Ethnicity)

data$Race <- factor(data$Race, levels = c(1:7, 9), labels = c("American Indian/Alaska Native",
          "East Asian", "South Asian", "Native Hawaiian or other Pacific Islander",
          "Black or African American", "White", "More than one race", "Other or unknown"))
table(data$Race)

data$Ideol <- factor(data$Ideol, levels = c(1:7, 9), labels = c(
     "Strongly conservative", "Moderately conservative", "Slightly conservative",
     "Neutral, middle of the road", "Slightly liberal", "Moderately liberal", "Strongly liberal",
     "Don't know/Prefer not to answer"))
table(data$Ideol)

data$Income <- factor(data$Income, levels = c(1:5, 7), labels = c("below $25,000",
     "$25,000 to $44,999", "$50,000 to $69,999", "$70,000 to $99,999", "$100,000 or above",
     "Don't know/Prefer not to answer"))
table(data$Income)

data$Education <- factor(data$Education, levels = 1:5, labels = c("Grade school/some high school",
     "High school diploma", "Some college, no degree", "College degree", "Graduate degree"))
table(data$Education)

op <- par(mfrow = c(1,3))
barplot(table(data$Gender), col = "white", main = "Gender")
hist(data$Age, xlab = "Age", main = "Age")
barplot(table(data$Race), col = "white", main = "Race", names.arg = c("Am Indian", "East Asian",
     "South Asian", "Hawaiian", "Black", "White", "Multiracial", "Other"), cex.names = 0.75)
par(op)

op <- par(mfrow = c(1, 3))
barplot(table(data$Ideol), col = "white", main = "Ideology",
        names.arg = c("Strong C", "Mod C", "Slight C", "Neutral", "Slight L",
                      "Mod L", "Strong L", "NA"), cex.names = 0.75)
barplot(table(data$Education), col = "white", main = "Education",
        names.arg = c("Grade sch", "High sch", "Some college", "College degree",
                      "Grad degree"), cex.names = 0.75)
barplot(table(data$Income), col = "white", main = "Household income",
        names.arg = c("< $25k", "< $50k", "< $70k",
                      "< $100k", "$100k+", "NA"), cex.names = 0.75)
par(op)


# Figure 1 ----------------------------------------------------------------
op <- par(mfrow = c(2, 2))
par(mar=c(3.5, 10, 3.5, 3.5) + .1)
barplot(table(data$Race), col = "white", main = "Race", horiz = TRUE, las = 1,
        names.arg = c("American Indian", "East Asian", "Sout Asian", "Hawaiian", "Black", "White",
                      "Multiracial", "Other"))
barplot(table(data$Ideol), col = "white", main = "Ideology", horiz = TRUE, las = 1,
        names.arg = c("Strong conservative", "Moderate conservative", "Slight conservative", "Neutral",
                      "Slight liberal", "Moderate liberal", "Strong liberal", "NA"))
barplot(table(data$Education), col = "white", main = "Education", horiz = TRUE, las = 1,
        names.arg = c("Grade school", "High school", "Some college", "College degree",
                      "Graduate degree"))
barplot(table(data$Income), col = "white", main = "Household income", horiz = TRUE, las = 1,
        names.arg = c("< $25k", "< $50k", "< $70k",
                      "< $100k", "$100k+", "NA"))
par(op)


# Calculating mean valence and arousal values -----------------------------

valence_means <- apply(data[data$valar == "Valence",
                    which(colnames(data) == "I1"):which(colnames(data) == "I900")], 2,
                    mean, na.rm = TRUE)

valence_N <- vector()
i <- NULL
for (i in which(colnames(data) == "I1"):which(colnames(data) == "I900")) {
     valence_N[i] <- length(data[data$valar == "Valence", i][is.na(data[data$valar == "Valence", i]) == FALSE])
}
valence_N <- valence_N[is.na(valence_N) == FALSE]

table(valence_N)

arousal_means <- apply(data[data$valar == "Arousal",
                    which(colnames(data) == "I1"):which(colnames(data) == "I900")], 2,
                    mean, na.rm = TRUE)

arousal_N <- vector()
i <- NULL
for (i in which(colnames(data) == "I1"):which(colnames(data) == "I900")) {
     arousal_N[i] <- length(data[data$valar == "Arousal", i][is.na(data[data$valar == "Arousal", i]) == FALSE])
}
arousal_N <- arousal_N[is.na(arousal_N) == FALSE]

table(arousal_N)

cor(valence_means, arousal_means)
cor.test(valence_means, arousal_means)

table(valence_means > 4, arousal_means > 4)


# Figure 4 ----------------------------------------------------------------
plot(valence_means, arousal_means, xlab = "Valence", ylab = "Arousal", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence and arousal ratings by image", col = "lightblue")
text(2, 6.5, bquote(R == .(format(round(cor(valence_means, arousal_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

# Valence and arousal by image category -----------------------------------

data_long <- read.csv("OASIS_data_long.csv", header = TRUE, sep = ",")

categories <- data_long$category[data_long$ID == data_long$ID[1]]
themes <- data_long$theme[data_long$ID == data_long$ID[1]]

means <- cbind(valence_means, arousal_means, categories)
means <- data.frame(means)
means$valence_means <- as.numeric(paste(means$valence_means))
means$arousal_means <- as.numeric(paste(means$arousal_means))
categories <- factor(categories)
colvector <- recode(categories, "'Animal'='seagreen'; 'Object'='royalblue4'; 'Person'='palevioletred1'; 'Scene'='yellow3'")
colvector <- paste(colvector)


# Figure 5 ----------------------------------------------------------------
plot(valence_means, arousal_means, xlab = "Valence", ylab = "Arousal", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence and arousal ratings by image category", col = colvector)
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)
legend("topright", levels(categories), pch = 1, col = c("seagreen", "royalblue4", "palevioletred1", "yellow3"), bty = "n")



# Relationship between valence and arousal --------------------------------

lmfit1 <- lm(valence_means ~ arousal_means, data = means)        # Linear
summary(lmfit1)
AIC(lmfit1)

lmfit2 <- lm(valence_means ~ arousal_means + I(arousal_means^2), data = means)       # Quadratic
summary(lmfit2)
AIC(lmfit2)

lmfit3 <- lm(valence_means ~ arousal_means + I(arousal_means^2) + I(arousal_means^3), data = means)      # Cubic
summary(lmfit3)
AIC(lmfit3)

plot(valence_means, arousal_means, xlab = "Valence", ylab = "Arousal", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence and arousal ratings by image", col = "blue")
abline(lmfit1, col = "orange")
lines(sort(arousal_means), fitted(lmfit2)[order(arousal_means)], col = "red")
lines(sort(arousal_means), fitted(lmfit3)[order(arousal_means)], col = "green")
legend("bottomright", c("Linear fit", "Quadratic fit", "Cubic fit"),
       col = c("orange", "red", "green"),
       lty = 1, bty = "n", cex = .7)


# Relationship between means and standard deviations ----------------------

valence_sd <- apply(data[data$valar == "Valence",
                    which(colnames(data) == "I1"):which(colnames(data) == "I900")], 2,
                    sd, na.rm = TRUE)
arousal_sd <- apply(data[data$valar == "Arousal",
                    which(colnames(data) == "I1"):which(colnames(data) == "I900")], 2,
                    sd, na.rm = TRUE)

# Figure 2 ----------------------------------------------------------------
op <- par(mfrow = c(2,2))
hist(valence_means, xlab = "Mean valence rating", main = "Valence ratings by image",
     xlim = c(1, 7), ylim = c(0, 100), breaks = seq(1, 7, by = 0.2))
hist(arousal_means, xlab = "Mean arousal rating", main = "Arousal ratings by image",
     xlim = c(1, 7), ylim = c(0, 100), breaks = seq(1, 7, by = 0.2))
hist(valence_sd, xlab = "SD of valence rating", main = "SD of valence ratings by image",
     xlim = c(0, 3), ylim = c(0, 200), breaks = seq(0, 3, by = 0.1))
hist(arousal_sd, xlab = "SD of arousal rating", main = "SD of arousal ratings by image",
     xlim = c(0, 3), ylim = c(0, 200), breaks = seq(0, 3, by = 0.1))
par(op)


# Descriptives ------------------------------------------------------------
range(valence_means)
mean(valence_means)
median(valence_sd)
ks.test(valence_means, "punif", 1, 7)

range(arousal_means)
mean(arousal_means)
median(arousal_sd)
ks.test(arousal_means, "pnorm")

ratings <- cbind(means, valence_sd, arousal_sd, themes, valence_N, arousal_N)
head(ratings)
ratings <- ratings[, c(which(colnames(ratings) == "themes"), which(colnames(ratings) == "categories"),
                     which(colnames(ratings) == "valence_means"), which(colnames(ratings) == "valence_sd"),
                     which(colnames(ratings) == "valence_N"), which(colnames(ratings) == "arousal_means"),
                     which(colnames(ratings) == "arousal_sd"), which(colnames(ratings) == "arousal_N"))]
colnames(ratings) <- c("Theme", "Category", "Valence_mean", "Valence_SD", "Valence_N", "Arousal_mean", "Arousal_SD", "Arousal_N")


# Highest and lowest valence and arousal means and SDs --------------------

head(ratings[order(ratings$Valence_mean), ])
tail(ratings[order(ratings$Valence_mean), ])
head(ratings[order(ratings$Arousal_mean), ])
tail(ratings[order(ratings$Arousal_mean), ])
head(ratings[order(ratings$Valence_SD), ])
tail(ratings[order(ratings$Valence_SD), ])
head(ratings[order(ratings$Arousal_SD), ])
tail(ratings[order(ratings$Arousal_SD), ])


# Relationship between valence mean and SD --------------------------------

valenceFit <- lm(Valence_SD ~ Valence_mean, data = ratings)
valenceFit2 <- lm(Valence_SD ~ Valence_mean + I(Valence_mean^2), data = ratings)
valenceFit3 <- lm(Valence_SD ~ Valence_mean + I(Valence_mean^2) + I(Valence_mean^3), data = ratings)

anova(valenceFit, valenceFit2)
anova(valenceFit2, valenceFit3)


# Relationship between arousal mean and SD --------------------------------
arousalFit <- lm(Arousal_SD ~ Arousal_mean, data = ratings)
arousalFit2 <- lm(Arousal_SD ~ Arousal_mean + I(Arousal_mean^2), data = ratings)
arousalFit3 <- lm(Arousal_SD ~ Arousal_mean + I(Arousal_mean^2) + I(Arousal_mean^3), data = ratings)

anova(arousalFit, arousalFit2)
anova(arousalFit2, arousalFit3)

# Figure 3 ----------------------------------------------------------------
op <- par(mfrow = c(1,2))
plot(ratings$Valence_mean, ratings$Valence_SD, xlab = "Valence mean", ylab = "Valence SD", xlim = c(1, 7), ylim = c(0, 2.5),
     main = "Valence SD by valence mean", col = "lightblue")
text(2, 2.25, bquote(italic(R^2) == .(format(summary(valenceFit3)$r.squared, digits = 3))))
lines(sort(ratings$Valence_mean), fitted(valenceFit3)[order(ratings$Valence_mean)], col = "orange", lty = 2, lwd = 2)
plot(ratings$Arousal_mean, ratings$Arousal_SD, xlab = "Arousal mean", ylab = "Arousal SD", xlim = c(1, 7), ylim = c(0, 2.5),
     main = "Arousal SD by arousal mean", col = "lightblue")
text(2, 2.25, bquote(italic(R^2) == .(format(summary(arousalFit2)$r.squared, digits = 3))))
lines(sort(ratings$Arousal_mean), fitted(arousalFit2)[order(ratings$Arousal_mean)], col = "orange", lty = 2, lwd = 2)
par(op)

# Analyses by gender ------------------------------------------------------

# Men, valence ------------------------------------------------------------
head(data)
valencemen <- data[data$valar == "Valence" & data$Gender == "Male", ]

valencemen_means <- apply(valencemen[which(colnames(valencemen) == "I1"):which(colnames(valencemen) == "I900")], 2,
                       mean, na.rm = TRUE)

valencemen_N <- vector()
i <- NULL
for (i in which(colnames(valencemen) == "I1"):which(colnames(data) == "I900")) {
     valencemen_N[i] <- length(valencemen[, i][is.na(valencemen[, i]) == FALSE])
}
valencemen_N <- valencemen_N[is.na(valencemen_N) == FALSE]

table(valencemen_N)


# Women, valence ----------------------------------------------------------
valencewomen <- data[data$valar == "Valence" & data$Gender == "Female", ]

valencewomen_means <- apply(valencewomen[which(colnames(valencewomen) == "I1"):which(colnames(valencewomen) == "I900")], 2,
                          mean, na.rm = TRUE)

valencewomen_N <- vector()
i <- NULL
for (i in which(colnames(valencewomen) == "I1"):which(colnames(data) == "I900")) {
     valencewomen_N[i] <- length(valencewomen[, i][is.na(valencewomen[, i]) == FALSE])
}
valencewomen_N <- valencewomen_N[is.na(valencewomen_N) == FALSE]

table(valencewomen_N)

cor(valencemen_means, valencewomen_means)
cor.test(valencemen_means, valencewomen_means)

plot(valencemen_means, valencewomen_means, xlab = "Men", ylab = "Women", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence ratings by gender", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(valencemen_means, valencewomen_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

gendVal <- lm(valencemen_means ~ valencewomen_means)
summary(gendVal)
sort(abs(gendVal$residuals))[1:10]
sort(abs(gendVal$residuals), decreasing = TRUE)[1:10]

valencemen_means[which(names(valencemen_means) == "I564"):which(names(valencemen_means) == "I585")]
valencewomen_means[which(names(valencewomen_means) == "I564"):which(names(valencewomen_means) == "I585")]


# Men, arousal ------------------------------------------------------------
head(data)
arousalmen <- data[data$valar == "Arousal" & data$Gender == "Male", ]

arousalmen_means <- apply(arousalmen[which(colnames(arousalmen) == "I1"):which(colnames(arousalmen) == "I900")], 2,
                          mean, na.rm = TRUE)

arousalmen_N <- vector()
i <- NULL
for (i in which(colnames(arousalmen) == "I1"):which(colnames(data) == "I900")) {
     arousalmen_N[i] <- length(arousalmen[, i][is.na(arousalmen[, i]) == FALSE])
}
arousalmen_N <- arousalmen_N[is.na(arousalmen_N) == FALSE]

table(arousalmen_N)


# Women, arousal ----------------------------------------------------------
arousalwomen <- data[data$valar == "Arousal" & data$Gender == "Female", ]

arousalwomen_means <- apply(arousalwomen[which(colnames(arousalwomen) == "I1"):which(colnames(arousalwomen) == "I900")], 2,
                            mean, na.rm = TRUE)

arousalwomen_N <- vector()
i <- NULL
for (i in which(colnames(arousalwomen) == "I1"):which(colnames(data) == "I900")) {
     arousalwomen_N[i] <- length(arousalwomen[, i][is.na(arousalwomen[, i]) == FALSE])
}
arousalwomen_N <- arousalwomen_N[is.na(arousalwomen_N) == FALSE]

table(arousalwomen_N)

cor(arousalmen_means, arousalwomen_means)
cor.test(arousalmen_means, arousalwomen_means)

r.test(900, cor(valencemen_means, valencewomen_means), cor(arousalmen_means, arousalwomen_means))

plot(arousalmen_means, arousalwomen_means, xlab = "Men", ylab = "Women", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean arousal ratings by gender", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(arousalmen_means, arousalwomen_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

gendAro <- lm(arousalmen_means ~ arousalwomen_means)
summary(gendAro)
sort(abs(gendAro$residuals))[1:10]
sort(abs(gendAro$residuals), decreasing = TRUE)[1:10]

arousalmen_means[which(names(arousalmen_means) == "I564"):which(names(arousalmen_means) == "I585")]
arousalwomen_means[which(names(arousalwomen_means) == "I564"):which(names(arousalwomen_means) == "I585")]

arousalmen_means[which(names(arousalmen_means) == "I541"):which(names(arousalmen_means) == "I563")]
arousalwomen_means[which(names(arousalwomen_means) == "I541"):which(names(arousalwomen_means) == "I563")]


valencemen_sd <- apply(valencemen[which(colnames(valencemen) == "I1"):which(colnames(valencemen) == "I900")], 2,
                         sd, na.rm = TRUE)
valencewomen_sd <- apply(valencewomen[which(colnames(valencewomen) == "I1"):which(colnames(valencewomen) == "I900")], 2,
                         sd, na.rm = TRUE)
arousalmen_sd <- apply(arousalmen[which(colnames(arousalmen) == "I1"):which(colnames(arousalmen) == "I900")], 2,
                         sd, na.rm = TRUE)
arousalwomen_sd <- apply(arousalwomen[which(colnames(arousalwomen) == "I1"):which(colnames(arousalwomen) == "I900")], 2,
                         sd, na.rm = TRUE)

# Same analysis after removing explicit images ----------------------------

# Men, valence ------------------------------------------------------------
head(data)
valencemen <- data[data$valar == "Valence" & data$Gender == "Male", ]
valencemen2 <- valencemen[, -c((which(colnames(valencemen) == "I527")):(which(colnames(valencemen) == "I585")))]

valencemen2_means <- apply(valencemen2[which(colnames(valencemen2) == "I1"):which(colnames(valencemen2) == "I900")], 2,
                          mean, na.rm = TRUE)

valencemen2_N <- vector()
i <- NULL
for (i in which(colnames(valencemen2) == "I1"):which(colnames(data) == "I900")) {
     valencemen2_N[i] <- length(valencemen2[, i][is.na(valencemen2[, i]) == FALSE])
}
valencemen2_N <- valencemen2_N[is.na(valencemen2_N) == FALSE]

table(valencemen2_N)


# Women, valence ----------------------------------------------------------
valencewomen <- data[data$valar == "Valence" & data$Gender == "Female", ]
valencewomen2 <- valencewomen[, -c((which(colnames(valencewomen) == "I527")):(which(colnames(valencewomen) == "I585")))]

valencewomen2_means <- apply(valencewomen2[which(colnames(valencewomen2) == "I1"):which(colnames(valencewomen2) == "I900")], 2,
                            mean, na.rm = TRUE)

valencewomen2_N <- vector()
i <- NULL
for (i in which(colnames(valencewomen2) == "I1"):which(colnames(data) == "I900")) {
     valencewomen2_N[i] <- length(valencewomen2[, i][is.na(valencewomen2[, i]) == FALSE])
}
valencewomen2_N <- valencewomen2_N[is.na(valencewomen2_N) == FALSE]

table(valencewomen2_N)

cor(valencemen2_means, valencewomen2_means)

plot(valencemen2_means, valencewomen2_means, xlab = "Men", ylab = "Women", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence ratings by gender\n(no sexual images)", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(valencemen2_means, valencewomen2_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

# Men, arousal ------------------------------------------------------------
head(data)
arousalmen <- data[data$valar == "Arousal" & data$Gender == "Male", ]
arousalmen2 <- arousalmen[, -c((which(colnames(arousalmen) == "I527")):(which(colnames(arousalmen) == "I585")))]

arousalmen2_means <- apply(arousalmen2[which(colnames(arousalmen2) == "I1"):which(colnames(arousalmen2) == "I900")], 2,
                          mean, na.rm = TRUE)

arousalmen2_N <- vector()
i <- NULL
for (i in which(colnames(arousalmen2) == "I1"):which(colnames(data) == "I900")) {
     arousalmen2_N[i] <- length(arousalmen2[, i][is.na(arousalmen2[, i]) == FALSE])
}
arousalmen2_N <- arousalmen2_N[is.na(arousalmen2_N) == FALSE]

table(arousalmen2_N)


# Women, arousal ----------------------------------------------------------
arousalwomen <- data[data$valar == "Arousal" & data$Gender == "Female", ]
arousalwomen2 <- arousalwomen[, -c((which(colnames(arousalwomen) == "I527")):(which(colnames(arousalwomen) == "I585")))]

arousalwomen2_means <- apply(arousalwomen2[which(colnames(arousalwomen2) == "I1"):which(colnames(arousalwomen2) == "I900")], 2,
                            mean, na.rm = TRUE)

arousalwomen2_N <- vector()
i <- NULL
for (i in which(colnames(arousalwomen2) == "I1"):which(colnames(data) == "I900")) {
     arousalwomen2_N[i] <- length(arousalwomen2[, i][is.na(arousalwomen2[, i]) == FALSE])
}
arousalwomen2_N <- arousalwomen2_N[is.na(arousalwomen2_N) == FALSE]

table(arousalwomen2_N)

cor(arousalmen2_means, arousalwomen2_means)
cor.test(arousalmen2_means, arousalwomen2_means)

plot(arousalmen2_means, arousalwomen2_means, xlab = "Men", ylab = "Women", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean arousal ratings by gender\n(no sexual images)", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(arousalmen2_means, arousalwomen2_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

gendAro2 <- lm(arousalmen2_means ~ arousalwomen2_means)
summary(gendAro2)
sort(abs(gendAro2$residuals), decreasing = TRUE)[1:10]


# Comparison with IAPS ----------------------------------------------------
# Note: If you would like to run this code, please obtain the data file from the Center for Emotion and Attention (CSEA) at the University of Florida
IAPS <- read.delim("AllSubjects_1-20.txt")
head(IAPS)
cor(IAPS$valmn, IAPS$aromn)
cor.test(IAPS$valmn, IAPS$aromn)


# IAPS univariate descriptives --------------------------------------------
op <- par(mfrow = c(2,2))
hist(IAPS$valmn, xlab = "Mean valence rating", main = "Valence ratings by image",
     xlim = c(1, 9), ylim = c(0, 100), breaks = seq(1, 9, by = 0.2))
hist(IAPS$aromn, xlab = "Mean arousal rating", main = "Arousal ratings by image",
     xlim = c(1, 9), ylim = c(0, 100), breaks = seq(1, 9, by = 0.2))
hist(IAPS$valsd, xlab = "SD of valence rating", main = "SD of valence ratings by image",
     xlim = c(0, 3), ylim = c(0, 250), breaks = seq(0, 3, by = 0.1))
hist(IAPS$arosd, xlab = "SD of arousal rating", main = "SD of arousal ratings by image",
     xlim = c(0, 3), ylim = c(0, 250), breaks = seq(0, 3, by = 0.1))
par(op)

min(IAPS$valsd)
max(IAPS$valsd)
max(IAPS$arosd)


# Do OASIS scores and IAPS scores come from the same distribution? --------
standValOASIS <- scale(ratings$Valence_mean)
standValIAPS <- scale(IAPS$valmn)

ks.test(standValOASIS, standValIAPS)

standAroOASIS <- scale(ratings$Arousal_mean)
standAroIAPS <- scale(IAPS$aromn)
ks.test(standAroOASIS, standAroIAPS)

range(IAPS$valmn)
range(IAPS$aromn)
median(IAPS$valsd)
median(IAPS$arosd)


# IAPS relationship between mean and SD -----------------------------------
valenceFitIAPS <- lm(valsd ~ valmn, data = IAPS)
valenceFitIAPS2 <- lm(valsd ~ valmn + I(valmn^2), data = IAPS)
valenceFitIAPS3 <- lm(valsd ~ valmn + I(valmn^2) + I(valmn^3), data = IAPS)

anova(valenceFitIAPS, valenceFitIAPS2)
anova(valenceFitIAPS2, valenceFitIAPS3)

arousalFitIAPS <- lm(arosd ~ aromn, data = IAPS)
arousalFitIAPS2 <- lm(arosd ~ aromn + I(aromn^2), data = IAPS)
arousalFitIAPS3 <- lm(arosd ~ aromn + I(aromn^2) + I(aromn^3), data = IAPS)

anova(arousalFitIAPS, arousalFitIAPS2)
anova(arousalFitIAPS2, arousalFitIAPS3)

op <- par(mfrow = c(1,2))
plot(IAPS$valmn, IAPS$valsd, xlab = "Valence mean", ylab = "Valence SD", xlim = c(1, 9), ylim = c(0, 3),
     main = "Valence SD by valence mean", col = "blue")
text(2, 2.75, bquote(italic(R^2) == .(format(summary(valenceFitIAPS3)$r.squared, digits = 3))))
lines(sort(IAPS$valmn), fitted(valenceFitIAPS3)[order(IAPS$valmn)], col = "orange", lty = 2)
plot(IAPS$aromn, IAPS$arosd, xlab = "Arousal mean", ylab = "Arousal SD", xlim = c(1, 9), ylim = c(0, 3),
     main = "Arousal SD by arousal mean", col = "blue")
text(2, 2.75, bquote(italic(R^2) == .(format(summary(arousalFitIAPS2)$r.squared, digits = 3))))
lines(sort(IAPS$aromn), fitted(arousalFitIAPS2)[order(IAPS$aromn)], col = "orange", lty = 2)
par(op)

op <- par(mfrow = c(1,2))
plot(valence_means, arousal_means, xlab = "Valence", ylab = "Arousal", xlim = c(1, 7), ylim = c(1, 7),
     main = "Mean valence and arousal ratings by image\n(OASIS)", col = "lightblue")
text(2, 6.5, bquote(R == .(format(round(cor(valence_means, arousal_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)
plot(IAPS$valmn, IAPS$aromn, xlab = "Valence", ylab = "Arousal", xlim = c(1, 9), ylim = c(1, 9),
     main = "Mean valence and arousal ratings by image\n(IAPS)", col = "lightblue")
text(2, 8.5, bquote(R == .(format(round(cor(IAPS$valmn, IAPS$aromn), 3)))))
abline(h = 5, lty = 2)
abline(v = 5, lty = 2)
par(op)


# Descriptives by category ------------------------------------------------
ratings$Category <- factor(ratings$Category, labels = c("Animal", "Object", "Person", "Scene"))

with(ratings, by(Valence_mean, Category, mean))
with(ratings, by(Valence_mean, Category, sd))

with(ratings, by(Arousal_mean, Category, mean))
with(ratings, by(Arousal_mean, Category, sd))

cor.test(ratings$Valence_mean[ratings$Category == "Animal"], ratings$Arousal_mean[ratings$Category == "Animal"])
cor.test(ratings$Valence_mean[ratings$Category == "Object"], ratings$Arousal_mean[ratings$Category == "Object"])
cor.test(ratings$Valence_mean[ratings$Category == "Person"], ratings$Arousal_mean[ratings$Category == "Person"])
cor.test(ratings$Valence_mean[ratings$Category == "Scene"], ratings$Arousal_mean[ratings$Category == "Scene"])

str(r.test(900, cor(arousalmen_means, arousalwomen_means), cor(arousalmen2_means, arousalwomen2_means), n2 = 841))


# Reliability -------------------------------------------------------------
set.seed(1234)


# Valence dimension -------------------------------------------------------
valence <- data[data$valar == "Valence", ]
valence <- valence[, which(colnames(valence) == "I1"):which(colnames(valence) == "I900")]
head(valence)
dim(valence)
randGroups <- c(rep(1, 207), rep(2, 206))
splitHalf1 <- list()
splitHalf2 <- list()
means1 <- list()
means2 <- list()
groupAssignment <- list()
reliVect <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignment[[i]] <- sample(randGroups)
     splitHalf1[[i]] <- valence[groupAssignment[[i]] == 1, ]
     splitHalf2[[i]] <- valence[groupAssignment[[i]] == 2, ]
     means1[[i]] <- apply(splitHalf1[[i]], 2, mean, na.rm = TRUE)
     means2[[i]] <- apply(splitHalf2[[i]], 2, mean, na.rm = TRUE)
     reliVect[i] <- cor(means1[[i]], means2[[i]])
     print(i)
}

reliVal <- mean(reliVect)
sd(reliVect)
range(reliVect)


# Arousal dimension -------------------------------------------------------
arousal <- data[data$valar == "Arousal", ]
arousal <- arousal[, which(colnames(arousal) == "I1"):which(colnames(arousal) == "I900")]
head(arousal)
dim(arousal)
randGroupsAr <- c(rep(1, 205), rep(2, 204))
splitHalf1Ar <- list()
splitHalf2Ar <- list()
means1Ar <- list()
means2Ar <- list()
groupAssignmentAr <- list()
reliVectAr <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignmentAr[[i]] <- sample(randGroupsAr)
     splitHalf1Ar[[i]] <- arousal[groupAssignmentAr[[i]] == 1, ]
     splitHalf2Ar[[i]] <- arousal[groupAssignmentAr[[i]] == 2, ]
     means1Ar[[i]] <- apply(splitHalf1Ar[[i]], 2, mean, na.rm = TRUE)
     means2Ar[[i]] <- apply(splitHalf2Ar[[i]], 2, mean, na.rm = TRUE)
     reliVectAr[i] <- cor(means1Ar[[i]], means2Ar[[i]])
     print(i)
}

reliAr <- mean(reliVectAr)
sd(reliVectAr)
range(reliVectAr)

bins <- seq(0.65, 1, by = 0.003)


op <- par(mfrow = c(1,2))
hist(reliVect, xlim = c(0.65, 1), breaks = bins, main = "Distribution of split-half correlations\n(valence)", xlab = "Correlation")
abline(v = reliVal, col = "red")
hist(reliVectAr, xlim = c(0.65, 1), breaks = bins, main = "Distribution of split-half correlations\n(arousal)", xlab = "Correlation")
abline(v = reliAr, col = "red")
par(op)


# Only men, valence -------------------------------------------------------
valenceMen <- data[data$valar == "Valence" & data$Gender == "Male", ]
valenceMen <- valenceMen[, which(colnames(valenceMen) == "I1"):which(colnames(valenceMen) == "I900")]
head(valenceMen)
dim(valenceMen)
randGroupsMen <- c(rep(1, 103), rep(2, 102))
splitHalf1Men <- list()
splitHalf2Men <- list()
means1Men <- list()
means2Men <- list()
groupAssignmentMen <- list()
reliVectMen <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignmentMen[[i]] <- sample(randGroupsMen)
     splitHalf1Men[[i]] <- valenceMen[groupAssignmentMen[[i]] == 1, ]
     splitHalf2Men[[i]] <- valenceMen[groupAssignmentMen[[i]] == 2, ]
     means1Men[[i]] <- apply(splitHalf1Men[[i]], 2, mean, na.rm = TRUE)
     means2Men[[i]] <- apply(splitHalf2Men[[i]], 2, mean, na.rm = TRUE)
     reliVectMen[i] <- cor(means1Men[[i]], means2Men[[i]])
     print(i)
}

reliValenceMen <- mean(reliVectMen)

valenceWomen <- data[data$valar == "Valence" & data$Gender == "Female", ]
valenceWomen <- valenceWomen[, which(colnames(valenceWomen) == "I1"):which(colnames(valenceWomen) == "I900")]
head(valenceWomen)
dim(valenceWomen)
randGroupsWomen <- c(rep(1, 105), rep(2, 104))
splitHalf1Women <- list()
splitHalf2Women <- list()
means1Women <- list()
means2Women <- list()
groupAssignmentWomen <- list()
reliVectWomen <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignmentWomen[[i]] <- sample(randGroupsWomen)
     splitHalf1Women[[i]] <- valenceWomen[groupAssignmentWomen[[i]] == 1, ]
     splitHalf2Women[[i]] <- valenceWomen[groupAssignmentWomen[[i]] == 2, ]
     means1Women[[i]] <- apply(splitHalf1Women[[i]], 2, mean, na.rm = TRUE)
     means2Women[[i]] <- apply(splitHalf2Women[[i]], 2, mean, na.rm = TRUE)
     reliVectWomen[i] <- cor(means1Women[[i]], means2Women[[i]])
     print(i)
}

reliValenceWomen <- mean(reliVectWomen)


# Arousal, men ------------------------------------------------------------
arousalMen <- data[data$valar == "Arousal" & data$Gender == "Male", ]
arousalMen <- arousalMen[, which(colnames(arousalMen) == "I1"):which(colnames(arousalMen) == "I900")]
head(arousalMen)
dim(arousalMen)
randGroupsArMen <- c(rep(1, 99), rep(2, 98))
splitHalf1ArMen <- list()
splitHalf2ArMen <- list()
means1ArMen <- list()
means2ArMen <- list()
groupAssignmentArMen <- list()
reliVectArMen <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignmentArMen[[i]] <- sample(randGroupsArMen)
     splitHalf1ArMen[[i]] <- arousal[groupAssignmentArMen[[i]] == 1, ]
     splitHalf2ArMen[[i]] <- arousal[groupAssignmentArMen[[i]] == 2, ]
     means1ArMen[[i]] <- apply(splitHalf1ArMen[[i]], 2, mean, na.rm = TRUE)
     means2ArMen[[i]] <- apply(splitHalf2ArMen[[i]], 2, mean, na.rm = TRUE)
     reliVectArMen[i] <- cor(means1ArMen[[i]], means2ArMen[[i]])
     print(i)
}

reliArousalMen <- mean(reliVectArMen)


# Arousal, women ----------------------------------------------------------
arousalWomen <- data[data$valar == "Arousal" & data$Gender == "Female", ]
arousalWomen <- arousalWomen[, which(colnames(arousalWomen) == "I1"):which(colnames(arousalWomen) == "I900")]
head(arousalWomen)
dim(arousalWomen)
randGroupsArWomen <- c(rep(1, 108), rep(2, 107))
splitHalf1ArWomen <- list()
splitHalf2ArWomen <- list()
means1ArWomen <- list()
means2ArWomen <- list()
groupAssignmentArWomen <- list()
reliVectArWomen <- vector()
i <- NULL
for (i in 1:1000) {
     groupAssignmentArWomen[[i]] <- sample(randGroupsArWomen)
     splitHalf1ArWomen[[i]] <- arousal[groupAssignmentArWomen[[i]] == 1, ]
     splitHalf2ArWomen[[i]] <- arousal[groupAssignmentArWomen[[i]] == 2, ]
     means1ArWomen[[i]] <- apply(splitHalf1ArWomen[[i]], 2, mean, na.rm = TRUE)
     means2ArWomen[[i]] <- apply(splitHalf2ArWomen[[i]], 2, mean, na.rm = TRUE)
     reliVectArWomen[i] <- cor(means1ArWomen[[i]], means2ArWomen[[i]])
     print(i)
}

reliArousalWomen <- mean(reliVectArWomen)


# Correcting reliabilities using the Spearman–Brown prophecy formula -----------------------------------------
spearmanBrown <- function(rho, K) {
     (K*rho)/(1+(K-1)*rho)
}

reliVal
KvalenceMen <- dim(data[data$valar == "Valence", ])[1]/dim(data[data$valar == "Valence" & data$Gender == "Male", ])[1]
reliValenceMen
reliValenceMenCorr <- spearmanBrown(reliValenceMen, KvalenceMen)

KvalenceWomen <- dim(data[data$valar == "Valence", ])[1]/dim(data[data$valar == "Valence" & data$Gender == "Female", ])[1]
reliValenceWomen
reliValenceWomenCorr <- spearmanBrown(reliValenceWomen, KvalenceWomen)

reliAr
KvalenceMen <- dim(data[data$valar == "Arousal", ])[1]/dim(data[data$valar == "Arousal" & data$Gender == "Male", ])[1]
reliArousalMen
reliArousalMenCorr <- spearmanBrown(reliArousalMen, KvalenceMen)

KvalenceWomen <- dim(data[data$valar == "Arousal", ])[1]/dim(data[data$valar == "Arousal" & data$Gender == "Female", ])[1]
reliArousalWomen
reliArousalWomenCorr <- spearmanBrown(reliArousalWomen, KvalenceWomen)

valenceRel <- cbind(reliVal, reliValenceMen, reliValenceMenCorr, reliValenceWomen, reliValenceWomenCorr)
arousalRel <- cbind(reliAr, reliArousalMen, reliArousalMenCorr, reliArousalWomen, reliArousalWomenCorr)
reliabilityTable <- rbind(valenceRel, arousalRel)
rownames(reliabilityTable) <- c("Valence", "Arousal")
colnames(reliabilityTable) <- c("Overall", "Men unadjusted", "Men adjusted", "Women unadjusted", "Women adjusted")

reliabilityTable

sd(reliVect)
range(reliVect)

sd(reliVectAr)
range(reliVectAr)

sd(reliVectArWomen)
range(reliVectArWomen)

sd(reliVectArMen)
range(reliVectArMen)


# Analyses by demographic variables ---------------------------------------

# Age ---------------------------------------------------------------------

# Young, valence ----------------------------------------------------------
head(data)
valenceyoung <- data[data$valar == "Valence" & data$Age <= 34, ]

valenceyoung_means <- apply(valenceyoung[which(colnames(valenceyoung) == "I1"):which(colnames(valenceyoung) == "I900")], 2,
                          mean, na.rm = TRUE)

valenceyoung_N <- vector()
i <- NULL
for (i in which(colnames(valenceyoung) == "I1"):which(colnames(data) == "I900")) {
     valenceyoung_N[i] <- length(valenceyoung[, i][is.na(valenceyoung[, i]) == FALSE])
}
valenceyoung_N <- valenceyoung_N[is.na(valenceyoung_N) == FALSE]

table(valenceyoung_N)


# Old, valence ------------------------------------------------------------
valenceold <- data[data$valar == "Valence" & data$Age > 34, ]

valenceold_means <- apply(valenceold[which(colnames(valenceold) == "I1"):which(colnames(valenceold) == "I900")], 2,
                            mean, na.rm = TRUE)

valenceold_N <- vector()
i <- NULL
for (i in which(colnames(valenceold) == "I1"):which(colnames(data) == "I900")) {
     valenceold_N[i] <- length(valenceold[, i][is.na(valenceold[, i]) == FALSE])
}
valenceold_N <- valenceold_N[is.na(valenceold_N) == FALSE]

table(valenceold_N)

cor(valenceyoung_means, valenceold_means)
cor.test(valenceyoung_means, valenceold_means)

t.test(valenceyoung_means, valenceold_means, paired = TRUE)

plot(valenceyoung_means, valenceold_means, xlab = "Older participants", ylab = "Younger participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence ratings by age", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(valenceyoung_means, valenceold_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

ageVal <- lm(valenceyoung_means ~ valenceold_means)
summary(ageVal)
sort(abs(ageVal$residuals))[1:10]
sort(abs(ageVal$residuals), decreasing = TRUE)[1:10]

ageVal$residuals[names(ageVal$residuals) %in% names(sort(abs(ageVal$residuals), decreasing = TRUE)[1:10])]

# Young, arousal ----------------------------------------------------------
head(data)
arousalyoung <- data[data$valar == "Arousal" & data$Age <= 34,  ]

arousalyoung_means <- apply(arousalyoung[which(colnames(arousalyoung) == "I1"):which(colnames(arousalyoung) == "I900")], 2,
                          mean, na.rm = TRUE)

arousalyoung_N <- vector()
i <- NULL
for (i in which(colnames(arousalyoung) == "I1"):which(colnames(data) == "I900")) {
     arousalyoung_N[i] <- length(arousalyoung[, i][is.na(arousalyoung[, i]) == FALSE])
}
arousalyoung_N <- arousalyoung_N[is.na(arousalyoung_N) == FALSE]

table(arousalyoung_N)


# Old, arousal ------------------------------------------------------------
arousalold <- data[data$valar == "Arousal" & data$Age > 34,  ]

arousalold_means <- apply(arousalold[which(colnames(arousalold) == "I1"):which(colnames(arousalold) == "I900")], 2,
                            mean, na.rm = TRUE)

arousalold_N <- vector()
i <- NULL
for (i in which(colnames(arousalold) == "I1"):which(colnames(data) == "I900")) {
     arousalold_N[i] <- length(arousalold[, i][is.na(arousalold[, i]) == FALSE])
}
arousalold_N <- arousalold_N[is.na(arousalold_N) == FALSE]

table(arousalold_N)

cor(arousalyoung_means, arousalold_means)
cor.test(arousalyoung_means, arousalold_means)

t.test(arousalyoung_means, arousalold_means, paired = TRUE)

r.test(900, cor(valenceyoung_means, valenceold_means), cor(arousalyoung_means, arousalold_means))
r.test(900, cor(valenceyoung_means, arousalyoung_means), cor(valenceold_means, arousalold_means))

plot(arousalyoung_means, arousalold_means, xlab = "Older participants", ylab = "Younger participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean arousal ratings by age", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(arousalyoung_means, arousalold_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

ageAro <- lm(arousalyoung_means ~ arousalold_means)
summary(ageAro)
sort(abs(ageAro$residuals))[1:10]
sort(abs(ageAro$residuals), decreasing = TRUE)[1:10]

ageAro$residuals[names(ageAro$residuals) %in% names(sort(abs(ageAro$residuals), decreasing = TRUE)[1:10])]
ratings[ratings$ID %in% names(sort(abs(ageAro$residuals), decreasing = TRUE)[1:10]), ]
arousalyoung_means[names(arousalyoung_means) %in% names(sort(abs(ageAro$residuals), decreasing = TRUE)[1:10])]
arousalold_means[names(arousalold_means) %in% names(sort(abs(ageAro$residuals), decreasing = TRUE)[1:10])]

youngNude <- arousalyoung_means[527:585]
oldNude <- arousalold_means[527:585]

t.test(youngNude, oldNude, paired = TRUE)
mean(youngNude)
mean(oldNude)

valenceyoung_sd <- apply(valenceyoung[which(colnames(valenceyoung) == "I1"):which(colnames(valenceyoung) == "I900")], 2,
                       sd, na.rm = TRUE)
valenceold_sd <- apply(valenceold[which(colnames(valenceold) == "I1"):which(colnames(valenceold) == "I900")], 2,
                         sd, na.rm = TRUE)
arousalyoung_sd <- apply(arousalyoung[which(colnames(arousalyoung) == "I1"):which(colnames(arousalyoung) == "I900")], 2,
                       sd, na.rm = TRUE)
arousalold_sd <- apply(arousalold[which(colnames(arousalold) == "I1"):which(colnames(arousalold) == "I900")], 2,
                         sd, na.rm = TRUE)

# Ideology ----------------------------------------------------------------

# Liberal, valence --------------------------------------------------------
head(data)
valenceliberal <- data[data$valar == "Valence" & data$Ideol == "Slightly liberal"
                       | data$Ideol == "Moderately liberal" | data$Ideol == "Strongly liberal", ]

valenceliberal_means <- apply(valenceliberal[which(colnames(valenceliberal) == "I1"):which(colnames(valenceliberal) == "I900")], 2,
                            mean, na.rm = TRUE)

valenceliberal_N <- vector()
i <- NULL
for (i in which(colnames(valenceliberal) == "I1"):which(colnames(data) == "I900")) {
     valenceliberal_N[i] <- length(valenceliberal[, i][is.na(valenceliberal[, i]) == FALSE])
}
valenceliberal_N <- valenceliberal_N[is.na(valenceliberal_N) == FALSE]

table(valenceliberal_N)


# Conservative, valence ---------------------------------------------------
valencecons <- data[data$valar == "Valence" & data$Ideol == "Slightly conservative"
                    | data$Ideol == "Moderately conservative" | data$Ideol == "Strongly conservative", ]

valencecons_means <- apply(valencecons[which(colnames(valencecons) == "I1"):which(colnames(valencecons) == "I900")], 2,
                          mean, na.rm = TRUE)

valencecons_N <- vector()
i <- NULL
for (i in which(colnames(valencecons) == "I1"):which(colnames(data) == "I900")) {
     valencecons_N[i] <- length(valencecons[, i][is.na(valencecons[, i]) == FALSE])
}
valencecons_N <- valencecons_N[is.na(valencecons_N) == FALSE]

table(valencecons_N)

cor(valenceliberal_means, valencecons_means)
cor.test(valenceliberal_means, valencecons_means)

t.test(valenceliberal_means, valencecons_means, paired = TRUE)

plot(valenceliberal_means, valencecons_means, xlab = "Conservative participants", ylab = "Liberal participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence ratings by ideology", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(valenceliberal_means, valencecons_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

ideolVal <- lm(valenceliberal_means ~ valencecons_means)
summary(ideolVal)
sort(abs(ideolVal$residuals))[1:15]
sort(abs(ideolVal$residuals), decreasing = TRUE)[1:15]

ideolVal$residuals[names(ideolVal$residuals) %in% names(sort(abs(ideolVal$residuals), decreasing = TRUE)[1:20])]


# Liberal, arousal --------------------------------------------------------
head(data)
arousalliberal <- data[data$valar == "Arousal" & data$Ideol == "Slightly liberal"
                       | data$Ideol == "Moderately liberal" | data$Ideol == "Strongly liberal", ]

arousalliberal_means <- apply(arousalliberal[which(colnames(arousalliberal) == "I1"):which(colnames(arousalliberal) == "I900")], 2,
                            mean, na.rm = TRUE)

arousalliberal_N <- vector()
i <- NULL
for (i in which(colnames(arousalliberal) == "I1"):which(colnames(data) == "I900")) {
     arousalliberal_N[i] <- length(arousalliberal[, i][is.na(arousalliberal[, i]) == FALSE])
}
arousalliberal_N <- arousalliberal_N[is.na(arousalliberal_N) == FALSE]

table(arousalliberal_N)


# Conservative, arousal ---------------------------------------------------
arousalcons <- data[data$valar == "Arousal" & data$Ideol == "Slightly conservative"
                    | data$Ideol == "Moderately conservative" | data$Ideol == "Strongly conservative", ]

arousalcons_means <- apply(arousalcons[which(colnames(arousalcons) == "I1"):which(colnames(arousalcons) == "I900")], 2,
                          mean, na.rm = TRUE)

arousalcons_N <- vector()
i <- NULL
for (i in which(colnames(arousalcons) == "I1"):which(colnames(data) == "I900")) {
     arousalcons_N[i] <- length(arousalcons[, i][is.na(arousalcons[, i]) == FALSE])
}
arousalcons_N <- arousalcons_N[is.na(arousalcons_N) == FALSE]

table(arousalcons_N)

cor(arousalliberal_means, arousalcons_means)
cor.test(arousalliberal_means, arousalcons_means)

t.test(arousalliberal_means, arousalcons_means, paired = TRUE)

r.test(900, cor(valenceliberal_means, valencecons_means), cor(arousalliberal_means, arousalcons_means))
r.test(900, cor(valenceliberal_means, arousalliberal_means), cor(valencecons_means, arousalcons_means))

plot(arousalliberal_means, arousalcons_means, xlab = "Conservative participants", ylab = "Liberal participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean arousal ratings by ideology", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(arousalliberal_means, arousalcons_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

ideolAro <- lm(arousalliberal_means ~ arousalcons_means)
summary(ideolAro)
sort(abs(ideolAro$residuals))[1:10]
sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10]

ideolAro$residuals[names(ideolAro$residuals) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]
ratings[ratings$ID %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10]), ]
arousalliberal_means[names(arousalliberal_means) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]
arousalcons_means[names(arousalcons_means) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]

liberalNude <- arousalliberal_means[527:585]
consNude <- arousalcons_means[527:585]

t.test(liberalNude, consNude, paired = TRUE)
mean(liberalNude)
mean(consNude)

valenceliberal_sd <- apply(valenceliberal[which(colnames(valenceliberal) == "I1"):which(colnames(valenceliberal) == "I900")], 2,
                         sd, na.rm = TRUE)
valencecons_sd <- apply(valencecons[which(colnames(valencecons) == "I1"):which(colnames(valencecons) == "I900")], 2,
                       sd, na.rm = TRUE)
arousalliberal_sd <- apply(arousalliberal[which(colnames(arousalliberal) == "I1"):which(colnames(arousalliberal) == "I900")], 2,
                         sd, na.rm = TRUE)
arousalcons_sd <- apply(arousalcons[which(colnames(arousalcons) == "I1"):which(colnames(arousalcons) == "I900")], 2,
                       sd, na.rm = TRUE)

levels(data$Ideol)[8] <- "Dont know/Prefer not to answer"
ideolBinary <- recode(data$Ideol, "'Strongly conservative' = 1; 'Moderately conservative' = 1;
'Slightly conservative' = 1; 'Neutral, middle of the road' = NA; 'Slightly liberal' = 2;
'Moderately liberal' = 2; 'Strongly liberal' = 2; 'Dont know/Prefer not to answer' = NA")
table(ideolBinary)
chisq.test(table(ideolBinary))


# Income ------------------------------------------------------------------

# Low-income, valence -----------------------------------------------------
head(data)
table(data$Income)
valencelowinc <- data[data$valar == "Valence" & data$Income == "below $25,000" | data$Income == "$25,000 to $44,999", ]

valencelowinc_means <- apply(valencelowinc[which(colnames(valencelowinc) == "I1"):which(colnames(valencelowinc) == "I900")], 2,
                              mean, na.rm = TRUE)

valencelowinc_N <- vector()
i <- NULL
for (i in which(colnames(valencelowinc) == "I1"):which(colnames(data) == "I900")) {
     valencelowinc_N[i] <- length(valencelowinc[, i][is.na(valencelowinc[, i]) == FALSE])
}
valencelowinc_N <- valencelowinc_N[is.na(valencelowinc_N) == FALSE]

table(valencelowinc_N)


# High-income, valence ----------------------------------------------------
valencehighinc <- data[data$valar == "Valence" & data$Income == "$50,000 to $69,999" | data$Income == "$70,000 to $99,999"
                       | data$Income == "$100,000 or above", ]

valencehighinc_means <- apply(valencehighinc[which(colnames(valencehighinc) == "I1"):which(colnames(valencehighinc) == "I900")], 2,
                           mean, na.rm = TRUE)

valencehighinc_N <- vector()
i <- NULL
for (i in which(colnames(valencehighinc) == "I1"):which(colnames(data) == "I900")) {
     valencehighinc_N[i] <- length(valencehighinc[, i][is.na(valencehighinc[, i]) == FALSE])
}
valencehighinc_N <- valencehighinc_N[is.na(valencehighinc_N) == FALSE]

table(valencehighinc_N)

cor(valencelowinc_means, valencehighinc_means)
cor.test(valencelowinc_means, valencehighinc_means)

t.test(valencelowinc_means, valencehighinc_means, paired = TRUE)

plot(valencelowinc_means, valencehighinc_means, xlab = "High-income participants", ylab = "Low-income participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean valence ratings by income", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(valencelowinc_means, valencehighinc_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

incomeVal <- lm(valencelowinc_means ~ valencehighinc_means)
summary(incomeVal)
sort(abs(incomeVal$residuals))[1:15]
sort(abs(incomeVal$residuals), decreasing = TRUE)[1:15]

incomeVal$residuals[names(incomeVal$residuals) %in% names(sort(abs(incomeVal$residuals), decreasing = TRUE)[1:20])]

# Low-income, arousal ----------------------------------------------------
head(data)
arousallowinc <- data[data$valar == "Arousal" & data$Income == "below $25,000" | data$Income == "$25,000 to $44,999", ]

arousallowinc_means <- apply(arousallowinc[which(colnames(arousallowinc) == "I1"):which(colnames(arousallowinc) == "I900")], 2,
                              mean, na.rm = TRUE)

arousallowinc_N <- vector()
i <- NULL
for (i in which(colnames(arousallowinc) == "I1"):which(colnames(data) == "I900")) {
     arousallowinc_N[i] <- length(arousallowinc[, i][is.na(arousallowinc[, i]) == FALSE])
}
arousallowinc_N <- arousallowinc_N[is.na(arousallowinc_N) == FALSE]

table(arousallowinc_N)

# High-income, arousal ----------------------------------------------------
arousalhighinc <- data[data$valar == "Arousal" & data$Income == "$50,000 to $69,999" | data$Income == "$70,000 to $99,999"
                       | data$Income == "$100,000 or above", ]

arousalhighinc_means <- apply(arousalhighinc[which(colnames(arousalhighinc) == "I1"):which(colnames(arousalhighinc) == "I900")], 2,
                           mean, na.rm = TRUE)

arousalhighinc_N <- vector()
i <- NULL
for (i in which(colnames(arousalhighinc) == "I1"):which(colnames(data) == "I900")) {
     arousalhighinc_N[i] <- length(arousalhighinc[, i][is.na(arousalhighinc[, i]) == FALSE])
}
arousalhighinc_N <- arousalhighinc_N[is.na(arousalhighinc_N) == FALSE]

table(arousalhighinc_N)

cor(arousallowinc_means, arousalhighinc_means)
cor.test(arousallowinc_means, arousalhighinc_means)

t.test(arousallowinc_means, arousalhighinc_means, paired = TRUE)


plot(arousallowinc_means, arousalhighinc_means, xlab = "High-income participants", ylab = "Low-income participants", xlim = c(1, 7), ylim = c(1,7),
     main = "Mean arousal ratings by income", col = "blue")
text(2, 6.5, bquote(R == .(format(round(cor(arousallowinc_means, arousalhighinc_means), 3)))))
abline(h = 4, lty = 2)
abline(v = 4, lty = 2)

incomeAro <- lm(arousallowinc_means ~ arousalhighinc_means)
summary(incomeAro)
sort(abs(incomeAro$residuals))[1:10]
sort(abs(incomeAro$residuals), decreasing = TRUE)[1:10]

incomeAro$residuals[names(incomeAro$residuals) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]
arousallowinc_means[names(arousallowinc_means) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]
arousalhighinc_means[names(arousalhighinc_means) %in% names(sort(abs(ideolAro$residuals), decreasing = TRUE)[1:10])]

# Treating age and ideology as metric -------------------------------------

valenceOnly <- data[data$valar == "Valence", ]
corMatrVal <- matrix(nrow = dim(valenceOnly)[1], ncol = dim(valenceOnly)[1])
for (i in 1:dim(valenceOnly)[1]) {
     for (j in 1:dim(valenceOnly)[1]) {
          corMatrVal[i, j] <- cor(as.numeric(valenceOnly[i, which(colnames(valenceOnly) == "I1"):which(colnames(valenceOnly) == "I900")]),
                                as.numeric(valenceOnly[j, which(colnames(valenceOnly) == "I1"):which(colnames(valenceOnly) == "I900")]), use = "pairwise.complete.obs")
          print(i); print(j)
     }
}

arousalOnly <- data[data$valar == "Arousal", ]
corMatrAr <- matrix(nrow = dim(arousalOnly)[1], ncol = dim(arousalOnly)[1])
for (i in 1:dim(arousalOnly)[1]) {
     for (j in 1:dim(arousalOnly)[1]) {
          corMatrAr[i, j] <- cor(as.numeric(arousalOnly[i, which(colnames(arousalOnly) == "I1"):which(colnames(arousalOnly) == "I900")]),
                                  as.numeric(arousalOnly[j, which(colnames(arousalOnly) == "I1"):which(colnames(arousalOnly) == "I900")]), use = "pairwise.complete.obs")
          print(i); print(j)
     }
}

ageDiffVal <- matrix(nrow = dim(valenceOnly)[1], ncol = dim(valenceOnly)[1])
for (i in 1:dim(valenceOnly)[1]) {
     for (j in 1:dim(valenceOnly)[1]) {
          ageDiffVal[i, j] <- abs(valenceOnly$Age[i] - valenceOnly$Age[j])
          print(i); print(j)
     }
}

ageDiffAr <- matrix(nrow = dim(arousalOnly)[1], ncol = dim(arousalOnly)[1])
for (i in 1:dim(arousalOnly)[1]) {
     for (j in 1:dim(arousalOnly)[1]) {
          ageDiffAr[i, j] <- abs(arousalOnly$Age[i] - arousalOnly$Age[j])
          print(i); print(j)
     }
}

valenceOnly$ideolNum <- recode(valenceOnly$Ideol, "'Strongly conservative'=1;
                               'Moderately conservative'=2; 'Slightly conservative'=3;
                               'Neutral, middle of the road'=4; 'Slightly liberal'=5;
                               'Moderately liberal'=6; 'Strongly liberal'=7; else = NA")
valenceOnly$ideolNum <- as.numeric(paste(valenceOnly$ideolNum))

arousalOnly$ideolNum <- recode(arousalOnly$Ideol, "'Strongly conservative'=1;
                               'Moderately conservative'=2; 'Slightly conservative'=3;
                               'Neutral, middle of the road'=4; 'Slightly liberal'=5;
                               'Moderately liberal'=6; 'Strongly liberal'=7; else = NA")
arousalOnly$ideolNum <- as.numeric(paste(arousalOnly$ideolNum))

ideolDiffVal <- matrix(nrow = dim(valenceOnly)[1], ncol = dim(valenceOnly)[1])
for (i in 1:dim(valenceOnly)[1]) {
     for (j in 1:dim(valenceOnly)[1]) {
          ideolDiffVal[i, j] <- abs(valenceOnly$ideolNum[i] - valenceOnly$ideolNum[j])
          print(i); print(j)
     }
}

ideolDiffAr <- matrix(nrow = dim(arousalOnly)[1], ncol = dim(arousalOnly)[1])
for (i in 1:dim(arousalOnly)[1]) {
     for (j in 1:dim(arousalOnly)[1]) {
          ideolDiffAr[i, j] <- abs(arousalOnly$ideolNum[i] - arousalOnly$ideolNum[j])
          print(i); print(j)
     }
}


corLowerVal <- corMatrVal[lower.tri(corMatrVal)]
corLowerAr <- corMatrAr[lower.tri(corMatrAr)]
corLowerValAr <- c(corLowerVal, corLowerAr)

ageDiffVectVal <- ageDiffVal[lower.tri(ageDiffVal)]
ageDiffVectAr <- ageDiffAr[lower.tri(ageDiffAr)]
ageDiffVectValAr <- c(ageDiffVectVal, ageDiffVectAr)

ideolDiffVectVal <- ideolDiffVal[lower.tri(ideolDiffVal)]
ideolDiffVectAr <- ideolDiffAr[lower.tri(ideolDiffAr)]
ideolDiffVectValAr <- c(ideolDiffVectVal, ideolDiffVectAr)

# Correlation for age
hist(corLowerValAr, breaks = 10, main = "Distribution of pairwise correlations", xlab = "Pairwise correlation")
mean(corLowerValAr, na.rm = TRUE)
sd(corLowerValAr, na.rm = TRUE)
median(corLowerValAr, na.rm = TRUE)

mean(corLowerVal, na.rm = TRUE)
sd(corLowerVal, na.rm = TRUE)
median(corLowerVal, na.rm = TRUE)

mean(corLowerAr, na.rm = TRUE)
sd(corLowerAr, na.rm = TRUE)
median(corLowerAr, na.rm = TRUE)

r.test(length(corLowerVal[!is.na(corLowerVal)]), mean(corLowerVal, na.rm = TRUE),
       mean(corLowerAr, na.rm = TRUE), n2 = length(corLowerAr[!is.na(corLowerAr)]))


plot(ageDiffVectValAr, corLowerValAr, main = "Relationship between age differences and pairwise correlations",
     col = rainbow(2)[2], bg = rainbow(2, alpha = 0.3)[2], cex = 0.8, xlab = "Age differences",
     ylab = "Pairwise correlations")
cor(ageDiffVectValAr, corLowerValAr, use = "pairwise.complete.obs")


plot(ideolDiffVectValAr, corLowerValAr, main = "Relationship between ideology distance and pairwise correlations",
     col = rainbow(3)[3], bg = rainbow(3, alpha = 0.3)[3], cex = 0.8, xlab = "Ideology distance",
     ylab = "Pairwise correlations")
cor(ideolDiffVectValAr, corLowerValAr, use = "pairwise.complete.obs", method = "spearman")

# Were lists biased? ------------------------------------------------------

## Fitting the multilevel model
head(data_long)
str(data_long)
data_long$rating <- as.numeric(paste(data_long$rating))
data_long$ID <- factor(data_long$ID)
data_long$theme <- factor(data_long$theme)
data_long$category <- factor(data_long$category)


## Intercept only model (i.e., no predictors)
fit0 <- gls(rating ~ 1, data = data_long, na.action = na.exclude)
summary(fit0)

## Random intercept for subject
fit1 <- lmer(rating ~ 1 + (1 | ID), data = data_long)
anova(fit0, fit1)
summary(fit1)

## Random intercept for subject and image
fit2 <- lmer(rating ~ 1 + (1 | ID) + (1 | theme), data = data_long)
anova(fit1, fit2)
summary(fit2)

## List
fit3 <- lmer(rating ~ List + (1 | ID) + (1 | theme), data = data_long)
anova(fit2, fit3)
summary(fit3)


# Lists & categories ------------------------------------------------------

longNa <- data_long[!(is.na(data_long$rating)), ]
table(data$List)

listCat <- table(longNa$List, longNa$category)
for (i in 1:4) {
     listCat[i, ] <- table(longNa$List, longNa$category)[i, ]/table(data$List)[i]
}

listCat <- round(listCat)
chisq.test(listCat)


# Gender effects ----------------------------------------------------------

## Random intercept for subject
genderFit1 <- lmer(rating ~ 1 + (1 | ID), data = data_long, REML = FALSE)
summary(genderFit1)

## Random intercept for subject and image
genderFit2 <- lmer(rating ~ 1 + (1 | ID) + (1 | theme), data = data_long, REML = FALSE)
anova(genderFit1, genderFit2)
summary(genderFit2)

## Adding main effect for gender
genderFit3 <- lmer(rating ~ Gender + (1 | ID) + (1 | theme), data = data_long, REML = FALSE)
anova(genderFit2, genderFit3)
summary(genderFit3)

## Addig main effect for valence vs. arousal
genderFit4 <- lmer(rating ~ Gender + valar + (1 | ID) + (1 | theme), data = data_long, REML = FALSE)
anova(genderFit3, genderFit4)
summary(genderFit4)

## Adding interaction between gender and valar
genderFit5 <- lmer(rating ~ Gender * valar + (1 | ID) + (1 | theme), data = data_long, REML = FALSE)
anova(genderFit4, genderFit5)
summary(genderFit5)