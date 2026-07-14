install.packages("jmastats")
library(jmastats)
data("stations", package = "jmastats")

# extract lon/lat
stations$lon <- sapply(stations$geometry, function(x) x[1])
stations$lat <- sapply(stations$geometry, function(x) x[2])

# remove geometry
stations$geometry <- NULL
# clean NA
stations <- stations[!is.na(stations$lon) & !is.na(stations$lat), ]
stations$lon <- as.numeric(stations$lon)
stations$lat <- as.numeric(stations$lat)
# export
write.csv(stations, "data/jma_located_original.csv", row.names = FALSE, na = "")