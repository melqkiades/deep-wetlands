
library(readr)
library(dplyr)
library(ggplot2)
library(plotly)
library(lubridate)

# df <- read.table(text = x, sep ="\t", header = TRUE, stringsAsFactors = FALSE)
df <- read_csv("/tmp/paper_water_estimates_kavsjon.csv",
               col_types = cols(
                 Extension = col_double(),
                 Date = col_date(format = "%Y-%m-%d")
               ))
df <- rename(df, Extension = Extension, Date = Date)
df <- select(df, Extension, Date)
df <- mutate(df, Date= as.Date(Date, format= "%Y-%m-%d"))

# ndwi_df <- read.csv("/tmp/paper_ndwi_water_estimates.csv", select = c("Water", "Date"), col.names = c("Extension", "Date"))
ndwi_df <- read_csv("/tmp/paper_ndwi_water_estimates_kavsjon.csv",
               col_types = cols(
                 Water = col_double(),
                 Date = col_date(format = "%Y-%m-%d")
               ))
ndwi_df <- rename(ndwi_df, Extension = Water, Date = Date)
ndwi_df <- select(ndwi_df, Extension, Date)
ndwi_df <- mutate(ndwi_df, Date= as.Date(Date, format= "%Y-%m-%d"))

# df <- summarize(summary_variable = sum(value))

# df <- group_by(month = lubridate::floor_date(df.Date, "month"))

# df <- df %>%
#   mutate(month = format(Date, "%m"), year_month = format(Date, "%Y-%m"), year = format(Date, "%Y")) %>%
#   group_by(year_month) %>%
#   summarise(min = min(Extension), max = max(Extension), mean = mean(Extension))

df_grouped <- df %>%
  mutate(month = format(Date, "%m"), year_month = format(Date, "%Y-%m"), year = format(Date, "%Y")) %>%
  group_by(year_month) %>%
  summarise(min = min(Extension), max = max(Extension), mean = mean(Extension))

df_grouped$year_month <- as.Date(paste0(df_grouped$year_month, "-01"), format = "%Y-%m-%d")


ndwi_df_grouped <- ndwi_df %>%
  mutate(month = format(Date, "%m"), year_month = format(Date, "%Y-%m"), year = format(Date, "%Y")) %>%
  group_by(year_month) %>%
  summarise(min = min(Extension), max = max(Extension), mean = mean(Extension))

ndwi_df_grouped$year_month <- as.Date(paste0(ndwi_df_grouped$year_month, "-01"), format = "%Y-%m-%d")

df_grouped <- mutate(df_grouped, method = "SAR")
ndwi_df_grouped <- mutate(ndwi_df_grouped, method = "NDWI")

df_merged <- bind_rows(df_grouped, ndwi_df_grouped)

# Define the function to get the start and end dates of the season
get_season_dates <- function(date) {
  year <- year(date)
  season <- get_season(date)
  if (season == "Spring") {
    start_date <- as.Date(paste0(year, "-03-01"))
    end_date <- as.Date(paste0(year, "-05-31"))
  } else if (season == "Summer") {
    start_date <- as.Date(paste0(year, "-06-01"))
    end_date <- as.Date(paste0(year, "-08-31"))
  } else if (season == "Fall") {
    start_date <- as.Date(paste0(year, "-09-01"))
    end_date <- as.Date(paste0(year, "-11-30"))
  } else {
    start_date <- as.Date(paste0(year, "-12-01"))
    if (year == max(year(df$date))) {
      end_date <- as.Date("2022-02-28")
    } else {
      end_date <- as.Date(paste0(year+1, "-02-28"))
    }
  }
  return(c(start_date, end_date))
}


# Define the function to get the season
get_season <- function(date) {
  month <- month(date)
  if (month >= 3 & month <= 5) {
    return("Spring")
  } else if (month >= 6 & month <= 8) {
    return("Summer")
  } else if (month >= 9 & month <= 11) {
    return("Fall")
  } else {
    return("Winter")
  }
}

# Define the function to get the start and end dates of the season
get_season_dates <- function(date) {
  year <- year(date)
  season <- get_season(date)
  if (season == "Spring") {
    start_date <- as.Date(paste0(year, "-03-01"))
    end_date <- as.Date(paste0(year, "-05-31"))
  } else if (season == "Summer") {
    start_date <- as.Date(paste0(year, "-06-01"))
    end_date <- as.Date(paste0(year, "-08-31"))
  } else if (season == "Fall") {
    start_date <- as.Date(paste0(year, "-09-01"))
    end_date <- as.Date(paste0(year, "-11-30"))
  } else {
    start_date <- as.Date(paste0(year, "-12-01"))
    if (year == max(year(df$date))) {
      end_date <- as.Date("2022-02-28")
    } else {
      end_date <- as.Date(paste0(year+1, "-02-28"))
    }
  }
  return(c(start_date, end_date))
}

# Add the season, start date, and end date columns to the data frame
df_merged$season <- sapply(df_merged$year_month, get_season)
df_merged[, c("start_date", "end_date")] <- t(sapply(df_merged$year_month, get_season_dates))

df_merged$start_date <- as.Date(df_merged$start_date, origin = "1970-01-01")
df_merged$end_date <- as.Date(df_merged$end_date, origin = "1970-01-01")


ggplot(df_grouped, aes(x = year_month, y = min)) +
  geom_line() +
  xlab("Date") +
  ylab("Min Extension") +
  ggtitle("Line Chart")


ggplot(df_grouped, aes(x = year_month, y = min)) +
  geom_bar(stat = "identity") +
  xlab("Date") +
  ylab("Min Extension") +
  ggtitle("Bar Plot")

ggplot(df_merged, aes(x = year_month, y = min, fill = method)) +
  geom_bar(stat = "identity", position="dodge") +
  xlab("Category") +
  ylab("Value") +
  ggtitle("Bar Chart")

ggplot(df_merged, aes(x = year_month, y = min, color = method)) +
  geom_point() +
  xlab("Date") +
  ylab("Min Extension") +
  ggtitle("Scatter Plot")


ggplot(data = df_merged, aes(x = year_month, y = min)) +
  # geom_rect(aes(xmin = year_month, xmax = max_date, fill = Season,
  #               ymin = -Inf, ymax = Inf),
  #           fill = c("lightgreen", "lightsalmon", "lightgoldenrod"), alpha = 0.5) +
  geom_rect(
    data = df_merged, # or rect_data for actual season end boundary
    mapping = aes(
      xmin = start_date,
      xmax = end_date,
      ymin = -Inf,
      ymax = Inf,
      fill = season
    ), alpha = 0.5
  ) +
  geom_point(aes(color = method), size = 3) +
  geom_line(aes(group = method, color = method)) +
  scale_color_manual(values = c("red", "blue", "green"), name="Method") +
  scale_fill_manual(values=c("lightgoldenrod","lightgreen","lightsalmon"), name="Season") +
  labs(x="Date", y="Water extension (m2)")

  # + theme_set(theme_gray() + theme(legend.key=element_blank()))
  # + theme_set(theme_bw() + theme(legend.key=element_blank()))


# ggplotly(p)

ggplot(df_merged, aes(x = year_month, y = min)) +
  geom_rect(aes(xmin = year_month, xmax = dplyr::lead(year_month), ymin = -0.5, ymax = Inf, fill = group),
            alpha = 0.5) +
  geom_point(size = 2.5) +
  theme_classic()


df_merged

