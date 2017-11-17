rm(list=ls())
library(mbest)
library(data.table)
library(stringr)
library(logging); basicConfig()
library(doParallel); registerDoParallel(6)
utils <- modules::use("utils.R")

TTS <- 4918279

members <- fread("data/raw/members.csv")
songs <- fread("data/raw/songs.csv")
train <- fread("data/raw/train.csv")
train <- merge(train, members, by="msno")
train <- merge(train, songs, by="song_id")
# ignore genre
cats <- c("city", "gender", "registered_via", "artist_name",
          "composer", "lyricist", "language")
numerics <- c("song_length", "bd", "registration_init_time",
              "expiration_date")
for (x in cats) {
  train[[x]] <- addNA(as.factor(train[[x]]))
  msg <- sprintf("%s has %d unique values", x, length(unique(train[[x]])))
  loginfo(msg)
}
train[, song_length := scales::squish(song_length, c(0, 500000))]

val <- train[TTS:nrow(train), ]
train <- train[1:(TTS-1), ]
rm(list=c("members", "songs"))

# how many songs has a user seen
counts <-train[, .N, by=msno][['N']] 
summary(counts)
hist(log(counts))

# fit a song model
formulas <- list(control="target ~ 1 + city + gender + registered_via + poly(song_length, 3)")
formulas[["bias"]] <- sprintf("%s + (1 | song_id)", formulas[["control"]])
formulas[["super"]] <- sprintf("%s + (1 + city + gender|song_id)", formulas[["control"]])
models <- lapply(formulas, utils$fit, df_train=train)
aucs <- lapply(models, utils$eval, df_test=val)
loginfo("I fit some models using song-id random effects")
loginfo(aucs)

# fit a member model
m_formulas <- list()
m_formulas[["control"]] <- 