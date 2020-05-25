library(caret)
library(tidyverse)
library(mlbench)

data <- read_csv('~/Downloads/aflstats/stats.csv')

team_name <- data %>% 
  mutate(new_index = paste(Season, Player)) %>% 
  select(new_index, Team)

#aggregate features per player per season
features <- data %>% 
  mutate(new_index = paste(Season, Player)) %>% 
  group_by(new_index) %>% 
  summarise('Height' = max(Height),
            'Weight' = max(Weight),
            'Position' = max(Position),
            'Disposals' = sum(Disposals),
            'Kicks' = sum(Kicks),
            'Marks' = sum(Marks),
            'Handballs' = sum(Handballs),
            'Goals' = sum(Goals),
            'Behinds' = sum(Behinds),
            'Hitouts' = sum(Hitouts),
            'Tackles' = sum(Tackles),
            'Rebound50s' = sum(Rebound50s),
            'Inside50s' = sum(Inside50s),
            'Clearances' = sum(Clearances),
            'Clangers' = sum(Clangers),
            'FreesFor' = sum(FreesFor),
            'FreesAgainst' = sum(FreesAgainst),
            'BrownlowVotes' = sum(BrownlowVotes),
            'ContendedPossessions' = sum(ContendedPossessions),
            'UncontendedPossessions' = sum(UncontendedPossessions),
            'ContestedMarks' = sum(ContestedMarks),
            'MarksInside50' = sum(MarksInside50),
            'OnePercenters' = sum(OnePercenters),
            'Bounces' = sum(Bounces),
            'GoalsAssists' = sum(GoalAssists),
            'PercentagePlayed' = mean(PercentPlayed)) %>% 
  mutate(Position = if_else(Position == 'Midfield, Ruck', 'Ruck', Position)) %>% ##Edit 1 row to avoid introducing new category in test data for 2012
  column_to_rownames(var = 'new_index')

years <- c('2012', '2013', '2014', '2015', '2016', '2017', '2018')

create_test_set <- function(x) {
  
  test <- features %>% 
    rownames_to_column() %>% 
    slice(grep(x, rowname)) %>% 
    column_to_rownames(var = 'rowname') %>% 
    select(-BrownlowVotes)

}

create_train_set <- function(x) {
  
  train_years <- years[!years %in% x]
  years_str <- paste(train_years, collapse = '|')
  
  train <- features %>% 
  rownames_to_column() %>% 
  slice(grep(years_str, rowname)) %>% 
  column_to_rownames(var = 'rowname')
  
}

test_list <- lapply(years, create_test_set)
train_list <- lapply(years, create_train_set)

result_list <- list()

for (i in seq(7)) {
  
  model <- train(BrownlowVotes ~ .,
        data = train_list[[i]],
        method = "lm",
        metric = "RMSE")
  
  predictions <- predict(model, test_list[[i]])
  
  predictions <- as.data.frame(predictions) %>% 
    rownames_to_column()
  
  results <- features %>% 
    rownames_to_column() %>% 
    inner_join(team_name, by = c('rowname' = 'new_index')) %>% 
    select(rowname, BrownlowVotes, Team) %>% 
    distinct() %>% 
    inner_join(predictions, by = 'rowname') %>%
    rename(player = rowname) %>% 
    mutate(delta = abs(BrownlowVotes - predictions)) %>% 
    arrange(desc(predictions)) %>% 
    rownames_to_column()
  
  result_list[[i]] <- results
}

joined_results <- plyr::join_all(result_list, by = 'rowname', type = 'left')

joined_results <- joined_results[2:36] 

joined_results %>%
  head(12) %>%
  select(starts_with('player')) %>% View()

#output feature importance
ggplot(varImp(model))

model

result_list_perclub <- list()

#Ranking top players for each club
for (i in seq(7)) {
  
  result_list_perclub[[i]] <- result_list[[i]] %>% 
    group_by(Team) %>% 
    mutate(rank = dense_rank(desc(predictions))) %>% 
    arrange(Team, rank) %>% 
    filter(rank == 1) %>% #More than just top player because of illegability of suspended players
    select(-rowname)
  
}

result_list_perclub[[1]] %>% 
  left_join(result_list_perclub[[2]], by = 'Team') %>% View()


