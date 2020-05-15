```python
import pymongo
import pandas as pd
from pymongo import MongoClient


import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot')
```

## For analysis, we'll begin by importing all of our data from MongoDb


```python
client = MongoClient()
db_nba= client.nba


collection_p = db_nba.player_stats
df_nba = pd.DataFrame(list(collection_p.find()))

collection_t = db_nba.team_stats
df_team = pd.DataFrame(list(collection_t.find()))

collection_r = db_nba.team_rosters
df_r = pd.DataFrame(list(collection_r.find()))
```

Let's take a high-level look at our player stats dataframe.


```python
df_nba.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13168 entries, 0 to 13167
    Data columns (total 29 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   _id                13168 non-null  object 
     1   Player_ID          13168 non-null  int64  
     2   GROUP_VALUE        13168 non-null  object 
     3   TEAM_ID            13168 non-null  int64  
     4   TEAM_ABBREVIATION  13168 non-null  object 
     5   GP                 13168 non-null  int64  
     6   W                  13168 non-null  int64  
     7   L                  13168 non-null  int64  
     8   FGM                13168 non-null  int64  
     9   FGA                13168 non-null  int64  
     10  FG_PCT             13168 non-null  float64
     11  FG3M               13168 non-null  int64  
     12  FG3A               13168 non-null  int64  
     13  FG3_PCT            13168 non-null  float64
     14  FTM                13168 non-null  int64  
     15  FTA                13168 non-null  int64  
     16  FT_PCT             13168 non-null  float64
     17  OREB               13168 non-null  int64  
     18  DREB               13168 non-null  int64  
     19  REB                13168 non-null  int64  
     20  AST                13168 non-null  int64  
     21  TOV                13168 non-null  int64  
     22  STL                13168 non-null  int64  
     23  BLK                13168 non-null  int64  
     24  BLKA               13168 non-null  int64  
     25  PF                 13168 non-null  int64  
     26  PFD                13168 non-null  int64  
     27  PTS                13168 non-null  int64  
     28  PLUS_MINUS         13168 non-null  int64  
    dtypes: float64(3), int64(23), object(3)
    memory usage: 2.9+ MB


## Looks like we can clean this up a bit. 

#### First, we don't need the '_id' column, as this is a Mongo convention to identify individual entries.
#### Second, we can rename the 'GROUP_VALUE' column to 'season', based on our knowledge of that column in builder the scrapers.
#### Lastly, let's convert all column names to lowercase for ease of indexing.


```python
df_nba.drop(columns=['_id'], inplace = True)
df_nba.rename(columns={'GROUP_VALUE':'season'}, inplace = True)
```


```python
d_lower = dict()
for i in df_nba.columns:
    d_lower[i] = i.lower()

df_nba.rename(columns=d_lower, inplace = True)
df_nba.columns
```




    Index(['player_id', 'season', 'team_id', 'team_abbreviation', 'gp', 'w', 'l',
           'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta',
           'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'blka',
           'pf', 'pfd', 'pts', 'plus_minus'],
          dtype='object')



#### The last bit of preliminary cleaning will be to get rid of all entries for the 2019-20 season, as the season was stalled, so we won't have any target values. We can do this via indexing.


```python
df_nba.drop(df_nba[df_nba['season'] == '2019-20'].index , inplace=True)
```

# We want to investigate the predictivity of regular season wins based on a team roster's individual efficiency metrics from the prior year.
### More simply, do efficiency metrics truly capture a player's contribution to his team's success.

There are a handful of efficiency metrics that are worth invesitgating here:

Martin Manley's Efficiency ('EFF') = ((Points + Rebounds + Assists + Steals + Blocks) - (Missed Field Goals + Missed Free Throws + Turnovers))/ Games Played

European Performance Index Rating ('PIR') = ((Points + Rebounds + Assists + Steals + Blocks + Fouls Drawn) - (Missed Field Goals + Missed Free Throws + Turnovers + Shots Rejected + Fouls Committed))/ Games Played

Plus-Minus Avg('+/-') = Seasonal plus-minus / Games Played : This describes the point differential for each game with a player on the floor without keeping track of more specific individual metrics. I.e., how does the score spread change when a player is in the game?

Note: We will not be considering John Hollinger's Player Efficiency Rating ('PER'): It is the most frequently used alternative, however PER is derived by a very complex calculation designed to compensate for different teams' varying style of play, among other factors, and PER scores do not differ markedly from EFF scores. Additionally, because players may change teams from year to year, the PER score from the prior year may misrepresent the player's efficiency within the context of the new team's possibly different style of play. 

This study is more so interested in an individual player's efficiency being predictive of team success, regardless of coaching and playing styles. Furthermore, looking at these raw statistics will help inform to what extent a coach should seek to opitimize all individual player efficiencies, and which particular metric is most predictive of team success.

# Data
We pulled the most recent 20 seasons of NBA player data utilizing swar's nba_api - https://github.com/swar/nba_api. 

This API pulls data from stats.nba.com. MongDB was utilized for data storage.

The various api scraping scripts can be found in the following files in this repository: nba_player_scraper.ipynb, nba_roster_scraper.ipynb, nba_team_scraper.ipynb

A full procedural breakdown of the methods used in this study can be found in the data_and_plots.ipynb file in this repository.




# Let's calculate EFF, PIR, and +/- for each player on a per season basis. 


```python
df_nba.columns
```




    Index(['player_id', 'season', 'team_id', 'team_abbreviation', 'gp', 'w', 'l',
           'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta',
           'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'blka',
           'pf', 'pfd', 'pts', 'plus_minus'],
          dtype='object')




```python
"""EFF = ((Points + Rebounds + Assists + Steals + Blocks) - Missed Field Goals - Missed Free Throws - Turnovers))/ Games Played"""

df_nba['eff'] = ((df_nba['pts']+ df_nba['reb']+df_nba['ast']+df_nba['stl']+df_nba['blk']) \
                 - (df_nba['fga']-df_nba['fgm']) \
                 - (df_nba['fta']-df_nba['ftm']) \
                 - df_nba['tov']) / df_nba['gp']


"""PIR = ((Points + Rebounds + Assists + Steals + Blocks + Fouls Drawn) - Missed Field Goals - Missed Free Throws - Turnovers - Shots Rejected - Fouls Committed))/ Games Played"""
df_nba['pir'] = ((df_nba['pts']+ df_nba['reb']+df_nba['ast']+df_nba['stl']+df_nba['blk']+df_nba['pfd']) \
                 - (df_nba['fga']-df_nba['fgm']) \
                 - (df_nba['fta']-df_nba['ftm']) \
                 - df_nba['tov'] \
                 - df_nba['blka'] \
                 - df_nba['pf']) / df_nba['gp']

df_nba['+/-'] = df_nba['plus_minus'] / df_nba['gp']
```

## Distributions of our EFF, PIR, and +/- metrics.


```python
def plot_stat_dist(stat):
    
    
    """
    PARAMETERS:
    stat - string, column title of statistic of interest
    
    RETURNS:
    
    None - plots distribution of statistic
    """
    #Calculate all statistic averages and plot histogram to inform distribution

    mean = round(np.mean(df_nba[stat]),3)
    median = round(np.median(df_nba[stat]),3)

    fig, ax = plt.subplots(1,2,figsize=(12,5))

    #Histogram 
    plt.suptitle(f'Yearly Player {stat.upper()} Distribution from 1998-99 Season to Present')

    # Index to highest frequency bin
    counts, bins = ax[0].hist(df_nba[stat], bins = 40)[0],ax[0].hist(df_nba[stat], bins = 40)[1]

    idx_max = np.argmax(counts)
    upper = bins[idx_max+1]
    lower = bins[idx_max]
    mode = round(np.mean([upper,lower]),2)

    ax[0].axvline(mode, color = 'y', label = f'Mode {stat.upper()}: ~{mode}')
    ax[0].axvline(median, color = 'g', label = f'Median {stat.upper()}: {median}')
    ax[0].axvline(mean, color = 'b', label = f'Mean {stat.upper()}: {mean}')

    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel(f'{stat.upper()}')
    ax[0].legend()
    
    #Boxplot
    ax[1].boxplot(df_nba[stat].astype('float'))
    ax[1].set_ylabel(f'{stat.upper()}')

    
    plt.tight_layout()
    
    return None
```


```python
plot_stat_dist('eff')
#plt.savefig('eff_dist')
```


![png](output_17_0.png)



```python
plot_stat_dist('pir')
#plt.savefig('pir_dist')
```


![png](output_18_0.png)



```python
plot_stat_dist('+/-')
#plt.savefig('pm_dist')
```


![png](output_19_0.png)


## Now let's group the player data by player_id and season, so that we can pull the statistic values from a prior year, and use those as features for a team based on roster. 

First, we'll group the dataframe by players and season. We do this instead of grouping by teams and season, to allow for us to account for roster changes season over season.

We can then reference player_id values for the target season to ensure we're selecting the right features (i.e, player statistics from the previous year, regardless of team affiliation). 


```python
# First we'll group just by player so we can check if a particular season is relevant to them. 

df_by_player = df_nba.groupby(['player_id'])

```


```python
# Here we group by both player and season so that we can pull particular efficiency scores per player per year.

df_by_p_s = df_nba.groupby(['player_id','season'])

```

## Now, we can iterate through team rosters by season to pull the prior year's efficiency metrics per player. 

Investigate roster dataframe to determine how to properly pull ['eff'] from player dataframe.


```python
df_r.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 630 entries, 0 to 629
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   _id     630 non-null    object
     1   team    630 non-null    int64 
     2   season  630 non-null    object
     3   roster  630 non-null    object
    dtypes: int64(1), object(3)
    memory usage: 19.8+ KB


Again, we can drop the '_id' column, as this is just a mongo convention


```python
df_r.drop(columns = ['_id'], inplace = True)
```


```python
df_r.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>season</th>
      <th>roster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1610612737</td>
      <td>1998-99</td>
      <td>[673, 1533, 1544, 87, 1516, 3, 1852, 111, 770,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1610612738</td>
      <td>1998-99</td>
      <td>[692, 952, 1477, 1800, 344, 368, 35, 65, 72, 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1610612739</td>
      <td>1998-99</td>
      <td>[692, 226, 682, 1510, 1538, 916, 198, 1507, 18...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1610612740</td>
      <td>1998-99</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1610612741</td>
      <td>1998-99</td>
      <td>[457, 82, 699, 1869, 1736, 1757, 54, 1522, 160...</td>
    </tr>
  </tbody>
</table>
</div>



## For sake of iteration, we'll need to get all unique team IDs and seasons into two separate arrays.

We'll then group the roster dataframe by team and season, and use Pandas' get_group() method to pull the roster for a given season. We'll then iterate through the roster to select player efficiency scores from the prior year in the df_by_player dataframe via player_id and season. 


```python
teams = np.unique(df_team['TeamID'])
seasons = np.unique(df_nba['season'])

teams,seasons
```




    (array([1610612737, 1610612738, 1610612739, 1610612740, 1610612741,
            1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
            1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
            1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
            1610612757, 1610612758, 1610612759, 1610612760, 1610612761,
            1610612762, 1610612763, 1610612764, 1610612765, 1610612766]),
     array(['1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03',
            '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
            '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
            '2015-16', '2016-17', '2017-18', '2018-19'], dtype=object))



## Group roster dataframe by team_id and season


```python
df_r = df_r.groupby(['team','season'])

```

Because the line above just converted the variable df_r to a Pandas' groupby object, we'll need to call get_group() going forward to view and pull data.

Let's use our 'teams' and 'seasons' lists from above to do just that.

### Because we are interested in a pre-season prediction of wins based on team roster, we want to ensure that we're only pulling player data from players who were on the roster at the beginning of the year. We can check for multiple entries for a player within a given year, and then ensure that we only upload a statistic on behalf of a player to the team on which they began a season.


```python
#Generalized function for generating a feature matrix
def gen_X(teams,seasons,stat,n_p=17):  
    
    """
    PARAMETERS:
    
    teams - list of team IDs
    season - list of seasons
    stat - string of column title in player dataframe for statistic of interest
    n_p - int, number of players on a roster to consider. Defaults to 17 as this is the max number of active players
    a roster may have at a given time
    
    RETURNS:
    
    Feature matrix to be used for predicting wins and playoff rankings of teams
    """
    
    X = []

    # Index into each team
    for t in teams:

        # Index into each season
        for idx, s in enumerate(seasons):

            # We only care about prior year stats, and we can't index into the prior year if idx is 0
            if idx > 0:

                # create a row for our features array
                row = [int(t),s]

                #accumulator for efficiency scores
                lst_stat = []

                # select roster and loop through player_ids to pull efficiency scores from prior year in df_by_player
                roster = list(df_r.get_group((t.item(),s))['roster'])

                for p in roster[0]:

                    #for sake of testing the code, make sure a player_id is in our dataset
                    if p in df_nba['player_id'].to_numpy():

                        #Check a given season is relevant to a player
                        if seasons[idx-1] in df_by_player.get_group(p)['season'].to_numpy():

                            # verify a player started the season with team
                            # the -1 index is because team order is reverse chronilogical

                            if t == df_by_p_s.get_group((p,s))['team_id'].to_numpy()[-1]:          

                                """Select a player's efficiency score from the prior season, because 
                                we want to predict outcomes of current year. 
                                The index 0 ensures we pull cummulative stats for players who 
                                spent the prior season on multiple teams."""

                                stat_ = df_by_p_s.get_group((p,seasons[idx-1]))[stat].to_numpy()[0]
                                lst_stat.append(int(stat_))
                

            #Create accumulator array to account for differing sizes in rosters
                final_stat = np.zeros(19)
                for idx,stat_ in enumerate(lst_stat):
                    final_stat[idx] = int(stat_)

                """Sort efficiency scores in descending order such that we are comparing players 
                of equal team hierarchical rank within feature columns"""
                # take n_p best players per efficiency statistic
                final_stat = np.sort(final_stat)[::-1][:n_p]

                row = np.concatenate((row,final_stat), axis = None)

                X.append(row)
    
    # Generate dataframe and properly name columns
    df_X = pd.DataFrame(X)
    
    col_names = dict()

    for idx,col in enumerate(df_X.columns):
        if idx>1:
            col_names[idx] = f'{stat.upper()}_rank{idx-1}'
        elif idx == 0:
            col_names[idx] = 'team_id'
        else:
            col_names[idx] = 'season'
    
    df_X.rename(columns = col_names, inplace=True)
    
    return df_X
```


```python
df_X = gen_X(teams,seasons,'eff',10)
```


```python
df_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_id</th>
      <th>season</th>
      <th>EFF_rank1</th>
      <th>EFF_rank2</th>
      <th>EFF_rank3</th>
      <th>EFF_rank4</th>
      <th>EFF_rank5</th>
      <th>EFF_rank6</th>
      <th>EFF_rank7</th>
      <th>EFF_rank8</th>
      <th>EFF_rank9</th>
      <th>EFF_rank10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1610612737</td>
      <td>1998-99</td>
      <td>21.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1610612737</td>
      <td>1999-00</td>
      <td>20.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1610612737</td>
      <td>2000-01</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1610612737</td>
      <td>2001-02</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1610612737</td>
      <td>2002-03</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



## We have our features matrix (X) above. Now onto our targets. 


```python
df_team.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 624 entries, 0 to 623
    Data columns (total 14 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   _id               624 non-null    object
     1   SeasonID          624 non-null    object
     2   TeamID            624 non-null    int64 
     3   TeamCity          624 non-null    object
     4   TeamName          624 non-null    object
     5   Conference        624 non-null    object
     6   ConferenceRecord  624 non-null    object
     7   PlayoffRank       624 non-null    int64 
     8   ClinchIndicator   620 non-null    object
     9   Division          624 non-null    object
     10  DivisionRecord    624 non-null    object
     11  DivisionRank      624 non-null    int64 
     12  WINS              624 non-null    int64 
     13  LOSSES            624 non-null    int64 
    dtypes: int64(5), object(9)
    memory usage: 68.4+ KB


This study is interested in just the wins per team per season, so let's pullout the pertinent columns.


```python
df_team = df_team[['SeasonID','TeamID','WINS','LOSSES']]
df_team.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeasonID</th>
      <th>TeamID</th>
      <th>WINS</th>
      <th>LOSSES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21998</td>
      <td>1610612753</td>
      <td>33</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21998</td>
      <td>1610612762</td>
      <td>37</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21998</td>
      <td>1610612754</td>
      <td>33</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21998</td>
      <td>1610612759</td>
      <td>37</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21998</td>
      <td>1610612748</td>
      <td>33</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



## Note: There are only 50 games played in the 1998 season, so we'll need to transform that data to estimate the 82 game win total so it's consistent with the rest of our data.


```python
df_team['w'] = ((df_team['WINS'] / (df_team['WINS']+df_team['LOSSES'])) * 82).round()
```


```python
df_team.rename(columns = {'TeamID': 'team_id'}, inplace= True)
```

Now we'll drop the 'WINS' and 'LOSSES' columns as they're extraneous.


```python
df_team.drop(columns = ['WINS','LOSSES'], inplace = True)
```


```python
np.unique(df_team['SeasonID']),np.unique(df_nba['season'])
```




    (array(['21998', '21999', '22000', '22001', '22002', '22003', '22004',
            '22005', '22006', '22007', '22008', '22009', '22010', '22011',
            '22012', '22013', '22014', '22015', '22016', '22017', '22018'],
           dtype=object),
     array(['1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03',
            '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
            '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
            '2015-16', '2016-17', '2017-18', '2018-19'], dtype=object))



### We should also update our SeasonID columns so that its formatting is the same as that of 'season' in our df_X.


```python
season1 = np.unique(df_nba['season'])[1:]
season2 = np.unique(df_team['SeasonID'])

d = dict()
for s1, s2 in zip(season1,season2):
    d[s2] = s1
    
d
```




    {'21998': '1998-99',
     '21999': '1999-00',
     '22000': '2000-01',
     '22001': '2001-02',
     '22002': '2002-03',
     '22003': '2003-04',
     '22004': '2004-05',
     '22005': '2005-06',
     '22006': '2006-07',
     '22007': '2007-08',
     '22008': '2008-09',
     '22009': '2009-10',
     '22010': '2010-11',
     '22011': '2011-12',
     '22012': '2012-13',
     '22013': '2013-14',
     '22014': '2014-15',
     '22015': '2015-16',
     '22016': '2016-17',
     '22017': '2017-18',
     '22018': '2018-19'}




```python
df_team['season'] = [d[s] for s in df_team['SeasonID']]
```


```python
df_team.drop(columns = 'SeasonID', inplace = True)
```


```python
# Let's get everthing nice and chronological
df_team.sort_values(by = ['team_id','season'], inplace=True)
```


```python
df_team.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_id</th>
      <th>w</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1610612737</td>
      <td>51.0</td>
      <td>1998-99</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1610612737</td>
      <td>28.0</td>
      <td>1999-00</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1610612737</td>
      <td>25.0</td>
      <td>2000-01</td>
    </tr>
    <tr>
      <th>109</th>
      <td>1610612737</td>
      <td>33.0</td>
      <td>2001-02</td>
    </tr>
    <tr>
      <th>136</th>
      <td>1610612737</td>
      <td>35.0</td>
      <td>2002-03</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_win_dist():
    
    
    """
    PARAMETERS:
    None
    
    RETURNS:
    
    None - plots distribution of regular season wins
    """
    #Calculate all statistic averages and plot histogram to inform distribution

    mean = round(np.mean(df_team['w']),3)
    median = round(np.median(df_team['w']),3)

    fig, ax = plt.subplots(1,2,figsize=(12,5))

    #Histogram 
    plt.suptitle(f'Yearly Regular Season Wins Distribution from 1998-99 Season to Present')

    # Index to highest frequency bin
    counts, bins = ax[0].hist(df_team['w'], bins = 40)[0],ax[0].hist(df_team['w'], bins = 40)[1]

    idx_max = np.argmax(counts)
    upper = bins[idx_max+1]
    lower = bins[idx_max]
    mode = round(np.mean([upper,lower]),2)

    ax[0].axvline(mode, color = 'y', label = f'Mode: ~{mode}')
    ax[0].axvline(median, color = 'g', label = f'Median: {median}')
    ax[0].axvline(mean, color = 'b', label = f'Mean: {mean}')

    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Wins')
    ax[0].legend()
    
    #Boxplot
    ax[1].boxplot(df_team['w'].astype('float'))
    ax[1].set_ylabel('Wins')

    
    plt.tight_layout()
    
    return None
```


```python
plot_win_dist()
#plt.savefig('win_dist')
```


![png](output_53_0.png)



```python
df_team.info(),df_X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 624 entries, 6 to 611
    Data columns (total 3 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   team_id  624 non-null    int64  
     1   w        624 non-null    float64
     2   season   624 non-null    object 
    dtypes: float64(1), int64(1), object(1)
    memory usage: 19.5+ KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 630 entries, 0 to 629
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   team_id     630 non-null    object
     1   season      630 non-null    object
     2   EFF_rank1   630 non-null    object
     3   EFF_rank2   630 non-null    object
     4   EFF_rank3   630 non-null    object
     5   EFF_rank4   630 non-null    object
     6   EFF_rank5   630 non-null    object
     7   EFF_rank6   630 non-null    object
     8   EFF_rank7   630 non-null    object
     9   EFF_rank8   630 non-null    object
     10  EFF_rank9   630 non-null    object
     11  EFF_rank10  630 non-null    object
    dtypes: object(12)
    memory usage: 59.2+ KB





    (None, None)



## Uh-oh, looks like our shapes are different for our feature dataframe and our targets. Let's investigate the shape discrepancy.


```python
np.unique(df_X['team_id'], return_counts = True)
```




    (array(['1610612737', '1610612738', '1610612739', '1610612740',
            '1610612741', '1610612742', '1610612743', '1610612744',
            '1610612745', '1610612746', '1610612747', '1610612748',
            '1610612749', '1610612750', '1610612751', '1610612752',
            '1610612753', '1610612754', '1610612755', '1610612756',
            '1610612757', '1610612758', '1610612759', '1610612760',
            '1610612761', '1610612762', '1610612763', '1610612764',
            '1610612765', '1610612766'], dtype=object),
     array([21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
            21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]))




```python
np.unique(df_team['team_id'], return_counts = True)
```




    (array([1610612737, 1610612738, 1610612739, 1610612740, 1610612741,
            1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
            1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
            1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
            1610612757, 1610612758, 1610612759, 1610612760, 1610612761,
            1610612762, 1610612763, 1610612764, 1610612765, 1610612766]),
     array([21, 21, 21, 17, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
            21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 19]))



## Team_id 1610612740 only appears 17 times, and team_id 1610612766 only appears 19 times.

Perhaps this has to do with a team becoming defunct or moving locations. Let's figure out which franchises belong to these team_id values


```python
np.unique(df_nba.loc[df_nba['team_id']==1610612740]['team_abbreviation'], return_counts = True), df_team.loc[df_team['team_id']==1610612740]['season']
```




    ((array(['NOH', 'NOK', 'NOP'], dtype=object), array([173,  34, 158])),
     124    2002-03
     153    2003-04
     202    2004-05
     223    2005-06
     253    2006-07
     266    2007-08
     307    2008-09
     345    2009-10
     366    2010-11
     412    2011-12
     440    2012-13
     466    2013-14
     489    2014-15
     527    2015-16
     553    2016-17
     575    2017-18
     619    2018-19
     Name: season, dtype: object)




```python
np.unique(df_nba.loc[df_nba['team_id']==1610612766]['team_abbreviation'], return_counts = True), df_team.loc[df_team['team_id']==1610612766]['season']
```




    ((array(['CHA', 'CHH'], dtype=object), array([286,  73])),
     17     1998-99
     35     1999-00
     68     2000-01
     93     2001-02
     201    2004-05
     229    2005-06
     255    2006-07
     286    2007-08
     312    2008-09
     336    2009-10
     372    2010-11
     413    2011-12
     441    2012-13
     456    2013-14
     494    2014-15
     515    2015-16
     554    2016-17
     583    2017-18
     611    2018-19
     Name: season, dtype: object)



#### As suspected, both of the above franchises have undergone relocations in the most recent 20 years.

#### For New Orleans, we're missing the following seasons' data: 1998-99,1999-2000, 2000-01, 2001-02. 
#### For Charlotte, we're missing: 2002-03,2003-04

#### So, let's ensure that we're only pairing feature rows with records that actually have wins and playoff rank data.


```python
# store wins and playoff ranks pertinent to each season by team as targets.

def get_y(df_X,df_y,y_col):
    
    """
    PARAMETERS:
    
    df_X: Pandas DataFrame containing feature values
    df_y: Pandas DataFrame containing target values
    y_col: list of column names of desired target values
    
    RETURNS:
    DataFrame with both features and target values to be used for model fitting. 
    
    """

    for y in y_col:
        
        arr_y = []
        
        for t,s in zip(df_X['team_id'],df_X['season']):
            # verify a season is pertinent to a given team
            if s not in df_y.loc[df_y['team_id']==int(t)]['season'].to_numpy():
                y_ = None
                arr_y.append(y_)

            else:
                y_ = df_y.loc[(df_y['team_id']==int(t)) & (df_y['season'] == s)][y].to_numpy()[0]
                arr_y.append(y_)
        
        arr_y = np.array(arr_y)
        
        df_X[y] = arr_y
    
    #get rid of rows with null values in target columns
    df_X.dropna(inplace = True)
    
    return df_X
```


```python
df_X = gen_X(teams,seasons,'eff',10)
```


```python
X_eff = get_y(df_X,df_team,['w'])
```


```python
X_eff.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_id</th>
      <th>season</th>
      <th>EFF_rank1</th>
      <th>EFF_rank2</th>
      <th>EFF_rank3</th>
      <th>EFF_rank4</th>
      <th>EFF_rank5</th>
      <th>EFF_rank6</th>
      <th>EFF_rank7</th>
      <th>EFF_rank8</th>
      <th>EFF_rank9</th>
      <th>EFF_rank10</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1610612737</td>
      <td>1998-99</td>
      <td>21.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1610612737</td>
      <td>1999-00</td>
      <td>20.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1610612737</td>
      <td>2000-01</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1610612737</td>
      <td>2001-02</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1610612737</td>
      <td>2002-03</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



## Let's begin fitting and experimenting with the number of features to see if there's an optimal number of best player individual efficiency scores to include per team. A range of 5 to 17 feels approriate as 5 players need to be on the floor at a given time, and there can be a maximum of 17 players on the active roster of a team.

## We'll look at individual EFF, PIR, and +/- scores

Because none of our data is categorical, and we can't ensure a linear relationship between our features and targets, a GradientBoost Regressor feels appropriate for regular season wins predictions.


```python
import sklearn
```


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
```

### Score models by number of features and plot


```python
def score_n_features(stat):
    
    fig,ax = plt.subplots()
    """
    PARAMETERS:
    stat - string, column title of statistic of interest
    
    RETURNS:
    None - plots Gradient Boosting model scores associated with number of features
    """

    lst_score = []
    lst_n = []
    for n in range(5,18):

        #generate features and targets
        df_X = gen_X(teams,seasons,stat,n)
        X = get_y(df_X,df_team,['w'])
        
        y_w = X['w']
        
        X = X.drop(columns=['team_id','season','w'])
        
        #fit and score model
        gb_w = GradientBoostingRegressor()
        score = np.mean(cross_val_score(gb_w,X,y_w,cv=5))

        lst_score.append(round(score,2))
        lst_n.append(n)
    
    max_score = np.max(lst_score)
    max_feat = lst_n[np.argmax(lst_score)]
    ax.plot(lst_n,lst_score)
    ax.annotate(f'({max_feat},{max_score})',(max_feat,max_score))
    ax.set_title(f'Number of Features vs. Model Score: {stat.upper()}')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('R^2 Score')
    
    return None
    
```

## EFF results


```python
eff_ind = score_n_features('eff')
#plt.savefig('eff_score_nfeat')
```


![png](output_72_0.png)



```python
eff_ind
```

## PIR results


```python
pir_ind = score_n_features('pir')
#plt.savefig('pir_score_nfeat')
```


![png](output_75_0.png)



```python
pir_ind
```

## +/- results


```python
pm_ind = score_n_features('+/-')
#plt.savefig('pm_score_nfeat')
```


![png](output_78_0.png)



```python
pm_ind
```

## Individual Player Features Model Summary

#### Optimized Individual Player EFF: 14 players, cross validated R^2 of 0.17 | Optimized Individual Player PIR: 13 players, R^2 of 0.05 | Optimized Individual Player +/-:  15 players, R^2 of 0.37

When creating hierarchical rankings within teams, individual Player +/- is the most informative to a team's regular season success within the context of a Gradient Boosting model built on individual player values for a particular efficiency metric. Perhaps this is because +/- does the best job of capturing a player's ability to synergize with his teammates as it prioritizes a team's success when a player is on the floor, versus the individual player's stat-line. For example, if a player puts up 50 points, 20 assists, and 20 rebounds on a given night, but his team loses and his +/- score is negative, one could argue his amazing statline was just 'empty production' as it didn't do his team any good in terms of winning the game.


## So we know which feature level and individual efficiency metric (15,+/-) performs the best for the above models. Let's take a peak at feature importances to see if we can pinpoint which features are most informative.

### Feature importances can be a bit misleading. While they describe the extent to which a feature helps a model make accurate predictions, they don't describe directionality, nor correlation of the features to a particular target. 



```python
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
```

### Below is a plotting function that calculates feature importance for a given model and plots the scatter plot comparing the feature to the target value (regular season wins, in this case).

#### For sake of testing the relationship of each feature to the target, we'll conduct a Pearson R test for correlation within each feature.
#### The Pearson R has the following assumptions: 
1) Parallel level of measurement: both variables are continuous (check). 
2) Related pairs: every wins value is paired to a player efficiency score (check). 
3) Absence of outliers: let's run through once and then alter if we need to. 
4) Linearity: visually test with scatter plot and Pearson R p-value. We'll use an alpha of 0.05 as our threshold of statistical significance. 


```python
def plot_imp(stat,n_feat):
    
    
    cols = 4
    if n_feat%cols == 0:
        rows = int(n_feat/cols)
    else:
        rows = (n_feat//cols) + 1

    fig, ax = plt.subplots(rows, cols, sharex = True, sharey =True, figsize=(15,12))
    """
    PARAMETERS:
    stat - string, Column title for statistic of interest
    n_feat - int, number of features to be included in model
    
    RETURNS:
    -Array of feature titles sorted by importance
    -Plots boxplot distributions of features in order of features in order of feature importance.
    """
    
    df_X = gen_X(teams,seasons,stat,n_feat)
    X = get_y(df_X,df_team,['w'])
    y_w = X['w']
    X = X.drop(columns=['team_id','season','w'])
    
    #fit and get feature scores of model
    gb_w = GradientBoostingRegressor()
    gb_w.fit(X,y_w)
    imp = permutation_importance(gb_w,X,y_w)
    feat_imp = imp['importances_mean']
    feat_imp
    
    # rank feature importances and plot their underlying distribution
    order = np.argsort(feat_imp)[::-1]
    for idx,col in enumerate(X.columns[order]):
        
        
        r, p = pearsonr(X[col].astype('float'),y_w)
        mean_x = round(np.mean(df_nba[stat].astype('float')),2)
        max_y = np.max(y_w)

        
        if rows == 1:
            ax[idx%cols].scatter(X[col].astype('float'), y_w, alpha = 0.5) 

            ax[idx%cols].set_title(f'{col}, Importance: {round(feat_imp[order][idx],3)}', fontsize = 10)
            ax[idx%cols].set_ylabel('Regular Season Wins')
            ax[idx%cols].set_xlabel(f'Player {stat.upper()}')
            ax[idx%cols].text(mean_x,max_y, f'Pearson r: {round(r,2)} | p-valule:{round(p,3)}',fontsize = 8)
        else:
            ax[idx//cols,idx%cols].scatter(X[col].astype('float'), y_w, alpha = 0.5) 

            ax[idx//cols,idx%cols].set_title(f'{col}, Importance: {round(feat_imp[order][idx],3)}', fontsize = 10)
            if idx%cols == 0:
                ax[idx//cols,idx%cols].set_ylabel('Regular Season Wins')
            ax[idx//cols,idx%cols].set_xlabel(f'Player {stat.upper()}')
            ax[idx//cols,idx%cols].text(mean_x,max_y, f'Pearson r: {round(r,2)} | p-valule:{round(p,3)}',fontsize = 8)
            

    plt.tight_layout()
       
    return X.columns[order]
    
```


```python
pm_imp = plot_imp('+/-',17)
#plt.savefig('pm_imp')
```


![png](output_85_0.png)


Feature importance was not parallel to the hierarchical rankings of player +/- within each team. This suggests that bench players (ranks 6 and below) play an important role in predicting a team's success for a given season. 

Across all features, individual player +/- scores were positively correlated to regular season wins to a statistically significant degree. Does this hold true for both EFF and PIR?



```python
eff_imp = plot_imp('eff',14)
#plt.savefig('eff_imp')
```


![png](output_87_0.png)



```python
pir_imp = plot_imp('pir',13)
#plt.savefig('pir_imp')
```


![png](output_88_0.png)


### Looks like for the most part, linear relationships exist between EFF/PIR and wins. 
Exceptions: EFF - rank 13, rank 14 | PIR - rank 13, rank 12

### But it seems the Pearson r values for Player +/- and wins is higher across rankings than those of EFF and PIR. Let's confirm.  


```python
def pearson_r(stat):
    
    """
    PARAMETERS:
    stat - string, column title for statistic of interest
    
    RETURNS:
    list of column titles and their corresponding Pearson R test results
    """
    df_X = gen_X(teams,seasons,stat)
    X = get_y(df_X,df_team,['w'])
    y_w = X['w']
    X = X.drop(columns=['team_id','season','w'])
    
    res = []
    for idx,col in enumerate(X.columns):
        
        r, p = pearsonr(X[col].astype('float'),y_w)
        res.append(f'{col} - r: {round(r,2)}, p: {round(p,3)}')
    
    return res

```


```python
eff_pr = pearson_r('eff')
pir_pr = pearson_r('pir')
pm_pr = pearson_r('+/-')
```


```python
for idx in range(0,17):
    print(f'{eff_pr[idx]} | {pir_pr[idx]} | {pm_pr[idx]}')
```

    EFF_rank1 - r: 0.46, p: 0.0 | PIR_rank1 - r: 0.41, p: 0.0 | +/-_rank1 - r: 0.52, p: 0.0
    EFF_rank2 - r: 0.48, p: 0.0 | PIR_rank2 - r: 0.42, p: 0.0 | +/-_rank2 - r: 0.59, p: 0.0
    EFF_rank3 - r: 0.46, p: 0.0 | PIR_rank3 - r: 0.36, p: 0.0 | +/-_rank3 - r: 0.61, p: 0.0
    EFF_rank4 - r: 0.38, p: 0.0 | PIR_rank4 - r: 0.32, p: 0.0 | +/-_rank4 - r: 0.58, p: 0.0
    EFF_rank5 - r: 0.32, p: 0.0 | PIR_rank5 - r: 0.27, p: 0.0 | +/-_rank5 - r: 0.57, p: 0.0
    EFF_rank6 - r: 0.27, p: 0.0 | PIR_rank6 - r: 0.21, p: 0.0 | +/-_rank6 - r: 0.53, p: 0.0
    EFF_rank7 - r: 0.24, p: 0.0 | PIR_rank7 - r: 0.2, p: 0.0 | +/-_rank7 - r: 0.47, p: 0.0
    EFF_rank8 - r: 0.2, p: 0.0 | PIR_rank8 - r: 0.15, p: 0.0 | +/-_rank8 - r: 0.37, p: 0.0
    EFF_rank9 - r: 0.18, p: 0.0 | PIR_rank9 - r: 0.17, p: 0.0 | +/-_rank9 - r: 0.29, p: 0.0
    EFF_rank10 - r: 0.19, p: 0.0 | PIR_rank10 - r: 0.18, p: 0.0 | +/-_rank10 - r: 0.27, p: 0.0
    EFF_rank11 - r: 0.17, p: 0.0 | PIR_rank11 - r: 0.15, p: 0.0 | +/-_rank11 - r: 0.3, p: 0.0
    EFF_rank12 - r: 0.11, p: 0.008 | PIR_rank12 - r: 0.08, p: 0.051 | +/-_rank12 - r: 0.34, p: 0.0
    EFF_rank13 - r: 0.06, p: 0.105 | PIR_rank13 - r: 0.04, p: 0.365 | +/-_rank13 - r: 0.43, p: 0.0
    EFF_rank14 - r: 0.04, p: 0.361 | PIR_rank14 - r: 0.04, p: 0.357 | +/-_rank14 - r: 0.46, p: 0.0
    EFF_rank15 - r: 0.03, p: 0.467 | PIR_rank15 - r: 0.02, p: 0.668 | +/-_rank15 - r: 0.49, p: 0.0
    EFF_rank16 - r: 0.02, p: 0.558 | PIR_rank16 - r: 0.03, p: 0.393 | +/-_rank16 - r: 0.49, p: 0.0
    EFF_rank17 - r: 0.01, p: 0.789 | PIR_rank17 - r: -0.0, p: 0.927 | +/-_rank17 - r: 0.49, p: 0.0


Across the board, individual player +/-  is more strongly correlated with regular season wins than EFF and PIR

## However, the top R^2 of 0.37 from our Gradient Boosting models (+/-, 15 players) still leaves plenty to be desired. Let's pivot to some aggregated team statistics and see if those are more informative. 


```python
#Generalized function for generating a feature matrix with average team statistics based on roster
def gen_team_X(teams,seasons,stats):  
    
    """
    PARAMETERS:
    
    teams - list of team IDs
    season - list of seasons
    stats - list of strings: column titles in player dataframe for statistic of interest
    
    RETURNS:
    
    Feature matrix to be used for predicting wins and playoff rankings of teams
    """
    
    X = []

    # Index into each team
    for t in teams:

        # Index into each season
        for idx, s in enumerate(seasons):

            # We only care about prior year stats, and we can't index into the prior year if idx is 0
            if idx > 0:

                # create a row for our features array
                row = [int(t),s]

                for stat in stats:    
                    
                    #accumulator for efficiency scores
                    lst_stat = []

                    # select roster and loop through player_ids to pull efficiency scores from prior year in df_by_player
                    roster = list(df_r.get_group((t.item(),s))['roster'])

                    for p in roster[0]:

                        #for sake of testing the code, make sure a player_id is in our dataset
                        if p in df_nba['player_id'].to_numpy():

                            #Check a given season is relevant to a player
                            if seasons[idx-1] in df_by_player.get_group(p)['season'].to_numpy():

                                # verify a player started the season with team
                                # the -1 index is because team order is reverse chronilogical

                                if t == df_by_p_s.get_group((p,s))['team_id'].to_numpy()[-1]:          

                                    """Select a player's efficiency score from the prior season, because 
                                    we want to predict outcomes of current year. 
                                    The index 0 ensures we pull cummulative stats for players who 
                                    spent the prior season on multiple teams."""
                                    stat_ = df_by_p_s.get_group((p,seasons[idx-1]))[stat].to_numpy()[0]
                                    lst_stat.append(int(stat_))
                    
                    if len(lst_stat)<1:
                        final_stat = None
                    #take mean of statistic and append to row
                    else:
                        final_stat = np.mean(lst_stat)
                    
                    row = np.concatenate((row,final_stat), axis = None)



                # add roster average statistics to df
                X.append(row)
    
    # Generate dataframe and properly name columns
    df_X = pd.DataFrame(X)
    
    col_names = dict()

    for idx,col in enumerate(df_X.columns):
        if idx>1:
            col_names[idx] = stats[idx-2]
        elif idx == 0:
            col_names[idx] = 'team_id'
        else:
            col_names[idx] = 'season'
    
    df_X.rename(columns = col_names, inplace=True)
    
    return df_X
```


```python
df_X_team = gen_team_X(teams,seasons,['eff','pir','+/-'])
```


```python
df_X_team.shape
```




    (630, 5)




```python
get_y(df_X_team,df_team,['w'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_id</th>
      <th>season</th>
      <th>eff</th>
      <th>pir</th>
      <th>+/-</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1610612737</td>
      <td>1998-99</td>
      <td>11.0</td>
      <td>8.3</td>
      <td>0.5</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1610612737</td>
      <td>1999-00</td>
      <td>9.1</td>
      <td>6.4</td>
      <td>0.4</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1610612737</td>
      <td>2000-01</td>
      <td>7.3076923076923075</td>
      <td>5.153846153846154</td>
      <td>-1.6923076923076923</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1610612737</td>
      <td>2001-02</td>
      <td>8.4</td>
      <td>5.866666666666666</td>
      <td>-1.2</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1610612737</td>
      <td>2002-03</td>
      <td>9.692307692307692</td>
      <td>6.846153846153846</td>
      <td>-2.3846153846153846</td>
      <td>35</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>625</th>
      <td>1610612766</td>
      <td>2014-15</td>
      <td>10.153846153846153</td>
      <td>9.692307692307692</td>
      <td>-0.6153846153846154</td>
      <td>33</td>
    </tr>
    <tr>
      <th>626</th>
      <td>1610612766</td>
      <td>2015-16</td>
      <td>9.071428571428571</td>
      <td>8.571428571428571</td>
      <td>-0.7142857142857143</td>
      <td>48</td>
    </tr>
    <tr>
      <th>627</th>
      <td>1610612766</td>
      <td>2016-17</td>
      <td>9.714285714285714</td>
      <td>9.5</td>
      <td>0.5</td>
      <td>36</td>
    </tr>
    <tr>
      <th>628</th>
      <td>1610612766</td>
      <td>2017-18</td>
      <td>12.272727272727273</td>
      <td>12.0</td>
      <td>0.09090909090909091</td>
      <td>36</td>
    </tr>
    <tr>
      <th>629</th>
      <td>1610612766</td>
      <td>2018-19</td>
      <td>9.75</td>
      <td>9.833333333333334</td>
      <td>-0.5</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 6 columns</p>
</div>




```python
def score_team(stats):
    
    """
    PARAMETERS:
    stats - list of strings, column titles pertinent to statistics of interest
    
    RETURNS:
    None - prints Gradient Boosting model scores associated with number of features
    """

    #generate features and targets
    df_X = gen_team_X(teams,seasons,stats)
    X = get_y(df_X,df_team,['w'])

    y_w = X['w']

    X = X.drop(columns=['team_id','season','w'])

    #fit and score model
    gb_w = GradientBoostingRegressor()
    score = round(np.mean(cross_val_score(gb_w,X,y_w,cv=5)),3)

    print(f'Aggregate team statistics model score: {score}')
    
    return None
```


```python
score_team(['eff','pir','+/-'])
```

    Aggregate team statistics model score: 0.394


### So we've improved about .024 from our individual +/- model before to R^2 =~0.394.
While this improvement is slight, it suggest that a team's average efficiency if more informative to regular season success than the roster's individual efficiency values.

### Time for another round of feature importance analysis.


```python
def plot_imp_team(stats):
    
    n_feat = len(stats)
    
    cols = 3
    if n_feat%cols == 0:
        rows = int(n_feat/cols)
    else:
        rows = (n_feat//cols) + 1
        
    fig, ax = plt.subplots(rows, cols, sharey = True, figsize=(12,5))
    """
    PARAMETERS:
    stat - list of strings, column titles for statistics of interest

    
    RETURNS:
    -Array of feature titles sorted by importance
    -Plots boxplot distributions of features in order of features in order of feature importance.
    """
    
    df_X = gen_team_X(teams,seasons,stats)
    X = get_y(df_X,df_team,['w'])

    y_w = X['w']

    X.drop(columns=['team_id','season','w'],inplace=True)
    
    #fit and get feature scores of model
    gb_w = GradientBoostingRegressor()
    gb_w.fit(X,y_w)
    imp = permutation_importance(gb_w,X,y_w)
    feat_imp = imp['importances_mean']
    feat_imp
    
    # rank feature importances and plot their underlying distribution
    order = np.argsort(feat_imp)[::-1]
    for idx,col in enumerate(X.columns[order]):
        
        
        r, p = pearsonr(X[col].astype('float'),y_w)
        mean_x = round(np.mean(X[col].astype('float')),2)
        max_y = np.max(y_w)

        
        if rows == 1:
            ax[idx%cols].scatter(X[col].astype('float'), y_w, alpha = 0.5) 

            ax[idx%cols].set_title(f'{col}, Importance: {round(feat_imp[order][idx],3)}', fontsize = 10)
            ax[idx%cols].set_ylabel('Regular Season Wins')
            ax[idx%cols].set_xlabel(f'Player {col.upper()}')
            ax[idx%cols].text(mean_x,max_y, f'Pearson r: {round(r,2)} | p-valule:{round(p,3)}',fontsize = 8)
        else:
            ax[idx//cols,idx%cols].scatter(X[col].astype('float'), y_w, alpha = 0.5) 

            ax[idx//cols,idx%cols].set_title(f'{col}, Importance: {round(feat_imp[order][idx],3)}', fontsize = 10)
            if idx%cols == 0:
                ax[idx//cols,idx%cols].set_ylabel('Regular Season Wins')
            ax[idx//cols,idx%cols].set_xlabel(f'Player {col.upper()}')
            ax[idx//cols,idx%cols].text(mean_x,max_y, f'Pearson r: {round(r,2)} | p-valule:{round(p,3)}',fontsize = 8)
            

    plt.tight_layout()
       
    return X.columns[order]
```


```python
team_agg_imp = plot_imp_team(['eff','pir','+/-'])
#plt.savefig('team_agg_imp')
```


![png](output_102_0.png)


## Feature Importance by Average Team Stats Summary
Consistent with the prior models focusing on one efficiency metric, all three metrics show a significant linear relatiopnship with regular season wins. Again, +/- shines as the most predictive of a team's seasonal outcome. More than twice as much as EFF and four times as much as PIR. There also appears to be a parallel relationship between an efficiency metric's Pearson r value and feature importance for this particular model.

## Let's try and beat an R^2 of 0.394.
Idea: because an inclusion of all three efficiency metrics seemed to improve our model above, let's fit the model including the individual player efficiency score for all three metrics. 



```python
# Generate dataframes for all three metrics
    
df_X_eff = gen_X(teams,seasons,'eff')
df_X_pir = gen_X(teams,seasons,'pir')
df_X_pm = gen_X(teams,seasons,'+/-')
```


```python
# join dataframes

df_m1 = pd.merge(df_X_eff,df_X_pir, on = ['team_id','season'], how='inner')
df_all_ind = pd.merge(df_m1,df_X_pm, on = ['team_id','season'], how='inner')
```


```python
X = get_y(df_all_ind,df_team,['w'])
y_w = X['w']
X = X.drop(columns=['team_id','season','w'])
```


```python
# Fit and score

gb_w = GradientBoostingRegressor()
print(f'Cross Validation R^2 Score including all individual efficiency scores across metrics: {round(np.mean(cross_val_score(gb_w,X,y_w,cv=5)),3)}')
```

    Cross Validation R^2 Score including all individual efficiency scores across metrics: 0.37


### Shoot, our score actually dropped slightly from just the team averages for each efficiency metric, and it performed no better than fitting on individual +/- scores from the prior season. 
The average efficiency metric scores within teams is more informative to predicting regular season results across teams than all individual metric scores.
Let's try one more, super simple model. A roster's average +/- based on the prior season.


```python
score_team(['+/-'])
```

    Aggregate team statistics model score: 0.313


The results above show we're below the best R^2 value of 0.394 (average team efficiency, all three metrics), so it does seem that in tandem, EFF and PIR can help further inform the model working alongside +/-.

# Conclusion

Within the context of Gradient Boosting regression modeling, the NBA efficiency metric of a player's prior season +/- is the most predictive of regular season success based on a team's roster at the beginning of a season and, both on an individual player and team average basis in comparison to prior season EFF and prior season PIR.

Across all levels in player hierarchy for a given team, +/- is more strongly correlated to regular season wins than EFF and PIR. 

This suggests that +/- may be the most comprehensive measurement of a player's efficiency in contributing to a team winning games. While +/- takes into account scoring differentials during a player's time on the court, EFF and PIR only take into account the changes in the individual player's statline. 

Possible implication: while further study is needed, focusing on optimizing the +/- metric across players during a season could help inform coaching styles and may lead to improved success for a team.

Bottom line: Of all the modeling approaches taken in this study, the best cross validated R^2 score we generated was 0.394. This suggests the prior season's efficiency metrics of an NBA roster on their own do a mediocre job of capturing the variance in the number of regular season wins. 

# Further Study:
-Employ different model types and see if the relationship between efficiency metrics is upheld.

-Investigate the predicitivity of a player's propensity for injury or being traded, and how that in interaction with efficiency metrics may effect regular season wins. 

-Compare efficiency metrics' predictivity versus more conventional NBA statistics, e.g., points scored, field goal percentage, minutes played


```python

```
