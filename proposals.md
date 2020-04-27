All of the three projects below will use a regression approach for predictive modeling as a default, although I acknowledge the possibility of altering that approach after EDA.  

# Bit of a stretch (most difficult data to scrape and quantify) - Trade volume b/w countries versus proximity, population, and trade law.

Sources:
Trade law: https://www.law.cornell.edu/uscode/text/15
Top trading partners: https://www.census.gov/foreign-trade/statistics/highlights/top/index.html
Most popular ports in country (to compute average distance to popular US ports)http://www.worldportsource.com/countries.php
Populations: https://www.worldometers.info/world-population/population-by-country/

Procedure: 
Control for population between countries and proximity (given the cost of transport), and investigate the relationship between trade law and trading volume between various countries and the US. 

# Next most difficult, but likely doable - NBA team analysis - can the prior season’s statistics per player be leveraged to accurately predict their regular season record and / or playoff chances?

Sources:
https://stats.nba.com/players/traditional/?sort=PTS&dir=-1&Season=1996-97&SeasonType=Regular%20Season&PerMode=Totals

Procedure:
Pull player totals’ per season over the last 23 seasons to compute individual player efficiencies. Then use beginning of the season rosters to plug in the player efficiencies (in descending order) to predict regular season wins and / or if they will make the playoffs


# Least difficult - Price of gold versus central bank funds rate and top global stock indices by volume:
Sources:
Top gdp countries, nominal and GDP: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
Central bank rates: https://www.global-rates.com/interest-rates/central-banks/central-banks.aspx
Price of gold: https://goldprice.org/

Procedure:
Investigate the relationship between central bank behaviors globally, as well as stock market performances, to make inferences about pricing behavior of gold



```python

```
