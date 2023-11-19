## Quick Start Guide

1. Clone the project from this repository:

```bash
git clone https://github.com/JabalnurIT/api-for-pretrained-distilbertmodelformaskedlm-with-fastapi.git
```

2. Open a new bash terminal (it is recommended to use bash) and install the dependency

```bash
pipenv install --dev
```

After finish the installation, run the shell pipenv (In case you don't have pipenv, try to google how to install it)

```bash
pipenv shell
```

That bash terminal should change to pipenv terminal. After that run this code on that terminal.

```bash
chmod +x bin/start_server
bin/start_server
```

3. Download the pretrained model from this link:

```bash
chmod +x bin/download_model
bin/download_model
```

After that, put the file inside the project assets.

```bash
./assets/
```

4. Open a new bash terminal and run this into the terminal, each code for one function, make sure you have install httpie on your computer

```bash
http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] [MASK],[MASK] [MASK] and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"

http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK],british singer and writer,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
```

In the code above to use the fill_mask_text function. Don't forget to put [MASK] on it.

```bash
http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
http POST http://127.0.0.1:8000/perplexity text="1,Q42,q [MASK],british singer  and writer,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
```

In the code above to use the perplexity function. Don't put [MASK] on it.

retrain the model

```bash
http POST http://127.0.0.1:8000/retrain texts="Twitter Scrape CIKM,Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape,https://archive.org/details/twitter_cikm_2010,Social Networks,,2009###Facebook Data,Facebook Data Scrape (2005),https://archive.org/details/oxford-2005-facebook-matrix,Social Networks,,2005###LAW graphs,Facebook Social Networks from LAW (since 2007),http://law.di.unimi.it/datasets.php,Social Networks,,2007###Foursquare from,Foursquare from UMN/Sarwat (2013),https://archive.org/details/201309_foursquare_dataset_umn,Social Networks,,2013###Twitter Scrape,Twitter Scrape Calufa May 2011,http://archive.org/details/2011-05-calufa-twitter-sql,Social Networks,,2011###Youtube Video,Youtube Video Social Graph in 2007,2008,http://netsg.cs.sfu.ca/youtubedata/,Social Networks,,2007###FBI Hate Crime 2013,FBI Hate Crime 2013 - aggregated data,https://github.com/emorisse/FBI-Hate-Crime-Statistics/tree/master/2013,Social Sciences,GitHub,2013###GSS,General Social Survey (GSS) since 1972,http://gss.norc.org,Social Sciences,,1972"
```

generate dataset

```bash
http POST http://127.0.0.1:8000/generate texts="Twitter Scrape CIKM,Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape,https://archive.org/details/twitter_cikm_2010,Social Networks,,2009###Facebook Data,Facebook Data Scrape (2005),https://archive.org/details/oxford-2005-facebook-matrix,Social Networks,,2005###LAW graphs,Facebook Social Networks from LAW (since 2007),http://law.di.unimi.it/datasets.php,Social Networks,,2007###Foursquare from,Foursquare from UMN/Sarwat (2013),https://archive.org/details/201309_foursquare_dataset_umn,Social Networks,,2013###Twitter Scrape,Twitter Scrape Calufa May 2011,http://archive.org/details/2011-05-calufa-twitter-sql,Social Networks,,2011###Youtube Video,Youtube Video Social Graph in 2007,2008,http://netsg.cs.sfu.ca/youtubedata/,Social Networks,,2007###FBI Hate Crime 2013,FBI Hate Crime 2013 - aggregated data,https://github.com/emorisse/FBI-Hate-Crime-Statistics/tree/master/2013,Social Sciences,GitHub,2013###GSS,General Social Survey (GSS) since 1972,http://gss.norc.org,Social Sciences,,1972"
```

## Support

Found this repo usefull? Support this effort by giving a star on GitHub, and don't be shy to collaborate with me. Meet me in [instagram](https://www.instagram.com/jibi.bm2/).

