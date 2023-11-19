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

3. Open a new bash terminal and run this into the terminal, each code for one endpoint, make sure you have install httpie on your computer. It have one parameter, texts. The texts parameter is the list of the dataset that you want to retrain or generate. Each of text is separated by '###'.

```bash
http POST http://127.0.0.1:8000/retrain texts="Twitter Scrape CIKM,Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape,https://archive.org/details/twitter_cikm_2010,Social Networks,,2009###Facebook Data,Facebook Data Scrape (2005),https://archive.org/details/oxford-2005-facebook-matrix,Social Networks,,2005###LAW graphs,Facebook Social Networks from LAW (since 2007),http://law.di.unimi.it/datasets.php,Social Networks,,2007###Foursquare from,Foursquare from UMN/Sarwat (2013),https://archive.org/details/201309_foursquare_dataset_umn,Social Networks,,2013###Twitter Scrape,Twitter Scrape Calufa May 2011,http://archive.org/details/2011-05-calufa-twitter-sql,Social Networks,,2011###Youtube Video,Youtube Video Social Graph in 2007,2008,http://netsg.cs.sfu.ca/youtubedata/,Social Networks,,2007###FBI Hate Crime 2013,FBI Hate Crime 2013 - aggregated data,https://github.com/emorisse/FBI-Hate-Crime-Statistics/tree/master/2013,Social Sciences,GitHub,2013###GSS,General Social Survey (GSS) since 1972,http://gss.norc.org,Social Sciences,,1972"
```

In the code above to use the retrain endpoint. You need to retrain the model if this is the first time you run this api.


```bash
http POST http://127.0.0.1:8000/generate texts="Twitter Scrape CIKM,Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape,https://archive.org/details/twitter_cikm_2010,Social Networks,,2009###Facebook Data,Facebook Data Scrape (2005),https://archive.org/details/oxford-2005-facebook-matrix,Social Networks,,2005###LAW graphs,Facebook Social Networks from LAW (since 2007),http://law.di.unimi.it/datasets.php,Social Networks,,2007###Foursquare from,Foursquare from UMN/Sarwat (2013),https://archive.org/details/201309_foursquare_dataset_umn,Social Networks,,2013###Twitter Scrape,Twitter Scrape Calufa May 2011,http://archive.org/details/2011-05-calufa-twitter-sql,Social Networks,,2011###Youtube Video,Youtube Video Social Graph in 2007,2008,http://netsg.cs.sfu.ca/youtubedata/,Social Networks,,2007###FBI Hate Crime 2013,FBI Hate Crime 2013 - aggregated data,https://github.com/emorisse/FBI-Hate-Crime-Statistics/tree/master/2013,Social Sciences,GitHub,2013###GSS,General Social Survey (GSS) since 1972,http://gss.norc.org,Social Sciences,,1972"
```

In the code above to use the generate endpoint. This endpoint will generate the exact same number of your parameters.


4. Deploy in AWS EC2

When you have deploy the api in AWS EC2, you need to change the host in the code above to your public ip address. You can find the public ip address in the EC2 dashboard.

If you want to run it in the background, you can use this code below on AWS EC2 terminal (Ubuntu).

a. Update 

```bash
sudo apt-get update
```

b. Install python3-pip

```bash
sudo apt install python3-pip
```

d. Install git
```bash
sudo apt install git
```

e. Clone the project from this repository:

```bash
sudo git clone path-to-this-repository
```

f. Change the directory to the project folder

```bash
cd ./distilgpt2-api-fastapi
```

g. Install git-lfs

```bash
sudo apt install git-lfs
```

h. Add global variable safe.directory
    
```bash
git config --global --add safe.directory /home/ubuntu/distilgpt2-api-fastapi
```

i. Initialize git-lfs

```bash
sudo git lfs install
```

j. Pull the model from git-lfs

```bash
sudo git lfs pull
```

k. Install pipenv

```bash
sudo pip install -r requirements.txt
```

l. Install TMUX (to run the server in the background)

```bash
sudo apt install tmux
```

m. Open tmux session with name 'distilgpt2-api-fastapi' in current directory

```bash
sudo tmux new -s distilgpt2-api-fastapi
```
or if you want to attach to the existing tmux session with name 'distilgpt2-api-fastapi'

```bash
sudo tmux attach -t distilgpt2-api-fastapi
```

n. Run the server

```bash
sudo python3 -m uvicorn generic.api:app --host 0.0.0.0 --port 8000 --reload
```

o. Stop the server

```bash
Ctrl + c
```

p. Detach from the tmux session

```bash
Ctrl + b + d
```

If you also want to delete the tmux session

```bash
Ctrl + b + x
```

q. check all tmux session

```bash
sudo tmux ls
```

r. delete tmux session

```bash
sudo tmux kill-session -t <id>
```

If you want to delete the tmux session with name 'distilgpt2-api-fastapi'

```bash
sudo tmux kill-session -t distilgpt2-api-fastapi
```

s. Don't forget to add the security group in AWS EC2 to allow All Traffic from Anywhere

## Support

Found this repo usefull? Support this effort by giving a star on GitHub, and don't be shy to collaborate with me. Meet me in [instagram](https://www.instagram.com/jibi.bm2/).

