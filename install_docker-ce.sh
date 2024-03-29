#!/usr/bin/env bash

sudo add-apt-repository ppa:criu/ppa --yes

sudo apt update

sudo apt --fix-broken install

sudo apt remove docker docker-engine docker.io containerd runc

sudo apt --yes install \
	apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        criu 

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

sudo groupadd -f docker

sudo apt --yes install docker-ce docker-ce-cli containerd.io

sudo usermod -aG docker $USER

newgrp docker 


