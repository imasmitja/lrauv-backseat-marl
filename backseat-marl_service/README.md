## Linux systemd service file installation notes

The provided service file and run scripts assume the backseat app has been cloned (or sym-linked) on the SBC at `/LRAUV/backseat-marl`. 

Execute the following as root:

```bash
# copy the service file into the systemd folder:
sudo cp backseat-marl.service /etc/systemd/system/.

# enable the service 
sudo systemctl daemon-reload
sudo systemctl enable backseat-marl.service
sudo systemctl start backseat-marl.service

# check
sudo systemctl status backseat-marl.service

# stop
sudo systemctl stop backseat-marl.service
```

Tested on Jetson Nano running Ubuntu 18.04
