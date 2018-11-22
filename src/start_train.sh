if [ -z "$(ps -ef | grep '[p]ython train')" ]
then
    nohup python train.py $1 &
fi
