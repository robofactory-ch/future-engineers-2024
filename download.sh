
if [[ `git status --porcelain` ]]; then
  echo "git repo dirty, aborting"
else
  rsync -Wave ssh pi@car.local:/home/pi/future-engineers-24/prog/ prog/
fi
