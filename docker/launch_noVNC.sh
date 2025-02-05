docker network create ros
docker run -d --rm --net=ros \
   --env="DISPLAY_WIDTH=2500" --env="DISPLAY_HEIGHT=1500" --env="RUN_XTERM=no" \
   --name=novnc -p=8080:8080 \
   theasp/novnc:latest